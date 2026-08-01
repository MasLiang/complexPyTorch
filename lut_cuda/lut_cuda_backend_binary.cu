 #include <torch/extension.h>
 #include <cuda.h>
 #include <cuda_runtime.h>
 #include <ATen/cuda/Atomic.cuh>
 #include <ATen/AccumulateType.h> 
 
 // ------------------------------------------------------------------------
 // 前向传播 Kernel：极速二值查表实现 (O(1) 复杂度)
 // ------------------------------------------------------------------------
 template <typename scalar_t>
 __global__ void lut_interp6_forward_kernel(
     const scalar_t* __restrict__ x, 
     const scalar_t* __restrict__ w, 
     scalar_t* __restrict__ y,
     int64_t total_ol, int64_t BR, int64_t L, int og) 
 {
     int64_t ol_start = static_cast<int64_t>(blockIdx.x) * gridDim.y + blockIdx.y;
     int64_t ol_stride = static_cast<int64_t>(gridDim.x) * gridDim.y;
     
     int64_t br_start = static_cast<int64_t>(blockIdx.z) * blockDim.x + threadIdx.x;
     int64_t stride = static_cast<int64_t>(gridDim.z) * blockDim.x;
 
     using acc_t = at::acc_type<scalar_t, true>;
     __shared__ acc_t s_w[64];
 
     for (int64_t ol_idx = ol_start; ol_idx < total_ol; ol_idx += ol_stride) {
         // 加载 LUT 权重到共享内存
         for (int i = threadIdx.x; i < 64; i += blockDim.x) 
             s_w[i] = static_cast<acc_t>(w[ol_idx * 64 + i]);
         __syncthreads();
 
         int64_t o_idx = ol_idx / L; 
         int64_t l_idx = ol_idx % L;
         int64_t g_idx = o_idx / og;
 
         for (int64_t br = br_start; br < BR; br += stride) {
             int64_t x_base = (g_idx * L + l_idx) * 6 * BR + br;
             
             // 终极优化：将 6 个离散浮点值打包成 1 个整型索引 (0~63)
             int current_idx = 0;
             #pragma unroll
             for (int j = 0; j < 6; ++j) {
                 // 判断是否激活 (>0.5 即可，因为输入是 0.0 或 1.0)
                 if (static_cast<float>(x[x_base + j * BR]) > 0.5f) {
                     current_idx |= (1 << (5 - j));
                 }
             }
             
             // O(1) 极速查表赋值
             y[ol_idx * BR + br] = static_cast<scalar_t>(s_w[current_idx]);
         }
         __syncthreads(); 
     }
 }
 
 // ------------------------------------------------------------------------
 // 反向传播 Kernel：极速差分求导与共享内存聚合
 // ------------------------------------------------------------------------
 template <typename scalar_t>
 __global__ void lut_interp6_backward_kernel(
     const scalar_t* __restrict__ grad_y, 
     const scalar_t* __restrict__ x, 
     const scalar_t* __restrict__ w,
     float* __restrict__ grad_x_fp32, 
     float* __restrict__ grad_w_fp32,
     int64_t total_ol, int64_t BR, int64_t L, int og)
 {
     int64_t ol_start = static_cast<int64_t>(blockIdx.x) * gridDim.y + blockIdx.y;
     int64_t ol_stride = static_cast<int64_t>(gridDim.x) * gridDim.y;
     
     using acc_t = at::acc_type<scalar_t, true>;
     __shared__ acc_t s_w[64];
     __shared__ float s_grad_w[64];
 
     for (int64_t ol_idx = ol_start; ol_idx < total_ol; ol_idx += ol_stride) {
         // 初始化共享内存
         for (int i = threadIdx.x; i < 64; i += blockDim.x) {
             s_w[i] = static_cast<acc_t>(w[ol_idx * 64 + i]);
             s_grad_w[i] = 0.0f;
         }
         __syncthreads(); 
 
         int64_t o_idx = ol_idx / L; 
         int64_t l_idx = ol_idx % L;
         int64_t g_idx = o_idx / og;
 
         int64_t br_start = static_cast<int64_t>(blockIdx.z) * blockDim.x;
         int64_t stride = static_cast<int64_t>(gridDim.z) * blockDim.x;
 
         for (int64_t br_base = br_start; br_base < BR; br_base += stride) {
             int64_t br = br_base + threadIdx.x;
             bool valid = (br < BR);
 
             if (valid) {
                 float dy = static_cast<float>(grad_y[ol_idx * BR + br]);
                 int64_t x_base = (g_idx * L + l_idx) * 6 * BR + br;
                 
                 // 1. O(1) 拿到当前输入映射的 LUT 索引
                 int current_idx = 0;
                 #pragma unroll
                 for (int j = 0; j < 6; ++j) {
                     if (static_cast<float>(x[x_base + j * BR]) > 0.5f) {
                         current_idx |= (1 << (5 - j));
                     }
                 }
 
                 // 2. 权重梯度：直接命中目标，只需一次 AtomicAdd
                 atomicAdd(&s_grad_w[current_idx], dy);

                 // 3. 输入梯度：降维为 O(6) 的极速差分求导
                 #pragma unroll
                 for (int k = 0; k < 6; ++k) {
                     int pos = 5 - k; 
                     int mask = 1 << pos;
                     
                     // 获取当前位为 0 和为 1 时的表项索引
                     int idx_0 = current_idx & (~mask);
                     int idx_1 = current_idx | mask;
                     
                     // 偏导数就是两者之差，没有任何冗余乘法
                     float dx_val = static_cast<float>(s_w[idx_1] - s_w[idx_0]);
                     
                     gpuAtomicAdd(&grad_x_fp32[x_base + k * BR], dy * dx_val);
                 }
             }
         }
         __syncthreads(); 
         
         // 汇聚当前 block 的权重梯度到 Global Memory
         for (int i = threadIdx.x; i < 64; i += blockDim.x) {
             if (s_grad_w[i] != 0.0f) {
                 gpuAtomicAdd(&grad_w_fp32[ol_idx * 64 + i], s_grad_w[i]);
             }
         }
         __syncthreads(); 
     }
 }
 
 // ------------------------------------------------------------------------
 // Host 端封装 (未修改，接口保持绝对兼容)
 // ------------------------------------------------------------------------
 torch::Tensor forward_cuda(torch::Tensor x, torch::Tensor w, int groups) {
     const int64_t out_num = w.size(0);
     const int64_t L = w.size(1);
     const int64_t BR = x.size(3);
 
     auto y = torch::empty({out_num, L, BR}, x.options());
     dim3 blocks(std::min((int)out_num, 65535), std::min((int)L, 65535), std::min((int)((BR + 63) / 64), 1024));
 
     AT_DISPATCH_FLOATING_TYPES_AND2(at::ScalarType::Half, at::ScalarType::BFloat16, x.scalar_type(), "lut_forward", [&] {
         lut_interp6_forward_kernel<scalar_t><<<blocks, 64>>>(
             x.data_ptr<scalar_t>(), w.data_ptr<scalar_t>(), y.data_ptr<scalar_t>(), 
             out_num * L, BR, L, static_cast<int>(out_num / groups)
         );
     });
     return y;
 }
 
 std::vector<torch::Tensor> backward_cuda(torch::Tensor grad_y, torch::Tensor x, torch::Tensor w, int groups) {
     const int64_t out_num = w.size(0);
     const int64_t L = w.size(1);
     const int64_t BR = x.size(3);
 
     auto grad_x_fp32 = torch::zeros_like(x, x.options().dtype(torch::kFloat32));
     auto grad_w_fp32 = torch::zeros_like(w, w.options().dtype(torch::kFloat32));
 
     dim3 blocks(std::min((int)out_num, 65535), std::min((int)L, 65535), std::min((int)((BR + 63) / 64), 1024));
 
     AT_DISPATCH_FLOATING_TYPES_AND2(at::ScalarType::Half, at::ScalarType::BFloat16, x.scalar_type(), "lut_backward", [&] {
         lut_interp6_backward_kernel<scalar_t><<<blocks, 64>>>(
             grad_y.data_ptr<scalar_t>(), x.data_ptr<scalar_t>(), w.data_ptr<scalar_t>(), 
             grad_x_fp32.data_ptr<float>(), grad_w_fp32.data_ptr<float>(), 
             out_num * L, BR, L, static_cast<int>(out_num / groups)
         );
     });
 
     return {grad_x_fp32.to(x.scalar_type()), grad_w_fp32.to(w.scalar_type())};
 }
 
 PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
     m.def("forward", &forward_cuda, "LUT 6 Forward");
     m.def("backward", &backward_cuda, "LUT 6 Backward");
 }

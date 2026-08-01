#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <ATen/cuda/Atomic.cuh>
#include <ATen/AccumulateType.h> 

// ------------------------------------------------------------------------
// 前向传播 Kernel：级联插值实现
// ------------------------------------------------------------------------
template <typename scalar_t>
__global__ void lut_interp6_forward_kernel(
    const scalar_t* __restrict__ x, 
    const scalar_t* __restrict__ w, 
    scalar_t* __restrict__ y,
    int64_t total_ol, int64_t BR, int64_t L, int og) 
{
    // 修复 Bug 2: 加入 Grid-Stride Loop，防止超出 65535 限制导致静默丢弃
    int64_t ol_start = static_cast<int64_t>(blockIdx.x) * gridDim.y + blockIdx.y;
    int64_t ol_stride = static_cast<int64_t>(gridDim.x) * gridDim.y;
    
    int64_t br_start = static_cast<int64_t>(blockIdx.z) * blockDim.x + threadIdx.x;
    int64_t stride = static_cast<int64_t>(gridDim.z) * blockDim.x;

    using acc_t = at::acc_type<scalar_t, true>;
    __shared__ acc_t s_w[64];

    for (int64_t ol_idx = ol_start; ol_idx < total_ol; ol_idx += ol_stride) {
        for (int i = threadIdx.x; i < 64; i += blockDim.x) 
            s_w[i] = static_cast<acc_t>(w[ol_idx * 64 + i]);
        __syncthreads();

        int64_t o_idx = ol_idx / L; 
        int64_t l_idx = ol_idx % L;
        int64_t g_idx = o_idx / og;

        for (int64_t br = br_start; br < BR; br += stride) {
            int64_t x_base = (g_idx * L + l_idx) * 6 * BR + br;
            acc_t xp[6];
            #pragma unroll
            for (int j = 0; j < 6; ++j) xp[j] = static_cast<acc_t>(x[x_base + j * BR]);

            acc_t v[64];
            #pragma unroll
            for (int i = 0; i < 64; ++i) v[i] = s_w[i];

            #pragma unroll
            for (int d = 0; d < 6; ++d) {
                #pragma unroll
                for (int i = 0; i < (1 << (5 - d)); ++i) 
                    v[i] = v[2 * i] * (1.0 - xp[5-d]) + v[2 * i + 1] * xp[5-d];
            }
            y[ol_idx * BR + br] = static_cast<scalar_t>(v[0]);
        }
        __syncthreads(); // 确保下一轮 ol_idx 加载 s_w 前同步
    }
}

// ------------------------------------------------------------------------
// 反向传播 Kernel：强制 FP32 累加保护
// ------------------------------------------------------------------------
template <typename scalar_t>
__global__ void lut_interp6_backward_kernel(
    const scalar_t* __restrict__ grad_y, 
    const scalar_t* __restrict__ x, 
    const scalar_t* __restrict__ w,
    float* __restrict__ grad_x_fp32, // 修复 Bug 1: 强制传入 FP32 影子缓冲
    float* __restrict__ grad_w_fp32,
    int64_t total_ol, int64_t BR, int64_t L, int og)
{
    int64_t ol_start = static_cast<int64_t>(blockIdx.x) * gridDim.y + blockIdx.y;
    int64_t ol_stride = static_cast<int64_t>(gridDim.x) * gridDim.y;
    
    int lane_id = threadIdx.x & 31;
    using acc_t = at::acc_type<scalar_t, true>;
    __shared__ acc_t s_w[64];
    __shared__ float s_grad_w[64]; // 使用 float 累加权重梯度

    for (int64_t ol_idx = ol_start; ol_idx < total_ol; ol_idx += ol_stride) {
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

            acc_t dy = valid ? static_cast<acc_t>(grad_y[ol_idx * BR + br]) : 0;
            int64_t x_base = valid ? (g_idx * L + l_idx) * 6 * BR + br : 0;
            
            acc_t xp[6] = {0}, xn[6] = {1, 1, 1, 1, 1, 1};
            if (valid) {
                #pragma unroll
                for (int j = 0; j < 6; ++j) {
                    xp[j] = static_cast<acc_t>(x[x_base + j * BR]);
                    xn[j] = 1.0 - xp[j];
                }
            }

            acc_t prob[64];
            #pragma unroll
            for (int i = 0; i < 64; ++i) {
                acc_t p = 1.0;
                #pragma unroll
                for (int j = 0; j < 6; ++j) p *= ((i >> (5 - j)) & 1) ? xp[j] : xn[j];
                prob[i] = p;

                acc_t dw = dy * p;
                #pragma unroll
                for (int offset = 16; offset > 0; offset /= 2) {
                    dw += __shfl_down_sync(0xffffffff, dw, offset);
                }
                
                if (lane_id == 0) {
                    gpuAtomicAdd(&s_grad_w[i], static_cast<float>(dw));
                }
            }

            if (valid) {
                #pragma unroll
                for (int k = 0; k < 6; ++k) {
                    acc_t dx_val = 0;
                    int pos = 5 - k; 
                    int mask = 1 << pos;
                    #pragma unroll
                    for (int c = 0; c < 32; ++c) {
                        int i0 = ((c >> pos) << (pos + 1)) | (c & (mask - 1));
                        int i1 = i0 | mask;
                        dx_val += (prob[i0] + prob[i1]) * (s_w[i1] - s_w[i0]);
                    }
                    // 核心修复：在高精度的 FP32 缓冲上做 AtomicAdd，杜绝由于 FP16 引起的梯度消失
                    gpuAtomicAdd(&grad_x_fp32[x_base + k * BR], static_cast<float>(dy * dx_val));
                }
            }
        }
        __syncthreads(); 
        for (int i = threadIdx.x; i < 64; i += blockDim.x) {
            if (s_grad_w[i] != 0.0f) {
                gpuAtomicAdd(&grad_w_fp32[ol_idx * 64 + i], s_grad_w[i]);
            }
        }
        __syncthreads(); 
    }
}

// ------------------------------------------------------------------------
// Host 端封装
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

    // 强制在 Host 端分配 Float32 影子缓冲，完全规避原子加法的精度丢失
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

    // 计算完成后，安全平滑地降回原始精度（FP16/BF16/FP32），还给 PyTorch
    return {grad_x_fp32.to(x.scalar_type()), grad_w_fp32.to(w.scalar_type())};
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward_cuda, "LUT 6 Forward");
    m.def("backward", &backward_cuda, "LUT 6 Backward");
}

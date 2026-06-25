#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <ATen/cuda/Atomic.cuh>
#include <ATen/AccumulateType.h> 

// ------------------------------------------------------------------------
// 前向传播 Kernel：多线性级联插值实现
// ------------------------------------------------------------------------
template <typename scalar_t, int K>
__global__ void lut_interp_K_floating_forward_kernel(
    const scalar_t* __restrict__ x, 
    const scalar_t* __restrict__ w, 
    scalar_t* __restrict__ y,
    int64_t total_ol, int64_t BR, int64_t L, int og) 
{
    constexpr int LUT_SIZE = 1 << K;

    int64_t ol_start = static_cast<int64_t>(blockIdx.x) * gridDim.y + blockIdx.y;
    int64_t ol_stride = static_cast<int64_t>(gridDim.x) * gridDim.y;
 
    int64_t br_start = static_cast<int64_t>(blockIdx.z) * blockDim.x + threadIdx.x;
    int64_t stride = static_cast<int64_t>(gridDim.z) * blockDim.x;

    using acc_t = at::acc_type<scalar_t, true>;
    __shared__ acc_t s_w[LUT_SIZE];

    for (int64_t ol_idx = ol_start; ol_idx < total_ol; ol_idx += ol_stride) {
        for (int i = threadIdx.x; i < LUT_SIZE; i += blockDim.x) 
            s_w[i] = static_cast<acc_t>(w[ol_idx * LUT_SIZE + i]);
        __syncthreads();

        int64_t o_idx = ol_idx / L; 
        int64_t l_idx = ol_idx % L;
        int64_t g_idx = o_idx / og;

        for (int64_t br = br_start; br < BR; br += stride) {
            int64_t x_base = (g_idx * L + l_idx) * K * BR + br;
            acc_t xp[K];
            #pragma unroll
            for (int j = 0; j < K; ++j) xp[j] = static_cast<acc_t>(x[x_base + j * BR]);

            acc_t v[LUT_SIZE];
            #pragma unroll
            for (int i = 0; i < LUT_SIZE; ++i) v[i] = s_w[i];

            // 🌟 核心：编译器自动展开的多维线性插值折减
            #pragma unroll
            for (int d = 0; d < K; ++d) {
                #pragma unroll
                for (int i = 0; i < (1 << ((K - 1) - d)); ++i) {
                    v[i] = v[2 * i] * (1.0 - xp[(K - 1) - d]) + v[2 * i + 1] * xp[(K - 1) - d];
                }
            }
            y[ol_idx * BR + br] = static_cast<scalar_t>(v[0]);
        }
        __syncthreads(); 
    }
}

// ------------------------------------------------------------------------
// 反向传播 Kernel：强制 FP32 累加保护与多项式求偏导
// ------------------------------------------------------------------------
template <typename scalar_t, int K>
__global__ void lut_interp_K_floating_backward_kernel(
    const scalar_t* __restrict__ grad_y, 
    const scalar_t* __restrict__ x, 
    const scalar_t* __restrict__ w,
    float* __restrict__ grad_x_fp32, 
    float* __restrict__ grad_w_fp32,
    int64_t total_ol, int64_t BR, int64_t L, int og)
{
    constexpr int LUT_SIZE = 1 << K;

    int64_t ol_start = static_cast<int64_t>(blockIdx.x) * gridDim.y + blockIdx.y;
    int64_t ol_stride = static_cast<int64_t>(gridDim.x) * gridDim.y;
 
    int lane_id = threadIdx.x & 31;
    using acc_t = at::acc_type<scalar_t, true>;
    __shared__ acc_t s_w[LUT_SIZE];
    __shared__ float s_grad_w[LUT_SIZE]; 

    for (int64_t ol_idx = ol_start; ol_idx < total_ol; ol_idx += ol_stride) {
        for (int i = threadIdx.x; i < LUT_SIZE; i += blockDim.x) {
            s_w[i] = static_cast<acc_t>(w[ol_idx * LUT_SIZE + i]);
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
            int64_t x_base = valid ? (g_idx * L + l_idx) * K * BR + br : 0;
         
            acc_t xp[K], xn[K];
            if (valid) {
                #pragma unroll
                for (int j = 0; j < K; ++j) {
                    xp[j] = static_cast<acc_t>(x[x_base + j * BR]);
                    xn[j] = 1.0 - xp[j];
                }
            }

            acc_t prob[LUT_SIZE];
            #pragma unroll
            for (int i = 0; i < LUT_SIZE; ++i) {
                acc_t p = 1.0;
                #pragma unroll
                for (int j = 0; j < K; ++j) p *= ((i >> ((K - 1) - j)) & 1) ? xp[j] : xn[j];
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
                for (int k = 0; k < K; ++k) {
                    acc_t dx_val = 0;
                    int pos = (K - 1) - k; 
                    int mask = 1 << pos;
                    
                    #pragma unroll
                    for (int c = 0; c < (1 << (K - 1)); ++c) {
                        int i0 = ((c >> pos) << (pos + 1)) | (c & (mask - 1));
                        int i1 = i0 | mask;
                        dx_val += (prob[i0] + prob[i1]) * (s_w[i1] - s_w[i0]);
                    }
                    gpuAtomicAdd(&grad_x_fp32[x_base + k * BR], static_cast<float>(dy * dx_val));
                }
            }
        }
        __syncthreads(); 
        for (int i = threadIdx.x; i < LUT_SIZE; i += blockDim.x) {
            if (s_grad_w[i] != 0.0f) {
                gpuAtomicAdd(&grad_w_fp32[ol_idx * LUT_SIZE + i], s_grad_w[i]);
            }
        }
        __syncthreads(); 
    }
}

// ------------------------------------------------------------------------
// Host 端封装 & 宏分发
// ------------------------------------------------------------------------
#define DISPATCH_FLOATING_FORWARD(K_VAL) \
    case K_VAL: \
        lut_interp_K_floating_forward_kernel<scalar_t, K_VAL><<<blocks, 64>>>( \
            x.data_ptr<scalar_t>(), w.data_ptr<scalar_t>(), y.data_ptr<scalar_t>(), \
            out_num * L, BR, L, static_cast<int>(out_num / groups)); \
        break;

torch::Tensor forward_cuda_floating(torch::Tensor x, torch::Tensor w, int groups, int K) {
    const int64_t out_num = w.size(0);
    const int64_t L = w.size(1);
    const int64_t BR = x.size(3);

    auto y = torch::empty({out_num, L, BR}, x.options());
    dim3 blocks(std::min((int)out_num, 65535), std::min((int)L, 65535), std::min((int)((BR + 63) / 64), 1024));

    AT_DISPATCH_FLOATING_TYPES_AND2(at::ScalarType::Half, at::ScalarType::BFloat16, x.scalar_type(), "lut_float_fw", [&] {
        switch (K) {
            DISPATCH_FLOATING_FORWARD(4)
            DISPATCH_FLOATING_FORWARD(5)
            DISPATCH_FLOATING_FORWARD(6)
            default: TORCH_CHECK(false, "Unsupported LUT K: ", K);
        }
    });
    return y;
}

#define DISPATCH_FLOATING_BACKWARD(K_VAL) \
    case K_VAL: \
        lut_interp_K_floating_backward_kernel<scalar_t, K_VAL><<<blocks, 64>>>( \
            grad_y.data_ptr<scalar_t>(), x.data_ptr<scalar_t>(), w.data_ptr<scalar_t>(), \
            grad_x_fp32.data_ptr<float>(), grad_w_fp32.data_ptr<float>(), \
            out_num * L, BR, L, static_cast<int>(out_num / groups)); \
        break;

std::vector<torch::Tensor> backward_cuda_floating(torch::Tensor grad_y, torch::Tensor x, torch::Tensor w, int groups, int K) {
    const int64_t out_num = w.size(0);
    const int64_t L = w.size(1);
    const int64_t BR = x.size(3);

    auto grad_x_fp32 = torch::zeros_like(x, x.options().dtype(torch::kFloat32));
    auto grad_w_fp32 = torch::zeros_like(w, w.options().dtype(torch::kFloat32));

    dim3 blocks(std::min((int)out_num, 65535), std::min((int)L, 65535), std::min((int)((BR + 63) / 64), 1024));

    AT_DISPATCH_FLOATING_TYPES_AND2(at::ScalarType::Half, at::ScalarType::BFloat16, x.scalar_type(), "lut_float_bw", [&] {
        switch (K) {
            DISPATCH_FLOATING_BACKWARD(4)
            DISPATCH_FLOATING_BACKWARD(5)
            DISPATCH_FLOATING_BACKWARD(6)
            default: TORCH_CHECK(false, "Unsupported LUT K: ", K);
        }
    });

    return {grad_x_fp32.to(x.scalar_type()), grad_w_fp32.to(w.scalar_type())};
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward_cuda_floating, "Floating LUT K Forward");
    m.def("backward", &backward_cuda_floating, "Floating LUT K Backward");
}
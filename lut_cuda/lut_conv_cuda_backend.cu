#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <ATen/cuda/Atomic.cuh>
#include <cstdint>

#define TILE_OC 32              
#define WARPS_PER_BLOCK 4       
#define FULL_MASK 0xffffffff
#define CEIL_DIV(M, N) (((M) + (N) - 1) / (N))

// ------------------------------------------------------------------------
// 极其安全的 VectorLoader 特化，替代 if constexpr
// ------------------------------------------------------------------------
template <typename scalar_t>
struct VectorLoader {
    static __device__ __forceinline__ void load(const scalar_t* src, float* dest) {
        #pragma unroll
        for(int v=0; v<(16/sizeof(scalar_t)); ++v) dest[v] = static_cast<float>(src[v]);
    }
};

template <>
struct VectorLoader<float> {
    static __device__ __forceinline__ void load(const float* src, float* dest) {
        if (reinterpret_cast<std::uintptr_t>(src) % 16 == 0) {
            reinterpret_cast<float4*>(dest)[0] = reinterpret_cast<const float4*>(src)[0];
        } else {
            #pragma unroll
            for(int v=0; v<4; ++v) dest[v] = src[v];
        }
    }
};

template <>
struct VectorLoader<at::Half> {
    static __device__ __forceinline__ void load(const at::Half* src, float* dest) {
        if (reinterpret_cast<std::uintptr_t>(src) % 16 == 0) {
            union { uint4 vec; half arr[8]; } packed;
            packed.vec = reinterpret_cast<const uint4*>(src)[0];
            #pragma unroll
            for(int v = 0; v < 8; ++v) dest[v] = __half2float(packed.arr[v]);
        } else {
            #pragma unroll
            for(int v=0; v<8; ++v) dest[v] = __half2float(src[v]);
        }
    }
};

template <>
struct VectorLoader<at::BFloat16> {
    static __device__ __forceinline__ void load(const at::BFloat16* src, float* dest) {
        if (reinterpret_cast<std::uintptr_t>(src) % 16 == 0) {
            union { uint4 vec; at::BFloat16 arr[8]; } packed;
            packed.vec = reinterpret_cast<const uint4*>(src)[0];
            #pragma unroll
            for(int v = 0; v < 8; ++v) dest[v] = static_cast<float>(packed.arr[v]);
        } else {
            #pragma unroll
            for(int v=0; v<8; ++v) dest[v] = static_cast<float>(src[v]);
        }
    }
};

// ------------------------------------------------------------------------
// 1. 模板化 Forward Kernel
// ------------------------------------------------------------------------
template <typename scalar_t, int LUT_K>
__launch_bounds__(128, 4)
__global__ void lut_conv_fp32_forward_kernel(
    const scalar_t* __restrict__ x, const scalar_t* __restrict__ w, const int32_t* __restrict__ offsets,
    scalar_t* __restrict__ y,
    int B, int in_C, int H, int W, int out_C, int OH, int OW,
    int stride, int oc_per_g, int lut_num) 
{
    constexpr int LUT_SIZE = 1 << LUT_K;
    const int lane_id = threadIdx.x; const int warp_id = threadIdx.y;
    const int global_warp_idx = blockIdx.x * blockDim.y + warp_id;
    const int oc_base = blockIdx.y * TILE_OC;

    bool is_valid_spatial = (global_warp_idx < B * OH * OW);
    const int ow = is_valid_spatial ? (global_warp_idx % OW) : 0; 
    const int oh = is_valid_spatial ? ((global_warp_idx / OW) % OH) : 0;
    const int b  = is_valid_spatial ? (global_warp_idx / (OH * OW)) : 0;
    
    __shared__ float s_w[64][TILE_OC]; 
    
    float y_accum = 0.0f;
    const int tid = warp_id * 32 + lane_id;

    for (int l = 0; l < lut_num; ++l) {
        for (int i = tid; i < LUT_SIZE * TILE_OC; i += 128) {
            int r = i / TILE_OC; int c = i % TILE_OC;
            if (oc_base + c < out_C) {
                s_w[r][c] = static_cast<float>(w[(l * LUT_SIZE + r) * out_C + oc_base + c]);
            } else {
                s_w[r][c] = 0.0f;
            }
        }
        __syncthreads();

        float loaded_x = 0.0f;
        if (lane_id < LUT_K && is_valid_spatial) {
            int rel_offset = offsets[l * LUT_K + lane_id];
            int base = ((b * H + oh * stride) * W + ow * stride) * in_C;
            loaded_x = static_cast<float>(x[base + rel_offset]);
        }
        
        float xp[6]; 
        #pragma unroll
        for (int j = 0; j < LUT_K; ++j) xp[j] = __shfl_sync(FULL_MASK, loaded_x, j);

        float v[64];
        #pragma unroll
        for (int i = 0; i < LUT_SIZE; ++i) v[i] = s_w[i][lane_id];

        #pragma unroll
        for (int d = 0; d < LUT_K; ++d) {
            #pragma unroll
            for (int i = 0; i < (1 << (LUT_K - 1 - d)); ++i) 
                v[i] = __fmaf_rn(xp[LUT_K - 1 - d], v[2*i+1] - v[2*i], v[2*i]);
        }
        y_accum += v[0];

        __syncthreads(); 
    }
    
    if (is_valid_spatial && oc_base + lane_id < out_C) {
        y[global_warp_idx * out_C + oc_base + lane_id] = static_cast<scalar_t>(y_accum);
    }
}

// ------------------------------------------------------------------------
// 2. 模板化 Backward Kernel
// ------------------------------------------------------------------------
template <typename scalar_t, int LUT_K>
__launch_bounds__(128, 4)
__global__ void lut_conv_fp32_backward_kernel(
    const scalar_t* __restrict__ grad_y, const scalar_t* __restrict__ x, const scalar_t* __restrict__ w,
    const int32_t* __restrict__ offsets,
    float* __restrict__ grad_x_fp32, float* __restrict__ grad_w_fp32,
    int B, int in_C, int H, int W, int out_C, int OH, int OW,
    int stride, int oc_per_g, int lut_num)
{
    constexpr int LUT_SIZE = 1 << LUT_K;
    const int lane_id = threadIdx.x; const int warp_id = threadIdx.y;
    const int global_warp_idx = blockIdx.x * blockDim.y + warp_id;
    const int oc_base = blockIdx.y * TILE_OC;
    
    bool is_valid_spatial = (global_warp_idx < B * OH * OW);
    bool valid_oc = (oc_base + lane_id < out_C);

    float dy = (is_valid_spatial && valid_oc) ? static_cast<float>(grad_y[global_warp_idx * out_C + oc_base + lane_id]) : 0.0f;
    __shared__ float s_w[64][TILE_OC];
    __shared__ float s_grad_w[WARPS_PER_BLOCK][64][TILE_OC]; 

    const int tid = warp_id * 32 + lane_id;

    for (int l = 0; l < lut_num; ++l) {
        for (int i = tid; i < WARPS_PER_BLOCK * 64 * TILE_OC; i += 128) {
            int w_idx = i / (64 * TILE_OC);
            int r_idx = (i / TILE_OC) % 64;
            int c_idx = i % TILE_OC;
            s_grad_w[w_idx][r_idx][c_idx] = 0.0f;
        }

        for (int i = tid; i < LUT_SIZE * TILE_OC; i += 128) {
            int r = i / TILE_OC; int c = i % TILE_OC;
            if (oc_base + c < out_C) {
                s_w[r][c] = static_cast<float>(w[(l * LUT_SIZE + r) * out_C + oc_base + c]);
            } else {
                s_w[r][c] = 0.0f;
            }
        }
        __syncthreads();
        
        float loaded_x = 0.0f; int loaded_idx = -1;
        if (lane_id < LUT_K && is_valid_spatial) {
            int rel_offset = offsets[l * LUT_K + lane_id];
            int base = ((global_warp_idx / (OH * OW) * H + (global_warp_idx % (OH * OW) / OW) * stride) * W + (global_warp_idx % OW) * stride) * in_C;
            loaded_idx = base + rel_offset;
            loaded_x = static_cast<float>(x[loaded_idx]);
        }
        
        float xp[6];
        #pragma unroll
        for (int j = 0; j < LUT_K; ++j) xp[j] = __shfl_sync(FULL_MASK, loaded_x, j);

        float prob[64];
        #pragma unroll
        for (int i = 0; i < LUT_SIZE; ++i) {
            float p = 1.0f;
            #pragma unroll
            for (int j = 0; j < LUT_K; ++j) {
                p *= ((i >> (LUT_K - 1 - j)) & 1) ? xp[j] : (1.0f - xp[j]);
            }
            prob[i] = p;
            
            if (dy != 0.0f) {
                atomicAdd(&s_grad_w[warp_id][i][lane_id], p * dy);
            }
        }

        #pragma unroll
        for (int k = 0; k < LUT_K; ++k) {
            float dx_val = 0.0f;
            int pos = LUT_K - 1 - k;
            int mask = 1 << pos;
            
            #pragma unroll
            for (int c = 0; c < (1 << (LUT_K - 1)); ++c) {
                int i0 = ((c >> pos) << (pos + 1)) | (c & (mask - 1));
                int i1 = i0 | mask;
                dx_val += (prob[i0] + prob[i1]) * (s_w[i1][lane_id] - s_w[i0][lane_id]);
            }
            dx_val *= dy;

            #pragma unroll
            for (int offset = 16; offset > 0; offset /= 2) dx_val += __shfl_down_sync(FULL_MASK, dx_val, offset);
            
            int target = __shfl_sync(FULL_MASK, loaded_idx, k);
            if (lane_id == 0 && target != -1 && dx_val != 0.0f) {
                atomicAdd(&grad_x_fp32[target], dx_val);
            }
        }

        __syncthreads(); 

        for (int i = tid; i < LUT_SIZE * TILE_OC; i += 128) {
            int r = i / TILE_OC; int c = i % TILE_OC;
            if (oc_base + c < out_C) {
                float total_grad = 0.0f;
                #pragma unroll
                for (int w = 0; w < WARPS_PER_BLOCK; ++w) total_grad += s_grad_w[w][r][c];
                if (total_grad != 0.0f) atomicAdd(&grad_w_fp32[(l * LUT_SIZE + r) * out_C + oc_base + c], total_grad);
            }
        }
        __syncthreads(); 
    }
}

// ------------------------------------------------------------------------
// 3. API 层与分发
// ------------------------------------------------------------------------
#define DISPATCH_FP32_FW(LUT_K)     case LUT_K:         lut_conv_fp32_forward_kernel<scalar_t, LUT_K><<<blocks, threads>>>(             x_cl.data_ptr<scalar_t>(), w_cont.data_ptr<scalar_t>(), offsets_cont.data_ptr<int32_t>(), y.data_ptr<scalar_t>(),             x_cl.size(0), x_cl.size(1), x_cl.size(2), x_cl.size(3), out_C, OH, OW, stride,             oc_per_g, w_cont.size(0)         ); break;

torch::Tensor forward_implicit(torch::Tensor x, torch::Tensor w, torch::Tensor offsets, int groups, int K, int kernel_size, int stride, int OH, int OW) {
    auto x_cl = x.contiguous(at::MemoryFormat::ChannelsLast);
    auto w_cont = w.contiguous();
    auto offsets_cont = offsets.contiguous();
    int out_C = w_cont.size(2);
    int oc_per_g = out_C / groups;

    auto y = torch::empty({x_cl.size(0), out_C, OH, OW}, x_cl.options().memory_format(at::MemoryFormat::ChannelsLast));
    dim3 threads(32, WARPS_PER_BLOCK), blocks(CEIL_DIV(x_cl.size(0) * OH * OW, WARPS_PER_BLOCK), CEIL_DIV(out_C, TILE_OC));

    AT_DISPATCH_FLOATING_TYPES_AND2(at::ScalarType::Half, at::ScalarType::BFloat16, x_cl.scalar_type(), "lut_fw", [&] {
        switch (K) {
            DISPATCH_FP32_FW(2)
            DISPATCH_FP32_FW(4)
            DISPATCH_FP32_FW(5)
            DISPATCH_FP32_FW(6)
            default: TORCH_CHECK(false, "Unsupported K: ", K);
        }
    });
    return y;
}

#define DISPATCH_FP32_BW(LUT_K)     case LUT_K:         lut_conv_fp32_backward_kernel<scalar_t, LUT_K><<<blocks, threads>>>(             grad_y_cl.data_ptr<scalar_t>(), x_cl.data_ptr<scalar_t>(), w_cont.data_ptr<scalar_t>(),             offsets_cont.data_ptr<int32_t>(), gx.data_ptr<float>(), gw.data_ptr<float>(),             x_cl.size(0), x_cl.size(1), x_cl.size(2), x_cl.size(3), out_C, OH, OW, stride,             oc_per_g, w_cont.size(0)         ); break;

std::vector<torch::Tensor> backward_implicit(torch::Tensor grad_y, torch::Tensor x, torch::Tensor w, torch::Tensor offsets, int groups, int K, int kernel_size, int stride, int OH, int OW) {
    auto x_cl = x.contiguous(at::MemoryFormat::ChannelsLast);
    auto grad_y_cl = grad_y.contiguous(at::MemoryFormat::ChannelsLast);
    auto w_cont = w.contiguous();
    auto offsets_cont = offsets.contiguous();
    int out_C = w_cont.size(2);
    int oc_per_g = out_C / groups;

    auto gx = torch::zeros_like(x_cl, x_cl.options().dtype(torch::kFloat32));
    auto gw = torch::zeros_like(w_cont, w_cont.options().dtype(torch::kFloat32));
    dim3 threads(32, WARPS_PER_BLOCK), blocks(CEIL_DIV(x_cl.size(0) * OH * OW, WARPS_PER_BLOCK), CEIL_DIV(out_C, TILE_OC));

    AT_DISPATCH_FLOATING_TYPES_AND2(at::ScalarType::Half, at::ScalarType::BFloat16, grad_y_cl.scalar_type(), "lut_bw", [&] {
        switch (K) {
            DISPATCH_FP32_BW(2)
            DISPATCH_FP32_BW(4)
            DISPATCH_FP32_BW(5)
            DISPATCH_FP32_BW(6)
            default: TORCH_CHECK(false, "Unsupported K: ", K);
        }
    });
    return {gx.to(x_cl.scalar_type()), gw.to(w_cont.scalar_type())};
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward_implicit); 
    m.def("backward", &backward_implicit);
}
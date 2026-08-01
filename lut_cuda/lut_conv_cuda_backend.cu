#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <ATen/cuda/Atomic.cuh>
#include <cstdint>

// ------------------------------------------------------------------------
// 1. 架构常量：对齐物理 Bank 与 SM 配额
// ------------------------------------------------------------------------
#define TILE_OC 32              
#define WARPS_PER_BLOCK 4       
#define FULL_MASK 0xffffffff
#define CEIL_DIV(M, N) (((M) + (N) - 1) / (N))

#define AT_DISPATCH_CUSTOM_TYPES(TYPE, NAME, ...) \
    AT_DISPATCH_SWITCH(TYPE, NAME, \
        AT_DISPATCH_CASE(at::ScalarType::Float, __VA_ARGS__) \
        AT_DISPATCH_CASE(at::ScalarType::Half, __VA_ARGS__) \
        AT_DISPATCH_CASE(at::ScalarType::BFloat16, __VA_ARGS__) \
    )

// ------------------------------------------------------------------------
// 2. 增强型对齐加载器 (Union 内存别名)
// ------------------------------------------------------------------------
template <typename scalar_t>
struct VectorLoader {
    static __device__ __forceinline__ void load(const scalar_t* src, float* dest) {
        if (reinterpret_cast<std::uintptr_t>(src) % 16 == 0) {
            if constexpr (sizeof(scalar_t) == 4) {
                reinterpret_cast<float4*>(dest)[0] = reinterpret_cast<const float4*>(src)[0];
            } else {
                uint4 raw = reinterpret_cast<const uint4*>(src)[0];
                const half* h = reinterpret_cast<const half*>(&raw);
                #pragma unroll
                for(int v=0; v<8; ++v) dest[v] = __half2float(h[v]);
            }
        } else {
            #pragma unroll
            for(int v=0; v<(16/sizeof(scalar_t)); ++v) dest[v] = static_cast<float>(src[v]);
        }
    }
};

template <>
struct VectorLoader<float> {
    static __device__ __forceinline__ void load(const float* src, float* dest) {
        reinterpret_cast<float4*>(dest)[0] = reinterpret_cast<const float4*>(src)[0];
    }
};

template <>
struct VectorLoader<at::Half> {
    static __device__ __forceinline__ void load(const at::Half* src, float* dest) {
        union { uint4 vec; half arr[8]; } packed;
        packed.vec = reinterpret_cast<const uint4*>(src)[0];
        #pragma unroll
        for(int v = 0; v < 8; ++v) dest[v] = __half2float(packed.arr[v]);
    }
};

template <>
struct VectorLoader<at::BFloat16> {
    static __device__ __forceinline__ void load(const at::BFloat16* src, float* dest) {
        union { uint4 vec; at::BFloat16 arr[8]; } packed;
        packed.vec = reinterpret_cast<const uint4*>(src)[0];
        #pragma unroll
        for(int v = 0; v < 8; ++v) dest[v] = static_cast<float>(packed.arr[v]);
    }
};

// ------------------------------------------------------------------------
// 3. Forward Kernel (完美融合 FMA 级联折减 + 寄存器保护)
// ------------------------------------------------------------------------
template <typename scalar_t>
__launch_bounds__(128, 4)
__global__ void lut_conv_fp32_forward_kernel(
    const scalar_t* __restrict__ x, const scalar_t* __restrict__ w, scalar_t* __restrict__ y,
    int B, int in_C, int H, int W, int out_C, int OH, int OW,
    int K, int stride, int pad_H, int pad_W, int ic_per_g, int oc_per_g, int lut_num) 
{
    const int lane_id = threadIdx.x; const int warp_id = threadIdx.y;
    const int global_warp_idx = blockIdx.x * blockDim.y + warp_id;
    const int oc_base = blockIdx.y * TILE_OC;

    bool is_valid_spatial = (global_warp_idx < B * OH * OW);
    const int ow = is_valid_spatial ? (global_warp_idx % OW) : 0; 
    const int oh = is_valid_spatial ? ((global_warp_idx / OW) % OH) : 0;
    const int b  = is_valid_spatial ? (global_warp_idx / (OH * OW)) : 0;
    
    const int ic_start = (oc_base / oc_per_g) * ic_per_g;
    const int max_sp = ic_per_g * K * K;

    __shared__ float s_w[64][TILE_OC];
    
    float y_accum = 0.0f;
    const int tid = warp_id * 32 + lane_id;
    int sp_idx_base = (lane_id < 6) ? lane_id : 0;

    for (int l = 0; l < lut_num; ++l) {
        int v_step = (sizeof(scalar_t) == 2) ? 8 : 4;
        for (int i = tid; i < 64 * (TILE_OC / v_step); i += 128) {
            int r = i / (TILE_OC / v_step); int vc = i % (TILE_OC / v_step);
            if (oc_base + vc * v_step < out_C)
                VectorLoader<scalar_t>::load(&w[(l * 64 + r) * out_C + oc_base + vc * v_step], &s_w[r][vc * v_step]);
        }
        __syncthreads();

        // 🎯 防御核心：隔离的 loaded_x 寄存器
        float loaded_x = 0.0f;
        if (lane_id < 6 && is_valid_spatial) {
            int ic = ic_start + (sp_idx_base / (K * K));
            int sp = sp_idx_base % (K * K);
            int cur_h = oh * stride - pad_H + sp / K;
            int cur_w = ow * stride - pad_W + sp % K;
            if (cur_h >= 0 && cur_h < H && cur_w >= 0 && cur_w < W)
                loaded_x = static_cast<float>(x[((b * H + cur_h) * W + cur_w) * in_C + ic]);
        }
        
        float xp[6];
        #pragma unroll
        for (int j = 0; j < 6; ++j) xp[j] = __shfl_sync(FULL_MASK, loaded_x, j);

        // 🎯 优化核心：恢复 FMA 级联折减 (寄存器利用率拉满)
        float v[16];
        #pragma unroll
        for (int i = 0; i < 16; ++i) {
            float v0 = __fmaf_rn(xp[5], s_w[i*4+1][lane_id] - s_w[i*4][lane_id], s_w[i*4][lane_id]);
            float v1 = __fmaf_rn(xp[5], s_w[i*4+3][lane_id] - s_w[i*4+2][lane_id], s_w[i*4+2][lane_id]);
            v[i] = __fmaf_rn(xp[4], v1 - v0, v0);
        }
        #pragma unroll
        for (int d = 0; d < 4; ++d) {
            #pragma unroll
            for (int i = 0; i < (1 << (3 - d)); ++i) 
                v[i] = __fmaf_rn(xp[3-d], v[2*i+1] - v[2*i], v[2*i]);
        }
        y_accum += v[0];

        sp_idx_base = (sp_idx_base + 6) % max_sp;
        __syncthreads(); 
    }
    
    if (is_valid_spatial && oc_base + lane_id < out_C) {
        y[global_warp_idx * out_C + oc_base + lane_id] = static_cast<scalar_t>(y_accum);
    }
}

// ------------------------------------------------------------------------
// 4. Backward Kernel (分段级联概率折减 + 寄存器保护)
// ------------------------------------------------------------------------
template <typename scalar_t>
__launch_bounds__(128, 4)
__global__ void lut_conv_fp32_backward_kernel(
    const scalar_t* __restrict__ grad_y, const scalar_t* __restrict__ x, const scalar_t* __restrict__ w,
    float* __restrict__ grad_x_fp32, float* __restrict__ grad_w_fp32,
    int B, int in_C, int H, int W, int out_C, int OH, int OW,
    int K, int stride, int pad_H, int pad_W, int ic_per_g, int oc_per_g, int lut_num)
{
    const int lane_id = threadIdx.x; const int warp_id = threadIdx.y;
    const int global_warp_idx = blockIdx.x * blockDim.y + warp_id;
    const int oc_base = blockIdx.y * TILE_OC;
    
    bool is_valid_spatial = (global_warp_idx < B * OH * OW);
    bool valid_oc = (oc_base + lane_id < out_C);

    float dy = (is_valid_spatial && valid_oc) ? static_cast<float>(grad_y[global_warp_idx * out_C + oc_base + lane_id]) : 0.0f;
    const int ic_start = (oc_base / oc_per_g) * ic_per_g;
    const int max_sp = ic_per_g * K * K;

    __shared__ float s_w[64][TILE_OC];
    __shared__ float s_grad_w[WARPS_PER_BLOCK][64][TILE_OC]; 

    const int tid = warp_id * 32 + lane_id;
    int sp_idx_base = (lane_id < 6) ? lane_id : 0;

    for (int l = 0; l < lut_num; ++l) {
        for (int i = tid; i < WARPS_PER_BLOCK * 64 * TILE_OC; i += 128) {
            int w_idx = i / (64 * TILE_OC);
            int r_idx = (i / TILE_OC) % 64;
            int c_idx = i % TILE_OC;
            s_grad_w[w_idx][r_idx][c_idx] = 0.0f;
        }

        int v_step = (sizeof(scalar_t) == 2) ? 8 : 4;
        for (int i = tid; i < 64 * (TILE_OC / v_step); i += 128) {
            int r = i / (TILE_OC / v_step); int vc = i % (TILE_OC / v_step);
            if (oc_base + vc * v_step < out_C)
                VectorLoader<scalar_t>::load(&w[(l * 64 + r) * out_C + oc_base + vc * v_step], &s_w[r][vc * v_step]);
        }
        __syncthreads();
        
        // 🎯 防御核心：隔离的 loaded_x 与 loaded_idx
        float loaded_x = 0.0f; int loaded_idx = -1;
        if (lane_id < 6 && is_valid_spatial) {
            int ic = ic_start + (sp_idx_base / (K * K));
            int sp = sp_idx_base % (K * K);
            int ch = (global_warp_idx % (OH * OW) / OW) * stride - pad_H + sp / K;
            int cw = (global_warp_idx % OW) * stride - pad_W + sp % K;
            if (ch >= 0 && ch < H && cw >= 0 && cw < W) {
                loaded_idx = ((global_warp_idx / (OH * OW) * H + ch) * W + cw) * in_C + ic;
                loaded_x = static_cast<float>(x[loaded_idx]);
            }
        }
        
        float xp[6];
        #pragma unroll
        for (int j = 0; j < 6; ++j) xp[j] = __shfl_sync(FULL_MASK, loaded_x, j);

        // 🎯 优化核心：恢复分段概率折减 (避免 64次全计算)
        float dx_accum[6] = {0.0f}; float v_seg[4] = {0.0f};

        #pragma unroll
        for (int g = 0; g < 4; ++g) {
            float p_g_base = (((g >> 1) & 1) ? xp[0] : (1.0f - xp[0])) * (((g & 1) ? xp[1] : (1.0f - xp[1])));
            float p_sub[16]; p_sub[0] = p_g_base; 
            
            #pragma unroll
            for (int d = 0; d < 4; ++d) {
                float xv = xp[d + 2];
                #pragma unroll
                for (int i = (1 << d) - 1; i >= 0; --i) {
                    p_sub[2 * i + 1] = p_sub[i] * xv; p_sub[2 * i] = p_sub[i] * (1.0f - xv);
                }
            }

            #pragma unroll
            for(int i = 0; i < 16; ++i) {
                if (dy != 0.0f) atomicAdd(&s_grad_w[warp_id][g * 16 + i][lane_id], p_sub[i] * dy);
                v_seg[g] += p_sub[i] * s_w[g * 16 + i][lane_id]; 
            }

            #pragma unroll
            for (int k = 0; k < 4; ++k) {
                int mask = 1 << (3 - k);
                #pragma unroll
                for (int i = 0; i < 8; ++i) {
                    int i0 = ((i >> (3-k)) << (4-k)) | (i & (mask - 1));
                    int i1 = i0 | mask;
                    dx_accum[k+2] += dy * (p_sub[i0] + p_sub[i1]) * (s_w[g*16 + i1][lane_id] - s_w[g*16 + i0][lane_id]);
                }
            }
        }
        
        dx_accum[0] = dy * ((1.0f - xp[1]) * (v_seg[2] - v_seg[0]) + xp[1] * (v_seg[3] - v_seg[1]));
        dx_accum[1] = dy * ((1.0f - xp[0]) * (v_seg[1] - v_seg[0]) + xp[0] * (v_seg[3] - v_seg[2]));

        #pragma unroll
        for (int k = 0; k < 6; ++k) {
            float val = dx_accum[k];
            #pragma unroll
            for (int offset = 16; offset > 0; offset /= 2) val += __shfl_down_sync(FULL_MASK, val, offset);
            int target = __shfl_sync(FULL_MASK, loaded_idx, k);
            if (lane_id == 0 && target != -1 && val != 0.0f) atomicAdd(&grad_x_fp32[target], val);
        }

        sp_idx_base = (sp_idx_base + 6) % max_sp;
        __syncthreads(); 

        for (int i = tid; i < 64 * TILE_OC; i += 128) {
            int r = i / TILE_OC; int c = i % TILE_OC;
            if (oc_base + c < out_C) {
                float total_grad = 0.0f;
                #pragma unroll
                for (int w = 0; w < WARPS_PER_BLOCK; ++w) {
                    total_grad += s_grad_w[w][r][c];
                }
                if (total_grad != 0.0f) {
                    atomicAdd(&grad_w_fp32[(l * 64 + r) * out_C + oc_base + c], total_grad);
                }
            }
        }
        __syncthreads(); 
    }
}

// ------------------------------------------------------------------------
// 5. API 层：铁血防御与组约束保护
// ------------------------------------------------------------------------
torch::Tensor forward_implicit(torch::Tensor x, torch::Tensor w, int groups, int K, int stride, int pad_H, int pad_W, int OH, int OW) {
    auto x_cl = x.contiguous(at::MemoryFormat::ChannelsLast);
    auto w_cont = w.contiguous(); 
    int out_C = w_cont.size(2);
    int oc_per_g = out_C / groups;

    TORCH_CHECK(oc_per_g % TILE_OC == 0 || groups == 1, 
        "Fatal: Channels per group (oc_per_g) must be a multiple of 32 to prevent Warp cross-group contamination.");
    
    int align_req = (x_cl.scalar_type() == at::ScalarType::Float) ? 4 : 8;
    TORCH_CHECK(out_C % align_req == 0, "Alignment: out_C must be multiple of ", align_req);
    TORCH_CHECK(reinterpret_cast<std::uintptr_t>(x_cl.data_ptr()) % 16 == 0, "Alignment: Input x must be 16-byte aligned.");
    TORCH_CHECK(reinterpret_cast<std::uintptr_t>(w_cont.data_ptr()) % 16 == 0, "Alignment: Weight w must be 16-byte aligned.");

    auto y = torch::empty({x_cl.size(0), out_C, OH, OW}, x_cl.options().memory_format(at::MemoryFormat::ChannelsLast));
    dim3 threads(32, WARPS_PER_BLOCK), blocks(CEIL_DIV(x_cl.size(0) * OH * OW, WARPS_PER_BLOCK), CEIL_DIV(out_C, TILE_OC));
    
    AT_DISPATCH_CUSTOM_TYPES(x_cl.scalar_type(), "lut_fw", [&] {
        lut_conv_fp32_forward_kernel<scalar_t><<<blocks, threads>>>(
            x_cl.data_ptr<scalar_t>(), w_cont.data_ptr<scalar_t>(), y.data_ptr<scalar_t>(), 
            x_cl.size(0), x_cl.size(1), x_cl.size(2), x_cl.size(3), out_C, OH, OW, K, stride, pad_H, pad_W,
            x_cl.size(1)/groups, oc_per_g, w_cont.size(0)
        );
    });
    return y;
}

std::vector<torch::Tensor> backward_implicit(torch::Tensor grad_y, torch::Tensor x, torch::Tensor w, int groups, int K, int stride, int pad_H, int pad_W, int OH, int OW) {
    auto x_cl = x.contiguous(at::MemoryFormat::ChannelsLast); auto grad_y_cl = grad_y.contiguous(at::MemoryFormat::ChannelsLast);
    auto w_cont = w.contiguous();
    int out_C = w_cont.size(2);
    int oc_per_g = out_C / groups;

    TORCH_CHECK(oc_per_g % TILE_OC == 0 || groups == 1, 
        "Fatal: Channels per group (oc_per_g) must be a multiple of 32 to prevent Warp cross-group contamination.");
    
    int align_req = (x_cl.scalar_type() == at::ScalarType::Float) ? 4 : 8;
    TORCH_CHECK(out_C % align_req == 0, "Alignment: out_C must be multiple of ", align_req);
    TORCH_CHECK(reinterpret_cast<std::uintptr_t>(x_cl.data_ptr()) % 16 == 0, "Alignment: x must be 16-byte aligned.");
    TORCH_CHECK(reinterpret_cast<std::uintptr_t>(w_cont.data_ptr()) % 16 == 0, "Alignment: w must be 16-byte aligned.");
    
    auto gx = torch::zeros_like(x_cl, x_cl.options().dtype(torch::kFloat32));
    auto gw = torch::zeros_like(w_cont, w_cont.options().dtype(torch::kFloat32));
    dim3 threads(32, WARPS_PER_BLOCK), blocks(CEIL_DIV(x_cl.size(0) * OH * OW, WARPS_PER_BLOCK), CEIL_DIV(out_C, TILE_OC));

    AT_DISPATCH_CUSTOM_TYPES(grad_y_cl.scalar_type(), "lut_bw", [&] {
        lut_conv_fp32_backward_kernel<scalar_t><<<blocks, threads>>>(
            grad_y_cl.data_ptr<scalar_t>(), x_cl.data_ptr<scalar_t>(), w_cont.data_ptr<scalar_t>(),
            gx.data_ptr<float>(), gw.data_ptr<float>(),
            x_cl.size(0), x_cl.size(1), x_cl.size(2), x_cl.size(3), out_C, OH, OW, K, stride, pad_H, pad_W,
            x_cl.size(1)/groups, oc_per_g, w_cont.size(0)
        );
    });
    return {gx.to(x_cl.scalar_type()), gw.to(w_cont.scalar_type())};
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward_implicit); m.def("backward", &backward_implicit);
}

#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstdint>

#define CEIL_DIV(M, N) (((M) + (N) - 1) / (N))
#define TILE_OC 32
#define WARPS_PER_BLOCK 8  
#define FULL_MASK 0xffffffff

__global__ void fused_pad_permute_pack_kernel(
    const int8_t* __restrict__ in_nchw, 
    uint32_t* __restrict__ out_nhwc_packed, 
    int B, int in_C, int H, int W, 
    int pad_top, int pad_bottom, int pad_left, int pad_right,
    int padded_H, int padded_W, int packed_C) 
{
    // 每个 Warp (32 threads) 负责输出矩阵的一个 [spatial_idx, c_block]
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    const int lane_id = tid % 32;
    const int warp_id = tid / 32;
    
    const int total_warps = B * padded_H * padded_W * packed_C;

    if (warp_id < total_warps) {
        int spatial_idx = warp_id / packed_C;
        int c_block = warp_id % packed_C;

        // 解析输出张量 (NHWC) 的坐标
        int b  = spatial_idx / (padded_H * padded_W);
        int ph = (spatial_idx / padded_W) % padded_H;
        int pw = spatial_idx % padded_W;

        // 逆向映射回原生输入张量 (NCHW) 的坐标
        int h = ph - pad_top;
        int w = pw - pad_left;

        int c = c_block * 32 + lane_id;
        int8_t val = 0;
        
        // 只有当坐标落在原生输入范围内，且通道合法时，才去全局内存拉取数据
        if (h >= 0 && h < H && w >= 0 && w < W && c < in_C) {
            // NCHW 内存布局的寻址公式
            int nchw_idx = ((b * in_C + c) * H + h) * W + w;
            val = in_nchw[nchw_idx];
        }

        // Warp 级硬件归约：瞬间完成 32 bits 打包
        uint32_t word = __ballot_sync(FULL_MASK, val > 0);

        // 由 Lane 0 写入最终目标显存
        if (lane_id == 0) {
            out_nhwc_packed[spatial_idx * packed_C + c_block] = word;
        }
    }
}


// ============================================================================
// 2. Forward Kernel: O(1) 极速软权重查表 (Soft Weight Lookup)
// ============================================================================
template <typename scalar_t, int LUT_NUM>
__global__ void lut_conv_forward_ultimate_kernel(
    const uint32_t* __restrict__ packed_x,  
    const float* __restrict__ w_soft,       // 🚀 接收连续的浮点权重 [lut_num, 64, out_C]
    const int32_t* __restrict__ offsets,    
    const int32_t* __restrict__ shifts,     
    scalar_t* __restrict__ y,         
    int B, int packed_C, int padded_H, int padded_W, 
    int out_C, int OH, int OW,
    int stride)
{
    const int lane_id = threadIdx.x; 
    const int warp_id = threadIdx.y; 
    const int global_warp_idx = blockIdx.x * blockDim.y + warp_id; 
    const int oc_base = blockIdx.y * TILE_OC;

    bool is_valid_spatial = (global_warp_idx < B * OH * OW);
    const int ow = is_valid_spatial ? (global_warp_idx % OW) : 0;
    const int oh = is_valid_spatial ? ((global_warp_idx / OW) % OH) : 0;
    const int b  = is_valid_spatial ? (global_warp_idx / (OH * OW)) : 0;

    float y_val = 0.0f; 
    const int act_oc = oc_base + lane_id;
    const bool valid_oc = (act_oc < out_C);

    // 预计算当前滑窗的绝对物理内存基址 (零边界越界检查)
    int base_ptr = 0;
    if (is_valid_spatial) {
        base_ptr = ((b * padded_H + (oh * stride)) * padded_W + (ow * stride)) * packed_C;
    }

    // 完美循环展开：无跳转、无气泡，榨干 ILP 并行度
    #pragma unroll

    for (int l = 0; l < LUT_NUM; ++l) {
        int local_bit = 0;
        if (is_valid_spatial && lane_id < 6) {
            int flat_conn_idx = l * 6 + lane_id;
            
            // 纯只读缓存寻址 (L1.5/Texture Cache)
            int abs_offset = __ldg(&offsets[flat_conn_idx]); 
            int shift_val  = __ldg(&shifts[flat_conn_idx]);

            uint32_t word = __ldg(&packed_x[base_ptr + abs_offset]);
            local_bit = (word >> shift_val) & 1;
        }

        // 瞬间拼接出当前输入命中的真值表 Index (0~63)
        int current_idx = __ballot_sync(0x3F, local_bit == 1);

        if (is_valid_spatial && valid_oc) {
            // 🚀 O(1) 终极魔法：直接读取软权重，瞬间完成！
            y_val += __ldg(&w_soft[(l * 64 + current_idx) * out_C + act_oc]); 
        }
    }

    if (is_valid_spatial && valid_oc) {
        y[global_warp_idx * out_C + act_oc] = static_cast<scalar_t>(y_val);
    }
}

// ============================================================================
// 3. Backward Kernel: Top-7 Sparsemax & 完美有限差分
// ============================================================================
template <typename scalar_t>
__global__ void lut_conv_backward_ultimate_kernel(
    const scalar_t* __restrict__ grad_y, 
    const uint32_t* __restrict__ packed_x,      // 🚀 直接读 packed_x 节约带宽
    const float* __restrict__ x_float_nchw,  
    const float* __restrict__ w_soft,           // 🚀 读软权重
    const int32_t* __restrict__ offsets,
    const int32_t* __restrict__ shifts,
    float* __restrict__ grad_x_fp32,     
    float* __restrict__ grad_w_fp32,    
    int B, int in_C, int H, int W,           
    int packed_C, int padded_H, int padded_W, int out_C, int OH, int OW,
    int stride, int lut_num, int padding, float tau)    
{
    const int lane_id = threadIdx.x; 
    const int warp_id = threadIdx.y; 
    const int global_warp_idx = blockIdx.x * blockDim.y + warp_id; 
    const int oc_base = blockIdx.y * TILE_OC;
    const int tid = warp_id * 32 + lane_id;
    const int act_oc = oc_base + lane_id;
    const bool valid_oc = (act_oc < out_C);

    // 8KB 共享内存，完美装下当前 Warp Block 对应的 64 态梯度缓存
    __shared__ float s_grad_w[64][TILE_OC];

    bool is_valid_spatial = (global_warp_idx < B * OH * OW);
    const int ow = is_valid_spatial ? (global_warp_idx % OW) : 0;
    const int oh = is_valid_spatial ? ((global_warp_idx / OW) % OH) : 0;
    const int b = is_valid_spatial ? (global_warp_idx / (OH * OW)) : 0;

    float dy_val = (is_valid_spatial && valid_oc) ? static_cast<float>(grad_y[global_warp_idx * out_C + act_oc]) : 0.0f;

    // Backward 对应的原生 int8 空间指针基址
    int base_ptr = 0;
    if (is_valid_spatial) {
        base_ptr = ((b * padded_H + (oh * stride)) * padded_W + (ow * stride)) * packed_C;
    }

    for (int l = 0; l < lut_num; ++l) {
        // 无 Bank Conflict 的显存清理
        for (int i = tid; i < 64 * TILE_OC; i += WARPS_PER_BLOCK * 32) {
            (&s_grad_w[0][0])[i] = 0.0f;
        }
        __syncthreads(); 

        int local_nchw_idx = -1; 
        int local_bit = 0;
        float f_val = 0.0f; 

        if (is_valid_spatial && lane_id < 6) {
            int flat_conn_idx = l * 6 + lane_id;
            int abs_offset = __ldg(&offsets[flat_conn_idx]); 
            int shift_val  = __ldg(&shifts[flat_conn_idx]);

            uint32_t word = __ldg(&packed_x[base_ptr + abs_offset]);
            local_bit = (word >> shift_val) & 1;

            int spatial_off = abs_offset / packed_C;
            int c_word = abs_offset % packed_C;
            int c = c_word * 32 + shift_val;

            int ph = spatial_off / padded_W;
            int pw = spatial_off % padded_W;
            int h = ph - padding;
            int w = pw - padding;
            if (h >= 0 && h < H && w >= 0 && w < W && c < in_C) {
                local_nchw_idx = ((b * in_C + c) * H + h) * W + w;
                f_val = __ldg(&x_float_nchw[local_nchw_idx]);
            }
        }

        int current_idx = __ballot_sync(0x3F, local_bit == 1);
        
        int x_indices[6];
        float x_f[6];
        #pragma unroll
        for (int j = 0; j < 6; ++j) {
            // 瞬间将 6 个输入的地址和浮点值广播给整个 Warp 的 32 个线程
            x_indices[j] = __shfl_sync(0xFFFFFFFF, local_nchw_idx, j);
            x_f[j] = __shfl_sync(0xFFFFFFFF, f_val, j);
        }

        // ==============================================================
        // 🚀 核心优化 1：Top-7 Sparsemax 更新权重
        // ==============================================================
        if (is_valid_spatial && valid_oc && dy_val != 0.0f) {
            float v[7]; 
            // 1. 找出 6 个输入比特中的最大绝对值 (最大惩罚代价)
            float max_mag = 1e-6f; 
            #pragma unroll
            for (int k = 0; k < 6; ++k) {
                max_mag = fmaxf(max_mag, fabsf(x_f[k]));
            }

            // 2. 引入退火因子 (从 Python 传入一个 0.0 到 1.0 的 schedule_ratio 即可)
            // 训练初期 schedule_ratio = 1.0 (全盘探索)
            // 训练末期 schedule_ratio = 0.0 (强制坍缩回 Top-1)
            float S = 1.0f;
            v[6] = 1.0f; // 中心点永远是 1.0

            #pragma unroll
            for (int k = 0; k < 6; ++k) {
                // 自适应归一化：相对距离 (0.0 到 1.0 之间)
                float rel_dist = fabsf(x_f[k]) / max_mag; 
                
                // 距离越远(接近1)，权重越小；距离越近(接近0)，权重越大
                // 乘以 schedule_ratio 强制退火
                float vk = (1.0f - rel_dist) * tau; 
                
                v[k] = vk;
                S += vk;
            }

            float inv_S = 1.0f / S; // 安全归一化因子
            
            // 3. 完美无冲突写入：中心状态
            atomicAdd(&s_grad_w[current_idx][lane_id], dy_val * (v[6] * inv_S));
            
            // 完美无冲突写入：6 个邻居
            #pragma unroll
            for (int k = 0; k < 6; ++k) {
                if (v[k] > 0.0f) { // 末期 schedule_ratio为0时，邻居被完美剔除！
                    int neighbor_idx = current_idx ^ (1 << k);
                    atomicAdd(&s_grad_w[neighbor_idx][lane_id], dy_val * (v[k] * inv_S));
                }
            }
        }

        // ==============================================================
        // 🚀 核心优化 2：基于软权重的完美有限差分 (求解 X 的梯度)
        // ==============================================================
        if (is_valid_spatial && valid_oc) {
            #pragma unroll
            for (int k = 0; k < 6; ++k) {
                int idx1 = current_idx | (1 << k);
                int idx0 = current_idx & ~(1 << k);
                
                float w1 = __ldg(&w_soft[(l * 64 + idx1) * out_C + act_oc]);
                float w0 = __ldg(&w_soft[(l * 64 + idx0) * out_C + act_oc]);
                float x_float_val = x_f[k];
                float ste_mask = (fabsf(x_float_val) <= 1.0f) ? 1.0f : 0.0f; 
                
                float dx_val = dy_val * (w1 - w0) * ste_mask;

                #pragma unroll
                for (int offset = 16; offset > 0; offset /= 2) {
                    dx_val += __shfl_down_sync(0xFFFFFFFF, dx_val, offset);
                }
                
                if (lane_id == 0 && x_indices[k] != -1 && fabs(dx_val) > 1e-4) {
                    atomicAdd(&grad_x_fp32[x_indices[k]], dx_val);
                }
            }
        }
        __syncthreads();

        // 极速刷回 Global Memory
        for (int i = tid; i < 64 * TILE_OC; i += WARPS_PER_BLOCK * 32) {
            int bit = i / TILE_OC; int toc = i % TILE_OC; int c_act = oc_base + toc;
            if (c_act < out_C) {
                float val = (&s_grad_w[0][0])[i];
                if (val != 0.0f) atomicAdd(&grad_w_fp32[((l * 64 + bit) * out_C) + c_act], val);
            }
        }
        __syncthreads(); 
    }
}

// ============================================================================
// 4. API 导出与 PyBind11
// ============================================================================

// (1) 供宏分发使用的模版调用包裹器
#define DISPATCH_LUT_KERNEL(LUT_VAL) \
    case LUT_VAL: \
        lut_conv_forward_ultimate_kernel<scalar_t, LUT_VAL><<<blocks, threads>>>( \
            reinterpret_cast<uint32_t*>(packed_x.data_ptr<int32_t>()), \
            w_soft.data_ptr<float>(), \
            offsets.data_ptr<int32_t>(), \
            shifts.data_ptr<int32_t>(), \
            y.data_ptr<scalar_t>(), \
            B, packed_C, padded_H, padded_W, out_C, OH, OW, stride \
        ); \
        break;

// (2) Forward Conv API
torch::Tensor forward_implicit_ultimate(
    torch::Tensor packed_x, torch::Tensor w_soft, torch::Tensor offsets, torch::Tensor shifts, 
    int B, int padded_H, int padded_W, int OH, int OW, int stride) 
{
    // 增加数据类型检查，防止上游 Python 传参错误
    TORCH_CHECK(w_soft.scalar_type() == torch::kFloat32, "w_soft MUST be float32.");
    
    int packed_C = packed_x.size(3);
    int lut_num = w_soft.size(0); 
    int out_C = w_soft.size(2); 

    auto y = torch::empty({B, out_C, OH, OW}, packed_x.options().dtype(torch::kFloat32).memory_format(at::MemoryFormat::ChannelsLast));
    
    dim3 threads(32, WARPS_PER_BLOCK); 
    dim3 blocks(CEIL_DIV(B * OH * OW, WARPS_PER_BLOCK), CEIL_DIV(out_C, TILE_OC));
    
    // 修正: 将编译期的 LUT_NUM 分发，嵌套在运行期的 scalar_t 分发之内
    AT_DISPATCH_FLOATING_TYPES_AND2(at::ScalarType::Half, at::ScalarType::BFloat16, y.scalar_type(), "lut_fw", [&] {
        switch (lut_num) {
            DISPATCH_LUT_KERNEL(12)
            DISPATCH_LUT_KERNEL(24)
            DISPATCH_LUT_KERNEL(48)
            DISPATCH_LUT_KERNEL(96)
            DISPATCH_LUT_KERNEL(192)
            DISPATCH_LUT_KERNEL(384)
            DISPATCH_LUT_KERNEL(768)
            DISPATCH_LUT_KERNEL(1536)
            DISPATCH_LUT_KERNEL(3072)
            default:
                TORCH_CHECK(false, "Unsupported lut_num: ", lut_num, ". Please add it to the DISPATCH macro.");
        }
    });

    return y;
}

// (3) Backward API
std::vector<torch::Tensor> backward_ultimate(
    torch::Tensor grad_y_nhwc, 
    torch::Tensor packed_x, 
    torch::Tensor x_float_nchw, 
    torch::Tensor w_soft, 
    torch::Tensor offsets, 
    torch::Tensor shifts, 
    torch::Tensor tau_tensor, 
    int padding, 
    int B, int padded_H, int padded_W, int OH, int OW, int stride)
{
    auto grad_y_c = grad_y_nhwc.contiguous(); 
    auto x_float_c = x_float_nchw.contiguous();
    float tau = tau_tensor.item<float>();
    int in_C = x_float_c.size(1); 
    int packed_C = CEIL_DIV(in_C, 32);
    int lut_num = w_soft.size(0); 
    int out_C = w_soft.size(2);

    int H = x_float_c.size(2);
    int W = x_float_c.size(3);

    auto grad_x_fp32 = torch::zeros_like(x_float_c, x_float_c.options().dtype(torch::kFloat32).memory_format(at::MemoryFormat::Contiguous));
    auto grad_w_fp32 = torch::zeros_like(w_soft, w_soft.options().dtype(torch::kFloat32).memory_format(at::MemoryFormat::Contiguous));

    dim3 threads(32, WARPS_PER_BLOCK); 
    dim3 blocks(CEIL_DIV(B * OH * OW, WARPS_PER_BLOCK), CEIL_DIV(out_C, TILE_OC));

    AT_DISPATCH_FLOATING_TYPES_AND2(at::ScalarType::Half, at::ScalarType::BFloat16, grad_y_c.scalar_type(), "lut_bw", [&] {
        lut_conv_backward_ultimate_kernel<scalar_t><<<blocks, threads>>>(
            grad_y_c.data_ptr<scalar_t>(), 
            reinterpret_cast<uint32_t*>(packed_x.data_ptr<int32_t>()), 
            x_float_c.data_ptr<float>(), 
            w_soft.data_ptr<float>(), 
            offsets.data_ptr<int32_t>(), 
            shifts.data_ptr<int32_t>(), 
            grad_x_fp32.data_ptr<float>(), 
            grad_w_fp32.data_ptr<float>(), 
            B, in_C, H, W, packed_C, padded_H, padded_W, out_C, OH, OW, 
            stride, lut_num, padding, tau
        );
    });

    return {grad_x_fp32, grad_w_fp32};
}

torch::Tensor fused_pre_process(
    torch::Tensor x_nchw, 
    int pad_top, int pad_bottom, int pad_left, int pad_right) 
{
    TORCH_CHECK(x_nchw.scalar_type() == torch::kInt8, "x MUST be torch.int8!");
    auto x_contig = x_nchw.contiguous();
    
    int B = x_contig.size(0); 
    int in_C = x_contig.size(1); 
    int H = x_contig.size(2); 
    int W = x_contig.size(3);
    
    int padded_H = H + pad_top + pad_bottom;
    int padded_W = W + pad_left + pad_right;
    int packed_C = CEIL_DIV(in_C, 32);
    
    int total_spatial = B * padded_H * padded_W;
    int total_warps = total_spatial * packed_C;
    
    // 直接分配最终目标的连续内存 (逻辑上是 NHWC，但这里当作 1D 数据对待)
    auto packed_x = torch::zeros({B, padded_H, padded_W, packed_C}, x_contig.options().dtype(torch::kInt32));
    
    int threads_per_block = 128; // 4 Warps per block
    int blocks = CEIL_DIV(total_warps * 32, threads_per_block); 
    
    fused_pad_permute_pack_kernel<<<blocks, threads_per_block>>>(
        x_contig.data_ptr<int8_t>(), 
        reinterpret_cast<uint32_t*>(packed_x.data_ptr<int32_t>()), 
        B, in_C, H, W, 
        pad_top, pad_bottom, pad_left, pad_right,
        padded_H, padded_W, packed_C
    );
    
    return packed_x;
}

// ============================================================================
// 5. PyBind11 模块导出
// ============================================================================
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fused_pre_process", &fused_pre_process, "Pack Int8");
    m.def("conv_bafw_forward", &forward_implicit_ultimate, "Ultimate Zero-Branch Pipelined LUT Forward");
    m.def("conv_bafw_backward", &backward_ultimate, "Ultimate Pipelined LUT Backward");
}

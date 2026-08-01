#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstdint>

#define CEIL_DIV(M, N) (((M) + (N) - 1) / (N))
#define TILE_OC 32
#define WARPS_PER_BLOCK 8  
#define FULL_MASK 0xffffffff

// ============================================================================
// 1. Warp-Level Bit-Packing Kernel: 完美 32-byte 合并访存
// 将 [B, padded_H, padded_W, in_C] 的 int8 压缩为 uint32_t
// ============================================================================
__global__ void pack_int8_to_uint32_warp_kernel(
    const int8_t* __restrict__ in, 
    uint32_t* __restrict__ out, 
    int spatial_elements, 
    int in_C, 
    int packed_C) 
{
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    const int lane_id = tid % 32;
    const int warp_id = tid / 32;
    
    const int total_warps = spatial_elements * packed_C;

    if (warp_id < total_warps) {
        int spatial_idx = warp_id / packed_C;
        int c_block = warp_id % packed_C;

        int c = c_block * 32 + lane_id;
        int8_t val = 0;
        
        if (c < in_C) {
            val = in[spatial_idx * in_C + c]; 
        }

        // Warp 级硬件归约：一行指令完成 32 个 int8 到 1 个 uint32 的打包
        uint32_t word = __ballot_sync(FULL_MASK, val > 0);

        if (lane_id == 0) {
            out[spatial_idx * packed_C + c_block] = word;
        }
    }
}

// ============================================================================
// 1.5. Fused Pre-processing Kernel: 一步完成 Padding + Permute + Bit-packing
// 接收 NCHW int8, 输出 NHWC packed_C uint32_t
// ============================================================================
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
    
    const int total_spatial = B * padded_H * padded_W;
    const int total_warps = total_spatial * packed_C;

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
// 2. Forward Kernel: 编译期完美展开 & 纯只读缓存寻址
// ============================================================================
template <typename scalar_t, int LUT_NUM>
__global__ void lut_conv_forward_ultimate_kernel(
    const uint32_t* __restrict__ packed_x,  // [B, padded_H, padded_W, packed_C]
    const uint64_t* __restrict__ w_lut,     // [lut_num, out_C] 64-bit 打包真值表
    const int32_t* __restrict__ offsets,    // [lut_num * 6] 绝对偏移量
    const int32_t* __restrict__ shifts,     // [lut_num * 6] bit位移
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

    int32_t y_val = 0; 
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
        uint64_t local_lut = valid_oc ? __ldg(&w_lut[l * out_C + act_oc]) : 0;

        int local_bit = 0;
        if (is_valid_spatial && lane_id < 6) {
            int flat_conn_idx = l * 6 + lane_id;
            
            // 纯只读缓存寻址 (L1.5/Texture Cache)
            int abs_offset = __ldg(&offsets[flat_conn_idx]); 
            int shift_val  = __ldg(&shifts[flat_conn_idx]);

            uint32_t word = __ldg(&packed_x[base_ptr + abs_offset]);
            local_bit = (word >> shift_val) & 1;
        }

        int current_idx = __ballot_sync(0x3F, local_bit == 1);

        if (is_valid_spatial && valid_oc) {
            y_val += ((local_lut >> current_idx) & 1); 
        }
    }

    if (is_valid_spatial && valid_oc) {
        y[global_warp_idx * out_C + act_oc] = static_cast<scalar_t>(y_val);
    }
}

// ============================================================================
// 3. Backward Kernel: 1-Neighborhood Sparsemax (Top-7 极速动态稀疏路由)
// ============================================================================
template <typename scalar_t>
__global__ void lut_conv_backward_ultimate_kernel(
    const scalar_t* __restrict__ grad_y, 
    const int8_t* __restrict__ x_padded,      
    const float* __restrict__ x_float_nchw,  // 🚀 [新增] 连续浮点输入，用于计算 |x| 的汉明距离
    const uint64_t* __restrict__ w_lut, 
    const int32_t* __restrict__ offsets,
    const int32_t* __restrict__ shifts,
    float* __restrict__ grad_x_fp32,     
    float* __restrict__ grad_w_fp32,    
    int B, int in_C, int H, int W,           // 🚀 [新增] 原始 H, W 用于浮点张量逆向寻址
    int packed_C, int padded_H, int padded_W, int out_C, int OH, int OW,
    int stride, int lut_num, int padding, float tau)    // 🚀 [新增] padding 用于坐标逆向映射
{
    const int lane_id = threadIdx.x; 
    const int warp_id = threadIdx.y; 
    const int global_warp_idx = blockIdx.x * blockDim.y + warp_id; 
    const int oc_base = blockIdx.y * TILE_OC;
    const int tid = warp_id * 32 + lane_id;
    const int act_oc = oc_base + lane_id;
    const bool valid_oc = (act_oc < out_C);

    __shared__ float s_grad_w[64][TILE_OC];

    bool is_valid_spatial = (global_warp_idx < B * OH * OW);
    const int ow = is_valid_spatial ? (global_warp_idx % OW) : 0;
    const int oh = is_valid_spatial ? ((global_warp_idx / OW) % OH) : 0;
    const int b = is_valid_spatial ? (global_warp_idx / (OH * OW)) : 0;

    float dy_val = (is_valid_spatial && valid_oc) ? static_cast<float>(grad_y[global_warp_idx * out_C + act_oc]) : 0.0f;

    // Backward 对应的原生 int8 空间指针基址
    int spatial_base_ptr = 0;
    if (is_valid_spatial) {
        spatial_base_ptr = (b * padded_H + (oh * stride)) * padded_W + (ow * stride);
    }

    for (int l = 0; l < lut_num; ++l) {
        // 无 Bank Conflict 的显存清理
        for (int i = tid; i < 64 * TILE_OC; i += WARPS_PER_BLOCK * 32) {
            (&s_grad_w[0][0])[i] = 0.0f;
        }
        __syncthreads(); 

        uint64_t local_lut = valid_oc ? w_lut[l * out_C + act_oc] : 0;

        int local_x_ptr = -1; 
        int local_bit = 0;
        float f_val = 0.0f; // 存储连续的浮点值

        if (is_valid_spatial && lane_id < 6) {
            int flat_conn_idx = l * 6 + lane_id;
            int abs_offset = offsets[flat_conn_idx]; 
            int shift_val  = shifts[flat_conn_idx];

            int spatial_off = abs_offset / packed_C;
            int c_word = abs_offset % packed_C;
            int c = c_word * 32 + shift_val;

            local_x_ptr = (spatial_base_ptr + spatial_off) * in_C + c;
            if (x_padded[local_x_ptr] > 0) local_bit = 1;

            // 📍 极速逆向映射：从 padded 的 abs_offset 找回原始的 NCHW float 值
            int ph = spatial_off / padded_W;
            int pw = spatial_off % padded_W;
            int h = ph - padding;
            int w = pw - padding;
            if (h >= 0 && h < H && w >= 0 && w < W && c < in_C) {
                int nchw_idx = ((b * in_C + c) * H + h) * W + w;
                f_val = x_float_nchw[nchw_idx];
            }
        }

        int current_idx = __ballot_sync(0x3F, local_bit == 1);
        
        int x_indices[6];
        float x_f[6];
        #pragma unroll
        for (int j = 0; j < 6; ++j) {
            // 瞬间将 6 个输入的地址和浮点值广播给整个 Warp 的 32 个线程
            x_indices[j] = __shfl_sync(0xFFFFFFFF, local_x_ptr, j);
            x_f[j] = __shfl_sync(0xFFFFFFFF, f_val, j);
        }

        // ==============================================================
        // 🚀 核心优化：Top-7 Sparsemax (1-Neighborhood) 计算与分发
        // ==============================================================
        if (is_valid_spatial && valid_oc && dy_val != 0.0f) {
            float v[7]; 
            // 1. 找出 6 个输入比特中的最大绝对值 (最大惩罚代价)
            float max_mag = 1.0; // 给一个极小的底，防止全0
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

        // Boolean Derivative (x 的梯度，保持完美简洁)
        if (is_valid_spatial && valid_oc) {
            #pragma unroll
            for (int k = 0; k < 6; ++k) {
                int idx1 = current_idx | (1 << k);
                int idx0 = current_idx & ~(1 << k);
                
                int bit1 = (local_lut >> idx1) & 1;
                int bit0 = (local_lut >> idx0) & 1;
                float dx_val = dy_val * (bit1 - bit0);

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
// 4. C++ API 层 & 宏分发 (Macro Dispatch)
// ============================================================================

// (1) Bit-Packing API
// (1) Bit-Packing API
torch::Tensor pack_padded_inputs(torch::Tensor padded_x) {
    TORCH_CHECK(padded_x.scalar_type() == torch::kInt8, "x MUST be torch.int8!");
    
    auto x_c = padded_x.contiguous(); 
    
    int B = x_c.size(0); int p_H = x_c.size(1); int p_W = x_c.size(2); int in_C = x_c.size(3);
    int packed_C = CEIL_DIV(in_C, 32);
    int spatial_elements = B * p_H * p_W;
    int total_warps = spatial_elements * packed_C;
    
    auto packed_x = torch::zeros({B, p_H, p_W, packed_C}, x_c.options().dtype(torch::kInt32).memory_format(at::MemoryFormat::Contiguous));
    
    int threads_per_block = 128;
    int blocks = CEIL_DIV(total_warps * 32, threads_per_block); 
    
    pack_int8_to_uint32_warp_kernel<<<blocks, threads_per_block>>>(
        x_c.data_ptr<int8_t>(),
        reinterpret_cast<uint32_t*>(packed_x.data_ptr<int32_t>()), 
        spatial_elements, in_C, packed_C
    );
    return packed_x;
}

// 供宏分发使用的模版调用包裹器
#define DISPATCH_LUT_KERNEL(LUT_VAL) \
    case LUT_VAL: \
        lut_conv_forward_ultimate_kernel<float, LUT_VAL><<<blocks, threads>>>( \
            reinterpret_cast<uint32_t*>(packed_x.data_ptr<int32_t>()), \
            reinterpret_cast<uint64_t*>(w_packed.data_ptr<int64_t>()), \
            offsets.data_ptr<int32_t>(), shifts.data_ptr<int32_t>(), \
            y.data_ptr<float>(), \
            B, packed_C, padded_H, padded_W, out_C, OH, OW, stride \
        ); \
        break;

// (2) Forward Conv API
torch::Tensor forward_implicit_ultimate(
    torch::Tensor packed_x, torch::Tensor w_packed, torch::Tensor offsets, torch::Tensor shifts,
    int B, int padded_H, int padded_W, int OH, int OW, int stride) 
{
    TORCH_CHECK(w_packed.scalar_type() == torch::kInt64, "w_packed MUST be int64.");
    
    int packed_C = packed_x.size(3);
    int lut_num = w_packed.size(0); 
    int out_C = w_packed.size(1); 
    
    auto y = torch::empty({B, out_C, OH, OW}, packed_x.options().dtype(torch::kFloat32).memory_format(at::MemoryFormat::ChannelsLast));
    
    dim3 threads(32, WARPS_PER_BLOCK); 
    dim3 blocks(CEIL_DIV(B * OH * OW, WARPS_PER_BLOCK), CEIL_DIV(out_C, TILE_OC));
    
    // 静态分发网络初始化时可能出现的所有 LUT 数量
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
    
    return y;
}

// (3) Backward API
std::vector<torch::Tensor> backward_ultimate(
    torch::Tensor grad_y_nhwc, 
    torch::Tensor x_int8_padded, 
    torch::Tensor x_float_nchw,   // 🚀 [新增参数] 
    torch::Tensor w_packed, 
    torch::Tensor offsets, 
    torch::Tensor shifts, 
    torch::Tensor tau_tensor,
    int padding,                  // 🚀 [新增参数]
    int B, int padded_H, int padded_W, int OH, int OW, int stride)
{
    TORCH_CHECK(x_int8_padded.scalar_type() == torch::kInt8, "x_padded MUST be int8.");
    TORCH_CHECK(w_packed.scalar_type() == torch::kInt64, "w_packed MUST be int64.");
    TORCH_CHECK(x_float_nchw.scalar_type() == torch::kFloat32, "x_float MUST be float32.");
    
    auto grad_y_c = grad_y_nhwc.contiguous();
    auto x_int8_c = x_int8_padded.contiguous();
    auto x_float_c = x_float_nchw.contiguous();

    float tau = tau_tensor.item<float>();
    int in_C = x_int8_c.size(3);
    int packed_C = CEIL_DIV(in_C, 32);
    int lut_num = w_packed.size(0);
    int out_C = w_packed.size(1);
    
    int H = x_float_c.size(2);
    int W = x_float_c.size(3);

    auto grad_x_padded = torch::zeros_like(x_int8_c, x_int8_c.options().dtype(torch::kFloat32).memory_format(at::MemoryFormat::Contiguous));
    auto grad_w = torch::zeros({lut_num, 64, out_C}, w_packed.options().dtype(torch::kFloat32).memory_format(at::MemoryFormat::Contiguous));

    dim3 threads(32, WARPS_PER_BLOCK); 
    dim3 blocks(CEIL_DIV(B * OH * OW, WARPS_PER_BLOCK), CEIL_DIV(out_C, TILE_OC));

    AT_DISPATCH_FLOATING_TYPES_AND2(at::ScalarType::Half, at::ScalarType::BFloat16, grad_y_c.scalar_type(), "lut_bw_ultimate", [&] {
        lut_conv_backward_ultimate_kernel<scalar_t><<<blocks, threads>>>(
            grad_y_c.data_ptr<scalar_t>(), 
            x_int8_c.data_ptr<int8_t>(), 
            x_float_c.data_ptr<float>(), 
            reinterpret_cast<uint64_t*>(w_packed.data_ptr<int64_t>()), 
            offsets.data_ptr<int32_t>(), 
            shifts.data_ptr<int32_t>(),
            grad_x_padded.data_ptr<float>(), 
            grad_w.data_ptr<float>(),
            B, in_C, H, W, packed_C, padded_H, padded_W, out_C, OH, OW, 
            stride, lut_num, padding, tau
        );
    });

    return {grad_x_padded, grad_w};
}

torch::Tensor fused_pre_process(
    torch::Tensor x_nchw, 
    int pad_top, int pad_bottom, int pad_left, int pad_right) 
{
    TORCH_CHECK(x_nchw.scalar_type() == torch::kInt8, "x MUST be torch.int8!");
    auto x_contig = x_nchw.contiguous(at::MemoryFormat::Contiguous);
    
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
    m.def("conv_bffb_forward", &forward_implicit_ultimate, "Ultimate Zero-Branch Pipelined LUT Forward");
    m.def("conv_bffb_backward", &backward_ultimate, "Ultimate Pipelined LUT Backward");
}

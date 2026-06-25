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
// 1. Warp-Level Bit-Packing Kernels
// ============================================================================
__global__ void pack_int8_to_uint32_warp_kernel(
    const int8_t* __restrict__ in, uint32_t* __restrict__ out, 
    int spatial_elements, int in_C, int packed_C) 
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
        
        if (c < in_C) val = in[spatial_idx * in_C + c]; 
        uint32_t word = __ballot_sync(FULL_MASK, val > 0);
        if (lane_id == 0) out[spatial_idx * packed_C + c_block] = word;
    }
}

__global__ void fused_pad_permute_pack_kernel(
    const int8_t* __restrict__ in_nchw, uint32_t* __restrict__ out_nhwc_packed, 
    int B, int in_C, int H, int W, int pad_top, int pad_bottom, int pad_left, int pad_right,
    int padded_H, int padded_W, int packed_C) 
{
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    const int lane_id = tid % 32;
    const int warp_id = tid / 32;
    const int total_spatial = B * padded_H * padded_W;
    const int total_warps = total_spatial * packed_C;

    if (warp_id < total_warps) {
        int spatial_idx = warp_id / packed_C;
        int c_block = warp_id % packed_C;
        int b  = spatial_idx / (padded_H * padded_W);
        int ph = (spatial_idx / padded_W) % padded_H;
        int pw = spatial_idx % padded_W;
        int h = ph - pad_top;
        int w = pw - pad_left;
        int c = c_block * 32 + lane_id;
        
        int8_t val = 0;
        if (h >= 0 && h < H && w >= 0 && w < W && c < in_C) {
            val = in_nchw[((b * in_C + c) * H + h) * W + w];
        }
        uint32_t word = __ballot_sync(FULL_MASK, val > 0);
        if (lane_id == 0) out_nhwc_packed[spatial_idx * packed_C + c_block] = word;
    }
}

// ============================================================================
// 2. 模板化 Forward Kernel: 支持任意 LUT_K
// ============================================================================
template <typename scalar_t, int LUT_K>
__global__ void lut_conv_forward_ultimate_kernel(
    const uint32_t* __restrict__ packed_x,  
    const uint64_t* __restrict__ w_lut,     
    const int32_t* __restrict__ offsets,    
    const int32_t* __restrict__ shifts,     
    scalar_t* __restrict__ y,         
    int B, int packed_C, int padded_H, int padded_W, 
    int out_C, int OH, int OW, int stride, int lut_num)
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

    int base_ptr = 0;
    if (is_valid_spatial) {
        base_ptr = ((b * padded_H + (oh * stride)) * padded_W + (ow * stride)) * packed_C;
    }

    for (int l = 0; l < lut_num; ++l) {
        uint64_t local_lut = valid_oc ? __ldg(&w_lut[l * out_C + act_oc]) : 0;

        int local_bit = 0;
        if (is_valid_spatial && lane_id < LUT_K) { 
            int flat_conn_idx = l * LUT_K + lane_id;
            
            int abs_offset = __ldg(&offsets[flat_conn_idx]); 
            int shift_val  = __ldg(&shifts[flat_conn_idx]);

            uint32_t word = __ldg(&packed_x[base_ptr + abs_offset]);
            local_bit = (word >> shift_val) & 1;
        }

        int ballot_idx = __ballot_sync(FULL_MASK, local_bit == 1);
        int current_idx = 0;
        #pragma unroll
        for (int j = 0; j < LUT_K; ++j) {
            current_idx |= ((ballot_idx >> j) & 1) << (LUT_K - 1 - j);
        }

        if (is_valid_spatial && valid_oc) {
            y_val += ((local_lut >> current_idx) & 1); 
        }
    }

    if (is_valid_spatial && valid_oc) {
        y[global_warp_idx * out_C + act_oc] = static_cast<scalar_t>(y_val);
    }
}

// ============================================================================
// 3. 模板化 Backward Kernel: 支持任意 LUT_K
// ============================================================================
template <typename scalar_t, int LUT_K>
__global__ void lut_conv_backward_ultimate_kernel(
    const scalar_t* __restrict__ grad_y, 
    const int8_t* __restrict__ x_padded,      
    const uint64_t* __restrict__ w_lut, 
    const int32_t* __restrict__ offsets,
    const int32_t* __restrict__ shifts,
    float* __restrict__ grad_x_fp32,     
    float* __restrict__ grad_w_fp32,    
    int B, int in_C, int packed_C, int padded_H, int padded_W, int out_C, int OH, int OW,
    int stride, int lut_num)
{
    constexpr int LUT_SIZE = 1 << LUT_K;
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

    int spatial_base_ptr = 0;
    if (is_valid_spatial) {
        spatial_base_ptr = (b * padded_H + (oh * stride)) * padded_W + (ow * stride);
    }

    for (int l = 0; l < lut_num; ++l) {
        for (int i = tid; i < 64 * TILE_OC; i += WARPS_PER_BLOCK * 32) {
            (&s_grad_w[0][0])[i] = 0.0f;
        }
        __syncthreads(); 

        uint64_t local_lut = valid_oc ? w_lut[l * out_C + act_oc] : 0;
        int local_x_ptr = -1; 
        int local_bit = 0;

        if (is_valid_spatial && lane_id < LUT_K) { 
            int flat_conn_idx = l * LUT_K + lane_id;
            int abs_offset = offsets[flat_conn_idx]; 
            int shift_val  = shifts[flat_conn_idx];
            int spatial_off = abs_offset / packed_C;
            int c_word = abs_offset % packed_C;
            int c = c_word * 32 + shift_val;

            local_x_ptr = (spatial_base_ptr + spatial_off) * in_C + c;
            if (x_padded[local_x_ptr] > 0) local_bit = 1;
        }

        int ballot_idx = __ballot_sync(FULL_MASK, local_bit == 1);
        int current_idx = 0;
        #pragma unroll
        for (int j = 0; j < LUT_K; ++j) {
            current_idx |= ((ballot_idx >> j) & 1) << (LUT_K - 1 - j);
        }
        
        int x_indices[6]; 
        #pragma unroll
        for (int j = 0; j < LUT_K; ++j) x_indices[j] = __shfl_sync(FULL_MASK, local_x_ptr, j);

        if (is_valid_spatial && dy_val != 0.0f) {
            atomicAdd(&s_grad_w[current_idx][lane_id], dy_val);
        }

        if (is_valid_spatial) {
            #pragma unroll
            for (int k = 0; k < LUT_K; ++k) { 
                int bit_mask = 1 << (LUT_K - 1 - k);
                int idx1 = current_idx | bit_mask;
                int idx0 = current_idx & ~bit_mask;
                
                int bit1 = (local_lut >> idx1) & 1;
                int bit0 = (local_lut >> idx0) & 1;
                float dx_val = valid_oc ? dy_val * (bit1 - bit0) : 0.0f;

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

        for (int i = tid; i < LUT_SIZE * TILE_OC; i += WARPS_PER_BLOCK * 32) {
            int bit = i / TILE_OC; int toc = i % TILE_OC; int c_act = oc_base + toc;
            if (c_act < out_C) {
                float val = (&s_grad_w[0][0])[i];
                if (val != 0.0f) atomicAdd(&grad_w_fp32[((l * LUT_SIZE + bit) * out_C) + c_act], val);
            }
        }
        __syncthreads(); 
    }
}

// ============================================================================
// 4. API & Macro Dispatch
// ============================================================================
torch::Tensor pack_padded_inputs(torch::Tensor padded_x) {
    TORCH_CHECK(padded_x.scalar_type() == torch::kInt8, "x MUST be torch.int8!");
    auto x_c = padded_x.contiguous(); 
    int B = x_c.size(0), p_H = x_c.size(1), p_W = x_c.size(2), in_C = x_c.size(3);
    int packed_C = CEIL_DIV(in_C, 32);
    int spatial_elements = B * p_H * p_W;
    
    auto packed_x = torch::zeros({B, p_H, p_W, packed_C}, x_c.options().dtype(torch::kInt32).memory_format(at::MemoryFormat::Contiguous));
    dim3 blocks(CEIL_DIV(spatial_elements * packed_C * 32, 128));
    
    pack_int8_to_uint32_warp_kernel<<<blocks, 128>>>(x_c.data_ptr<int8_t>(), reinterpret_cast<uint32_t*>(packed_x.data_ptr<int32_t>()), spatial_elements, in_C, packed_C);
    return packed_x;
}

torch::Tensor fused_pre_process(torch::Tensor x_nchw, int pad_top, int pad_bottom, int pad_left, int pad_right) {
    auto x_contig = x_nchw.contiguous(at::MemoryFormat::Contiguous);
    int B = x_contig.size(0), in_C = x_contig.size(1), H = x_contig.size(2), W = x_contig.size(3);
    int padded_H = H + pad_top + pad_bottom, padded_W = W + pad_left + pad_right;
    int packed_C = CEIL_DIV(in_C, 32);
    auto packed_x = torch::zeros({B, padded_H, padded_W, packed_C}, x_contig.options().dtype(torch::kInt32));
    
    dim3 blocks(CEIL_DIV(B * padded_H * padded_W * packed_C * 32, 128)); 
    fused_pad_permute_pack_kernel<<<blocks, 128>>>(x_contig.data_ptr<int8_t>(), reinterpret_cast<uint32_t*>(packed_x.data_ptr<int32_t>()), B, in_C, H, W, pad_top, pad_bottom, pad_left, pad_right, padded_H, padded_W, packed_C);
    
    return packed_x;
}

#define DISPATCH_BINARY_FW(LUT_K) \
    case LUT_K: \
        lut_conv_forward_ultimate_kernel<float, LUT_K><<<blocks, threads>>>( \
            reinterpret_cast<uint32_t*>(packed_x.data_ptr<int32_t>()), \
            reinterpret_cast<uint64_t*>(w_packed.data_ptr<int64_t>()), \
            offsets.data_ptr<int32_t>(), shifts.data_ptr<int32_t>(), \
            y.data_ptr<float>(), B, packed_C, padded_H, padded_W, out_C, OH, OW, stride, lut_num \
        ); break;

torch::Tensor forward_implicit_ultimate(
    torch::Tensor packed_x, torch::Tensor w_packed, torch::Tensor offsets, torch::Tensor shifts,
    int B, int padded_H, int padded_W, int OH, int OW, int stride, int K) 
{
    int packed_C = packed_x.size(3);
    int lut_num = offsets.size(0) / K; 
    int out_C = w_packed.size(1); 
    auto y = torch::empty({B, out_C, OH, OW}, packed_x.options().dtype(torch::kFloat32).memory_format(at::MemoryFormat::ChannelsLast));
    
    dim3 threads(32, WARPS_PER_BLOCK), blocks(CEIL_DIV(B * OH * OW, WARPS_PER_BLOCK), CEIL_DIV(out_C, TILE_OC));
    
    switch (K) {
        DISPATCH_BINARY_FW(2) 
        DISPATCH_BINARY_FW(4) 
        DISPATCH_BINARY_FW(5) 
        DISPATCH_BINARY_FW(6)
        default: TORCH_CHECK(false, "Unsupported LUT K: ", K);
    }
    return y;
}

#define DISPATCH_BINARY_BW(LUT_K) \
    case LUT_K: \
        lut_conv_backward_ultimate_kernel<scalar_t, LUT_K><<<blocks, threads>>>( \
            grad_y_c.data_ptr<scalar_t>(), x_int8_c.data_ptr<int8_t>(), \
            reinterpret_cast<uint64_t*>(w_packed.data_ptr<int64_t>()), \
            offsets.data_ptr<int32_t>(), shifts.data_ptr<int32_t>(), \
            grad_x_padded.data_ptr<float>(), grad_w.data_ptr<float>(), \
            B, in_C, packed_C, padded_H, padded_W, out_C, OH, OW, stride, lut_num \
        ); break;

std::vector<torch::Tensor> backward_ultimate(
    torch::Tensor grad_y_nhwc, torch::Tensor x_int8_padded, torch::Tensor w_packed, 
    torch::Tensor offsets, torch::Tensor shifts, 
    int B, int padded_H, int padded_W, int OH, int OW, int stride, int K) 
{
    auto grad_y_c = grad_y_nhwc.contiguous(), x_int8_c = x_int8_padded.contiguous();
    int in_C = x_int8_c.size(3), packed_C = CEIL_DIV(in_C, 32);
    int lut_num = offsets.size(0) / K, out_C = w_packed.size(1);
    
    auto grad_x_padded = torch::zeros_like(x_int8_c, x_int8_c.options().dtype(torch::kFloat32));
    auto grad_w = torch::zeros({lut_num, 1 << K, out_C}, w_packed.options().dtype(torch::kFloat32));

    dim3 threads(32, WARPS_PER_BLOCK), blocks(CEIL_DIV(B * OH * OW, WARPS_PER_BLOCK), CEIL_DIV(out_C, TILE_OC));

    // 使用标准的安全宏，防止编译失败
    AT_DISPATCH_FLOATING_TYPES_AND2(at::ScalarType::Half, at::ScalarType::BFloat16, grad_y_c.scalar_type(), "lut_bw", [&] {
        switch(K) {
            DISPATCH_BINARY_BW(2) 
            DISPATCH_BINARY_BW(4) 
            DISPATCH_BINARY_BW(5) 
            DISPATCH_BINARY_BW(6)
            default: TORCH_CHECK(false, "Unsupported LUT K: ", K);
        }
    });
    return {grad_x_padded, grad_w};
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("pack_padded_inputs", &pack_padded_inputs);
    m.def("fused_pre_process", &fused_pre_process);
    m.def("forward", &forward_implicit_ultimate);
    m.def("backward", &backward_ultimate);
}
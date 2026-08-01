import torch
import math
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
from torch.cuda.amp import custom_fwd, custom_bwd
import lut_cuda_grouped
import lut_cuda_grouped_binary
import lut_conv_fp32_cuda
import lut_conv_binary_cuda
import lut_conv_bffb_cuda
import lut_conv_bafw_cuda

# ============================================================================
# 1. 定义算子库 (DEF) - 定义签名
# ============================================================================
# 我们创建一个名为 "lut_lib" 的库
lut_lib_def = torch.library.Library("lut_lib", "DEF")

# 算子 1: Standard Grouped LUT
lut_lib_def.define("grouped_lut(Tensor x, Tensor w, int groups) -> Tensor")
lut_lib_def.define("grouped_lut_bw(Tensor grad_y, Tensor x, Tensor w, int groups) -> Tensor[]")

# 算子 2: Binary Grouped LUT
lut_lib_def.define("binary_lut(Tensor x, Tensor w, int groups) -> Tensor")
lut_lib_def.define("binary_lut_bw(Tensor grad_y, Tensor x, Tensor w, int groups) -> Tensor[]")

# 算子 3: Ultra Fused LUT-Conv (我们最新开发的极致卷积)
lut_lib_def.define("pack_padded_inputs(Tensor padded_x) -> Tensor")
lut_lib_def.define("binary_pre_process(Tensor padded_x, int pad_top, int pad_bottom, int pad_left, int pad_right) -> Tensor")
lut_lib_def.define("binary_conv(Tensor packed_x, Tensor w_packed, Tensor offsets, Tensor shifts, int B, int padded_H, int padded_W, int OH, int OW, int stride) -> Tensor")
lut_lib_def.define("binary_conv_bw(Tensor grad_y, Tensor x_padded, Tensor w_packed, Tensor offsets, Tensor shifts, int B, int padded_H, int padded_W, int OH, int OW, int stride) -> Tensor[]")

# 算子 4: 
lut_lib_def.define("bffb_conv(Tensor packed_x, Tensor w_packed, Tensor offsets, Tensor shifts, int B, int padded_H, int padded_W, int OH, int OW, int stride) -> Tensor")
lut_lib_def.define("bffb_conv_bw(Tensor grad_y, Tensor x_padded, Tensor x_float_nchw, Tensor w_packed, Tensor offsets, Tensor shifts, Tensor tau, int padding, int B, int padded_H, int padded_W, int OH, int OW, int stride) -> Tensor[]")

lut_lib_def.define("bafw_conv(Tensor packed_x, Tensor w_soft, Tensor offsets, Tensor shifts, int B, int padded_H, int padded_W, int OH, int OW, int stride) -> Tensor")
lut_lib_def.define("bafw_conv_bw(Tensor grad_y, Tensor packed_x, Tensor x_float_nchw, Tensor w_soft, Tensor offsets, Tensor shifts, Tensor tau, int padding, int B, int padded_H, int padded_W, int OH, int OW, int stride) -> Tensor[]")
lut_lib_def.define("fused_pre_process(Tensor x_nchw, int pad_top, int pad_bottom, int pad_left, int pad_right) -> Tensor")

lut_lib_def.define("fused_lut_conv(Tensor x, Tensor w, int groups, int K, int stride, int pad_H, int pad_W, int OH, int OW) -> Tensor")
lut_lib_def.define("fused_lut_conv_bw(Tensor grad_y, Tensor x, Tensor w, int groups, int K, int stride, int pad_H, int pad_W, int OH, int OW) -> Tensor[]")
# ============================================================================
# 2. CUDA 实现绑定 (IMPL CUDA) - 链接真实 Kernel
# ============================================================================
lut_lib_cuda = torch.library.Library("lut_lib", "IMPL", "CUDA")

# 绑定 1
lut_lib_cuda.impl("grouped_lut", lut_cuda_grouped.forward)
lut_lib_cuda.impl("grouped_lut_bw", lut_cuda_grouped.backward)

# 绑定 2
lut_lib_cuda.impl("binary_lut", lut_cuda_grouped_binary.forward)
lut_lib_cuda.impl("binary_lut_bw", lut_cuda_grouped_binary.backward)

# 绑定 3
lut_lib_cuda.impl("pack_padded_inputs", lut_conv_binary_cuda.pack_padded_inputs)
lut_lib_cuda.impl("binary_pre_process", lut_conv_binary_cuda.binary_pre_process)
lut_lib_cuda.impl("binary_conv", lut_conv_binary_cuda.conv_binary_forward)
lut_lib_cuda.impl("binary_conv_bw", lut_conv_binary_cuda.conv_binary_backward) 

# 绑定 4
lut_lib_cuda.impl("bffb_conv", lut_conv_bffb_cuda.conv_bffb_forward)
lut_lib_cuda.impl("bffb_conv_bw", lut_conv_bffb_cuda.conv_bffb_backward) 

lut_lib_cuda.impl("bafw_conv", lut_conv_bafw_cuda.conv_bafw_forward)
lut_lib_cuda.impl("bafw_conv_bw", lut_conv_bafw_cuda.conv_bafw_backward) 
lut_lib_cuda.impl("fused_pre_process", lut_conv_bafw_cuda.fused_pre_process) 

lut_lib_cuda.impl("fused_lut_conv", lut_conv_fp32_cuda.forward)
lut_lib_cuda.impl("fused_lut_conv_bw", lut_conv_fp32_cuda.backward)

# ============================================================================
# 3. Meta 实现绑定 (IMPL Meta) - 支持 torch.compile 追踪形状
# ============================================================================
lut_lib_meta = torch.library.Library("lut_lib", "IMPL", "Meta")

# Meta 1 & 2 (逻辑相同)
def grouped_lut_meta(x, w, groups):
    return torch.empty((w.size(0), w.size(1), x.size(3)), dtype=torch.float32, device=x.device)

def grouped_lut_bw_meta(grad_y, x, w, groups):
    grad_x_meta = torch.empty(x.size(), dtype=torch.float32, device=x.device)
    grad_w_meta = torch.empty(w.size(), dtype=torch.float32, device=w.device)
    return [grad_x_meta, grad_w_meta]

lut_lib_meta.impl("grouped_lut", grouped_lut_meta)
lut_lib_meta.impl("grouped_lut_bw", grouped_lut_bw_meta)
lut_lib_meta.impl("binary_lut", grouped_lut_meta)
lut_lib_meta.impl("binary_lut_bw", grouped_lut_bw_meta)

# Meta 3 (Ultra Fused Conv)

def binary_pre_process_meta(x_nchw, pad_top, pad_bottom, pad_left, pad_right):
    # 1. 获取原生输入的维度
    B = x_nchw.size(0)
    in_C = x_nchw.size(1)
    H = x_nchw.size(2)
    W = x_nchw.size(3)
    
    # 2. 计算物理 padding 后的空间分辨率
    padded_H = H + pad_top + pad_bottom
    padded_W = W + pad_left + pad_right
    
    # 3. 计算按 32-bit 打包后的通道数
    packed_C = (in_C + 31) // 32
    
    # 4. 返回一个只有“形状”和“步长规则”的空壳张量给 Dynamo
    # 注意：千万不要加 memory_format=torch.channels_last！
    # 因为我们在 C++ 里是以 {B, padded_H, padded_W, packed_C} 连续分配的，
    # 它在内存里就是绝对连续的 (Contiguous)。
    return torch.empty(
        (B, padded_H, padded_W, packed_C),
        dtype=torch.int32,
        device=x_nchw.device,
        memory_format=torch.contiguous_format
    )


def pack_padded_inputs_meta(padded_x):
    packed_C = (padded_x.size(3) + 31) // 32
    return torch.empty((padded_x.size(0), padded_x.size(1), padded_x.size(2), packed_C), dtype=torch.int32, device=padded_x.device)

def binary_conv_meta(packed_x, w_packed, offsets, shifts, B, padded_H, padded_W, OH, OW, stride):
    out_C = w_packed.size(1)
    return torch.empty((B, out_C, OH, OW), dtype=torch.float32, device=packed_x.device, memory_format=torch.channels_last)

def binary_conv_bw_meta(grad_y, x_padded, w_packed, offsets, shifts, B, padded_H, padded_W, OH, OW, stride):
    gx = torch.empty_like(x_padded, dtype=torch.float32, memory_format=torch.contiguous_format)
    gw = torch.empty((w_packed.size(0), 64, w_packed.size(1)), dtype=torch.float32, device=w_packed.device)
    return [gx, gw]

lut_lib_meta.impl("binary_pre_process", binary_pre_process_meta)
lut_lib_meta.impl("pack_padded_inputs", pack_padded_inputs_meta)
lut_lib_meta.impl("binary_conv", binary_conv_meta)
lut_lib_meta.impl("binary_conv_bw", binary_conv_bw_meta)

def bffb_conv_meta(packed_x, w_packed, offsets, shifts, B, padded_H, padded_W, OH, OW, stride):
    out_C = w_packed.size(1)
    return torch.empty((B, out_C, OH, OW), dtype=torch.float32, device=packed_x.device, memory_format=torch.channels_last)

def bffb_conv_bw_meta(grad_y, x_padded, x_float_nchw, w_packed, offsets, shifts, tau, padding, B, padded_H, padded_W, OH, OW, stride):
    gx = torch.empty_like(x_padded, dtype=torch.float32, memory_format=torch.contiguous_format)
    gw = torch.empty((w_packed.size(0), 64, w_packed.size(1)), dtype=torch.float32, device=w_packed.device)
    return [gx, gw]

def fused_pre_process_meta(x_nchw, pad_top, pad_bottom, pad_left, pad_right):
    B, in_C, H, W = x_nchw.shape
    padded_H = H + pad_top + pad_bottom
    padded_W = W + pad_left + pad_right
    packed_C = (in_C + 31) // 32
    
    return torch.empty((B, padded_H, padded_W, packed_C), dtype=torch.int32, device=x_nchw.device, memory_format=torch.contiguous_format)

def bafw_conv_meta(packed_x, w_soft, offsets, shifts, B, padded_H, padded_W, OH, OW, stride):
    out_C = w_soft.size(2)
    return torch.empty((B, out_C, OH, OW), dtype=torch.float32, device=packed_x.device, memory_format=torch.channels_last)

def bafw_conv_bw_meta(grad_y, packed_x, x_float, w_soft, offsets, shifts, tau, padding, B, padded_H, padded_W, OH, OW, stride):
    gx = torch.empty_like(x_float, dtype=torch.float32, memory_format=torch.contiguous_format)
    gw = torch.empty_like(w_soft, dtype=torch.float32,  memory_format=torch.contiguous_format)
    return [gx, gw]

lut_lib_meta.impl("bffb_conv", bffb_conv_meta)
lut_lib_meta.impl("bffb_conv_bw", bffb_conv_bw_meta)

lut_lib_meta.impl("bafw_conv", bafw_conv_meta)
lut_lib_meta.impl("bafw_conv_bw", bafw_conv_bw_meta)
lut_lib_meta.impl("fused_pre_process", fused_pre_process_meta)

def fused_lut_conv_meta(x, w, groups, K, stride, pad_H, pad_W, OH, OW):
    # x: [B, in_C, H, W], w: [lut_num, 64, out_C]
    B = x.size(0)
    out_C = w.size(2)
    # 极度重要：必须在 Meta 阶段就告诉 PyTorch 这是 ChannelsLast，
    # 否则底层 CUDA 可能会拿到非连续的内存块导致崩溃或性能断崖。
    return torch.empty(
        (B, out_C, OH, OW), 
        dtype=x.dtype, 
        device=x.device, 
        memory_format=torch.channels_last
    )

def fused_lut_conv_bw_meta(grad_y, x, w, groups, K, stride, pad_H, pad_W, OH, OW):
    grad_x_meta = torch.empty_like(x, memory_format=torch.channels_last)
    grad_w_meta = torch.empty_like(w)
    return [grad_x_meta, grad_w_meta]

lut_lib_meta.impl("fused_lut_conv", fused_lut_conv_meta)
lut_lib_meta.impl("fused_lut_conv_bw", fused_lut_conv_bw_meta)


class LUT6Function(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, w, groups):
        # x: [B, R, G, L, 6], w: [O, L, 64]
        B, R, G, L, _ = x.shape
        BR = B * R
        x_soa = x.view(BR, G, L, 6).permute(1, 2, 3, 0).contiguous()
        w_contiguous = w.contiguous()

        y_reshaped = torch.ops.lut_lib.binary_lut(x_soa, w_contiguous, groups)

        ctx.save_for_backward(x_soa.to(torch.int8), w_contiguous)
        ctx.groups = groups
        #ctx.shape_info = (B, R, G, L)

        O = w.shape[0]
        y = y_reshaped.view(O, L, B, R).permute(2, 3, 0, 1).contiguous()
        return y

    @staticmethod
    def backward(ctx, grad_y):
        x_soa_int8, w_contiguous = ctx.saved_tensors
        groups = ctx.groups
        x_soa = x_soa_int8.to(grad_y.dtype)
        B, R, O, L = grad_y.shape
        G = x_soa.shape[0]
        BR = B * R
        grad_y_reshaped = grad_y.permute(2, 3, 0, 1).reshape(O, L, BR).contiguous()
        grad_x_soa, grad_w = torch.ops.lut_lib.binary_lut_bw(grad_y_reshaped, x_soa, w_contiguous, groups)
        grad_x = grad_x_soa.view(G, L, 6, B, R).permute(3, 4, 0, 1, 2).contiguous()

        return grad_x, grad_w, None

def pack_weights_to_int64(w_q):
    w_int64 = w_q.to(torch.int64)
    shifts = torch.arange(64, dtype=torch.int64, device=w_q.device).view(1, 64, 1)
    w_packed = torch.sum(w_int64 << shifts, dim=1)
    return w_packed

class LUTBinaryConvFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, w_q, offsets, shifts, groups, K, stride, padding):
        ctx.in_dtype = x.dtype
        ctx.w_dtype = w_q.dtype
        ctx.params = (groups, K, stride, padding)
        
        B, in_C, H, W = x.shape
        padded_H = H + 2 * padding
        padded_W = W + 2 * padding
        OH = (H + 2 * padding - K) // stride + 1
        OW = (W + 2 * padding - K) // stride + 1

        x_int8 = x.to(torch.int8).contiguous() if x.dtype != torch.int8 else x.contiguous()
        packed_x = torch.ops.lut_lib.binary_pre_process(
            x_int8, padding, padding, padding, padding
        )

        w_packed = pack_weights_to_int64(w_q)

        y = torch.ops.lut_lib.binary_conv(
            packed_x, w_packed, offsets, shifts,
            B, padded_H, padded_W, OH, OW, stride
        )

        ctx.save_for_backward(x_int8, w_packed, offsets, shifts)

        return y

    @staticmethod
    def backward(ctx, grad_y):
        x_int8_nchw, w_packed, offsets, shifts = ctx.saved_tensors
        groups, K, stride, padding = ctx.params
        
        grad_y_nhwc = grad_y.permute(0, 2, 3, 1).contiguous()
        
        B, in_C, H, W = x_int8_nchw.shape
        padded_H = H + 2 * padding
        padded_W = W + 2 * padding
        OH, OW = grad_y_nhwc.shape[1], grad_y_nhwc.shape[2]

        if padding > 0:
            x_int8_padded_nchw = F.pad(x_int8_nchw, (padding, padding, padding, padding))
        else:
            x_int8_padded_nchw = x_int8_nchw
        x_int8_padded_nhwc = x_int8_padded_nchw.permute(0, 2, 3, 1).contiguous()

        grad_x_padded_fp32, grad_w_fp32 = torch.ops.lut_lib.binary_conv_bw(
            grad_y_nhwc, x_int8_padded_nhwc, w_packed, offsets, shifts,
            B, padded_H, padded_W, OH, OW, stride
        )

        if padding > 0:
            grad_x_nhwc = grad_x_padded_fp32[:, padding:-padding, padding:-padding, :]
        else:
            grad_x_nhwc = grad_x_padded_fp32
            
        grad_x_nchw = grad_x_nhwc.permute(0, 3, 1, 2).contiguous()

        return (
            grad_x_nchw.to(ctx.in_dtype), 
            grad_w_fp32.to(ctx.w_dtype), 
            None, None, None, None, None, None
        )

class LUTBffbConvFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x_bin, x_float, w_q, offsets, shifts, groups, K, stride, padding, tau):
        ctx.in_dtype = x_bin.dtype
        ctx.w_dtype = w_q.dtype
        ctx.params = (groups, K, stride, padding)
        
        B, in_C, H, W = x_bin.shape
        padded_H = H + 2 * padding
        padded_W = W + 2 * padding
        OH = (H + 2 * padding - K) // stride + 1
        OW = (W + 2 * padding - K) // stride + 1

        x_int8 = x_bin.to(torch.int8).contiguous() if x_bin.dtype != torch.int8 else x_bin.contiguous()
        packed_x = torch.ops.lut_lib.binary_pre_process(
            x_int8, padding, padding, padding, padding
        )

        w_packed = pack_weights_to_int64(w_q)

        y = torch.ops.lut_lib.bffb_conv(
            packed_x, w_packed, offsets, shifts,
            B, padded_H, padded_W, OH, OW, stride
        )

        ctx.save_for_backward(x_int8, x_float, w_packed, offsets, shifts, tau)

        return y

    @staticmethod
    def backward(ctx, grad_y):
        x_int8_nchw, x_float_nchw, w_packed, offsets, shifts, tau = ctx.saved_tensors
        groups, K, stride, padding = ctx.params
        
        grad_y_nhwc = grad_y.permute(0, 2, 3, 1).contiguous()
        
        B, in_C, H, W = x_int8_nchw.shape
        padded_H = H + 2 * padding
        padded_W = W + 2 * padding
        OH, OW = grad_y_nhwc.shape[1], grad_y_nhwc.shape[2]

        if padding > 0:
            x_int8_padded_nchw = F.pad(x_int8_nchw, (padding, padding, padding, padding))
        else:
            x_int8_padded_nchw = x_int8_nchw
        x_int8_padded_nhwc = x_int8_padded_nchw.permute(0, 2, 3, 1).contiguous()

        grad_x_padded_fp32, grad_w_fp32 = torch.ops.lut_lib.bffb_conv_bw(
            grad_y_nhwc, x_int8_padded_nhwc, x_float_nchw.contiguous(), w_packed, offsets, shifts,
            tau, padding, B, padded_H, padded_W, OH, OW, stride
        )

        if padding > 0:
            grad_x_nhwc = grad_x_padded_fp32[:, padding:-padding, padding:-padding, :]
        else:
            grad_x_nhwc = grad_x_padded_fp32
            
        grad_x_nchw = grad_x_nhwc.permute(0, 3, 1, 2).contiguous()

        return (
            grad_x_nchw.to(ctx.in_dtype), 
            None,
            grad_w_fp32.to(ctx.w_dtype), 
            None, None, None, None, None, None, None
        )

class LUTConvFP32Function(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, w, groups, K, stride, padding):
        ctx.in_dtype = x.dtype
        ctx.w_dtype = w.dtype
        
        B, in_C, H, W = x.shape
        out_C = w.shape[2]
        OH = (H + 2 * padding - K) // stride + 1
        OW = (W + 2 * padding - K) // stride + 1
        
        x_cl = x.contiguous(memory_format=torch.channels_last)
        w_cont = w.contiguous()
        
        ctx.save_for_backward(x_cl, w_cont)
        ctx.params = (groups, K, stride, padding)

        y = torch.ops.lut_lib.fused_lut_conv(
            x_cl, w_cont, groups, K, stride, padding, padding, OH, OW
        )
        return y

    @staticmethod
    def backward(ctx, grad_y):
        x_cl, w_cont = ctx.saved_tensors
        groups, K, stride, padding = ctx.params
        _, _, OH, OW = grad_y.shape
        
        grad_y_cl = grad_y.contiguous(memory_format=torch.channels_last)
        
        grads = torch.ops.lut_lib.fused_lut_conv_bw(
            grad_y_cl, x_cl, w_cont, groups, K, stride, padding, padding, OH, OW
        )
        
        grad_x = grads[0].to(ctx.in_dtype)
        grad_w = grads[1].to(ctx.w_dtype)
        
        return grad_x, grad_w, None, None, None, None


class LUTBafwConvFunction(torch.autograd.Function):
    @staticmethod
    @custom_fwd(cast_inputs=torch.float32)
    def forward(ctx, x_float, w_soft, offsets, shifts, padding, stride, tau):
        x_bin = torch.where(x_float >= 0, 
                            torch.tensor(1, dtype=torch.int8, device=x_float.device), 
                            torch.tensor(0, dtype=torch.int8, device=x_float.device))

        # 极速打包，包含 padding (1行代码完成)
        packed_x = torch.ops.lut_lib.fused_pre_process(x_bin, padding, padding, padding, padding)
        
        B, in_C, H, W = x_float.shape
        padded_H = H + 2 * padding
        padded_W = W + 2 * padding
        OH = (H + 2 * padding - 3) // stride + 1  # 假设 Kernel Size 为 3
        OW = (W + 2 * padding - 3) // stride + 1
        
        # O(1) 前向极速查表
        y = torch.ops.lut_lib.bafw_conv(
            packed_x, w_soft, offsets, shifts, 
            B, padded_H, padded_W, OH, OW, stride
        )
        
        # 缓存数据用于反向求导
        ctx.save_for_backward(packed_x, x_float, w_soft, offsets, shifts, tau)
        ctx.stride = stride
        ctx.padding = padding
        ctx.x_shape = (B, in_C, H, W)
        return y

    @staticmethod
    @custom_bwd
    def backward(ctx, grad_y):
        packed_x, x_float, w_soft, offsets, shifts, tau = ctx.saved_tensors
        stride = ctx.stride
        padding = ctx.padding
        B, in_C, H, W = ctx.x_shape
        
        padded_H = H + 2 * padding
        padded_W = W + 2 * padding
        OH, OW = grad_y.shape[2], grad_y.shape[3]
        
        grad_y_nhwc = grad_y.float().permute(0, 2, 3, 1).contiguous()

        grad_x, grad_w = torch.ops.lut_lib.bafw_conv_bw(
            grad_y_nhwc, packed_x, x_float, w_soft, offsets, shifts, tau, 
            padding, B, padded_H, padded_W, OH, OW, stride
        )
        
        # 仅 x_float 和 w_soft 需要梯度，其余返回 None
        return grad_x, grad_w, None, None, None, None, None

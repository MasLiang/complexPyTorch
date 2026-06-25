import torch
import math
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import custom_fwd, custom_bwd

# ============================================================================
# 仅导入我们最终编译出的 2 个终极动态 K 扩展库
# ============================================================================
import lut_cuda_binary
import lut_cuda_floating

lut_lib_def = torch.library.Library("lut_lib", "DEF")

lut_lib_def.define("pack_padded_inputs(Tensor padded_x) -> Tensor")
lut_lib_def.define("fused_pre_process(Tensor x_nchw, int pad_top, int pad_bottom, int pad_left, int pad_right) -> Tensor")
lut_lib_def.define("binary_conv(Tensor packed_x, Tensor w_packed, Tensor offsets, Tensor shifts, int B, int padded_H, int padded_W, int OH, int OW, int stride, int K) -> Tensor")
lut_lib_def.define("binary_conv_bw(Tensor grad_y, Tensor x_padded, Tensor w_packed, Tensor offsets, Tensor shifts, int B, int padded_H, int padded_W, int OH, int OW, int stride, int K) -> Tensor[]")

lut_lib_def.define("floating_conv(Tensor x, Tensor w, Tensor offsets, int groups, int K, int kernel_size, int stride, int OH, int OW) -> Tensor")
lut_lib_def.define("floating_conv_bw(Tensor grad_y, Tensor x, Tensor w, Tensor offsets, int groups, int K, int kernel_size, int stride, int OH, int OW) -> Tensor[]")

lut_lib_cuda = torch.library.Library("lut_lib", "IMPL", "CUDA")

lut_lib_cuda.impl("pack_padded_inputs", lut_cuda_binary.pack_padded_inputs)
lut_lib_cuda.impl("fused_pre_process", lut_cuda_binary.fused_pre_process)
lut_lib_cuda.impl("binary_conv", lut_cuda_binary.forward)
lut_lib_cuda.impl("binary_conv_bw", lut_cuda_binary.backward) 

lut_lib_cuda.impl("floating_conv", lut_cuda_floating.forward)
lut_lib_cuda.impl("floating_conv_bw", lut_cuda_floating.backward)

lut_lib_meta = torch.library.Library("lut_lib", "IMPL", "Meta")

def pack_padded_inputs_meta(padded_x):
    packed_C = (padded_x.size(3) + 31) // 32
    return torch.empty((padded_x.size(0), padded_x.size(1), padded_x.size(2), packed_C), dtype=torch.int32, device=padded_x.device)

def fused_pre_process_meta(x_nchw, pad_top, pad_bottom, pad_left, pad_right):
    B, in_C, H, W = x_nchw.shape
    padded_H, padded_W = H + pad_top + pad_bottom, W + pad_left + pad_right
    packed_C = (in_C + 31) // 32
    return torch.empty((B, padded_H, padded_W, packed_C), dtype=torch.int32, device=x_nchw.device, memory_format=torch.contiguous_format)

def binary_conv_meta(packed_x, w_packed, offsets, shifts, B, padded_H, padded_W, OH, OW, stride, K):
    out_C = w_packed.size(1)
    return torch.empty((B, out_C, OH, OW), dtype=torch.float32, device=packed_x.device, memory_format=torch.channels_last)

def binary_conv_bw_meta(grad_y, x_padded, w_packed, offsets, shifts, B, padded_H, padded_W, OH, OW, stride, K):
    gx = torch.empty_like(x_padded, dtype=torch.float32, memory_format=torch.contiguous_format)
    gw = torch.empty((w_packed.size(0), 1 << K, w_packed.size(1)), dtype=torch.float32, device=w_packed.device)
    return [gx, gw]

lut_lib_meta.impl("pack_padded_inputs", pack_padded_inputs_meta)
lut_lib_meta.impl("fused_pre_process", fused_pre_process_meta)
lut_lib_meta.impl("binary_conv", binary_conv_meta)
lut_lib_meta.impl("binary_conv_bw", binary_conv_bw_meta)


def floating_conv_meta(x, w, offsets, groups, K, kernel_size, stride, OH, OW):
    return torch.empty((x.size(0), w.size(2), OH, OW), dtype=x.dtype, device=x.device, memory_format=torch.channels_last)

def floating_conv_bw_meta(grad_y, x, w, offsets, groups, K, kernel_size, stride, OH, OW):
    return [torch.empty_like(x, memory_format=torch.channels_last), torch.empty_like(w)]

lut_lib_meta.impl("floating_conv", floating_conv_meta)
lut_lib_meta.impl("floating_conv_bw", floating_conv_bw_meta)


def pack_weights_to_int64(w_q, K):
    w_int64 = w_q.to(torch.int64)
    lut_size = 1 << K
    shifts = torch.arange(lut_size, dtype=torch.int64, device=w_q.device).view(1, lut_size, 1)
    w_packed = torch.sum(w_int64 << shifts, dim=1)
    return w_packed

class LUTBinaryConvFunction(torch.autograd.Function):
    @staticmethod
    @custom_fwd(cast_inputs=torch.float32) 
    def forward(ctx, x, w_q, offsets, shifts, groups, K, kernel_size, stride, padding):
        ctx.in_dtype = x.dtype
        ctx.w_dtype = w_q.dtype
        ctx.params = (groups, K, kernel_size, stride, padding)
        
        B, in_C, H, W = x.shape
        padded_H, padded_W = H + 2 * padding, W + 2 * padding
        OH = (H + 2 * padding - kernel_size) // stride + 1
        OW = (W + 2 * padding - kernel_size) // stride + 1

        x_int8 = torch.where(x >= 0, 
                             torch.tensor(1, dtype=torch.int8, device=x.device), 
                             torch.tensor(0, dtype=torch.int8, device=x.device))

        packed_x = torch.ops.lut_lib.fused_pre_process(x_int8, padding, padding, padding, padding)
        w_packed = pack_weights_to_int64(w_q, K)

        y = torch.ops.lut_lib.binary_conv(
            packed_x, w_packed, offsets, shifts,
            B, padded_H, padded_W, OH, OW, stride, K
        )

        ctx.save_for_backward(x_int8, w_packed, offsets, shifts)
        return y

    @staticmethod
    @custom_bwd
    def backward(ctx, grad_y):
        x_int8_nchw, w_packed, offsets, shifts = ctx.saved_tensors
        groups, K, kernel_size, stride, padding = ctx.params
        
        grad_y_nhwc = grad_y.float().permute(0, 2, 3, 1).contiguous()
        B, in_C, H, W = x_int8_nchw.shape
        padded_H, padded_W = H + 2 * padding, W + 2 * padding
        OH, OW = grad_y_nhwc.shape[1], grad_y_nhwc.shape[2]

        if padding > 0:
            x_int8_padded_nchw = F.pad(x_int8_nchw, (padding, padding, padding, padding))
        else:
            x_int8_padded_nchw = x_int8_nchw
        x_int8_padded_nhwc = x_int8_padded_nchw.permute(0, 2, 3, 1).contiguous()

        grad_x_padded_fp32, grad_w_fp32 = torch.ops.lut_lib.binary_conv_bw(
            grad_y_nhwc, x_int8_padded_nhwc, w_packed, offsets, shifts,
            B, padded_H, padded_W, OH, OW, stride, K
        )

        if padding > 0:
            grad_x_nhwc = grad_x_padded_fp32[:, padding:-padding, padding:-padding, :]
        else:
            grad_x_nhwc = grad_x_padded_fp32
            
        grad_x_nchw = grad_x_nhwc.permute(0, 3, 1, 2).contiguous()

        return (
            grad_x_nchw.to(ctx.in_dtype), grad_w_fp32.to(ctx.w_dtype), 
            None, None, None, None, None, None, None
        )

class LUTFloatingConvFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, w, offsets, groups, K, kernel_size, stride, padding):
        ctx.in_dtype = x.dtype
        ctx.w_dtype = w.dtype
        ctx.input_hw = (x.size(2), x.size(3))

        B, in_C, H, W = x.shape
        OH = (H + 2 * padding - kernel_size) // stride + 1
        OW = (W + 2 * padding - kernel_size) // stride + 1

        if padding > 0:
            x = F.pad(x, (padding, padding, padding, padding), "constant", 0.0)

        x_cl = x.contiguous(memory_format=torch.channels_last)
        w_cont = w.contiguous()
        offsets_cont = offsets.contiguous()

        ctx.save_for_backward(x_cl, w_cont, offsets_cont)
        ctx.params = (groups, K, kernel_size, stride, padding)

        y = torch.ops.lut_lib.floating_conv(
            x_cl, w_cont, offsets_cont, groups, K, kernel_size, stride, OH, OW
        )
        return y

    @staticmethod
    def backward(ctx, grad_y):
        x_cl, w_cont, offsets_cont = ctx.saved_tensors
        groups, K, kernel_size, stride, padding = ctx.params
        _, _, OH, OW = grad_y.shape

        grad_y_cl = grad_y.contiguous(memory_format=torch.channels_last)

        grads = torch.ops.lut_lib.floating_conv_bw(
            grad_y_cl, x_cl, w_cont, offsets_cont, groups, K, kernel_size, stride, OH, OW
        )

        grad_x = grads[0]
        if padding > 0:
            h, w = ctx.input_hw
            grad_x = grad_x[:, :, padding:padding + h, padding:padding + w]

        return grad_x.to(ctx.in_dtype), grads[1].to(ctx.w_dtype), None, None, None, None, None, None

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 19 10:30:02 2019

@author: Sebastien M. Popoff


Based on https://openreview.net/forum?id=H1T2hmZAb
"""
from typing import Optional
import math

import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.nn import (
    Module, Parameter, init,
    Conv2d, ConvTranspose2d, Linear, LSTM, GRU,
    BatchNorm1d, BatchNorm2d,
    PReLU
)

from .lut_backend import (
    LUT6Function,
    LUTBafwConvFunction,
    LUTBffbConvFunction,
    LUTConvFP32Function,
)
from .complexFunctions import (
    complex_relu,
    complex_tanh,
    complex_sigmoid,
    complex_max_pool2d,
    complex_avg_pool2d,
    complex_dropout,
    complex_dropout2d,
    complex_opposite,
    complex_binary_activation,
    complex_binary_weight,
)

def bimodal_initialization(tensor, init_cfg=[-1, 0.2, 1, 0.1]):
    mode1_mean, mode1_std = init_cfg[0], init_cfg[1]
    mode2_mean, mode2_std = init_cfg[2], init_cfg[3]
    if mode1_mean!=0 and mode2_mean!=0:
        mask = torch.bernoulli(torch.full_like(tensor, 0.5))

        mean = mask * mode1_mean + (1 - mask) * mode2_mean
        std  = mask * mode1_std  + (1 - mask) * mode2_std

        output = torch.normal(mean, std)

        with torch.no_grad():
            tensor.copy_(output)

def binary_gumbel_softmax(logits, tau=1.0, hard=1, w0y1=0):
    #if w0y1==0:
    #    if hard:
    #        y = logits*tau
    #    else:
    #        gumbel_noise = -torch.log(-torch.log(torch.rand_like(logits)))/tau
    #        y = (logits + gumbel_noise) * tau
    #else:
    if hard==1:
        binary_hard = torch.where(logits >= 0.0, 1.0, 0.0)
        binary_sample = (binary_hard - logits).detach() + logits
    else:
        y = logits * tau
        y = F.tanh(y)
        binary_sample = (y+1)*0.5
    return binary_sample

def apply_complex(fr, fi, input, dtype=torch.complex64):
    return (fr(input.real)-fi(input.imag)).type(dtype) \
        + 1j*(fr(input.imag)+fi(input.real)).type(dtype)


class ComplexDropout(Module):
    def __init__(self, p=0.5):
        super().__init__()
        self.p = p

    def forward(self, input):
        if self.training:
            return complex_dropout(input, self.p)
        else:
            return input


class ComplexDropout2d(Module):
    def __init__(self, p=0.5):
        super(ComplexDropout2d, self).__init__()
        self.p = p

    def forward(self, inp):
        if self.training:
            return complex_dropout2d(inp, self.p)
        else:
            return inp


class ComplexMaxPool2d(Module):
    def __init__(
        self,
        kernel_size,
        stride=None,
        padding=0,
        dilation=1,
        return_indices=False,
        ceil_mode=False,
    ):
        super(ComplexMaxPool2d, self).__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.ceil_mode = ceil_mode
        self.return_indices = return_indices

    def forward(self, inp):
        return complex_max_pool2d(
            inp,
            kernel_size=self.kernel_size,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
            ceil_mode=self.ceil_mode,
            return_indices=self.return_indices,
        )


class ComplexAvgPool2d(torch.nn.Module):

    def __init__(self, kernel_size, stride=None, padding=0,
                 ceil_mode=False, count_include_pad=True, divisor_override=None):
        super(ComplexAvgPool2d, self).__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.ceil_mode = ceil_mode
        self.count_include_pad = count_include_pad
        self.divisor_override = divisor_override

    def forward(self, inp):
        return complex_avg_pool2d(inp, kernel_size=self.kernel_size,
                                  stride=self.stride, padding=self.padding,
                                  ceil_mode=self.ceil_mode, count_include_pad=self.count_include_pad,
                                  divisor_override=self.divisor_override)


class ComplexReLU(Module):
    @staticmethod
    def forward(inp):
        return complex_relu(inp)


class ComplexSigmoid(Module):
    @staticmethod
    def forward(inp):
        return complex_sigmoid(inp)


class ComplexPReLU(Module):
    def __init__(self):
        super().__init__()
        self.r_prelu = PReLU()
        self.i_prelu = PReLU()

    @staticmethod
    def forward(self, inp):
        return self.r_prelu(inp.real) + 1j*self.i_prelu(inp.imag)


class ComplexTanh(Module):
    @staticmethod
    def forward(inp):
        return complex_tanh(inp)


class ComplexConvTranspose2d(Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        padding=0,
        output_padding=0,
        groups=1,
        bias=True,
        dilation=1,
        padding_mode="zeros",
    ):

        super().__init__()

        self.conv_tran_r = ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding,
                                           output_padding, groups, bias, dilation, padding_mode)
        self.conv_tran_i = ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding,
                                           output_padding, groups, bias, dilation, padding_mode)

    def forward(self, inp):
        return apply_complex(self.conv_tran_r, self.conv_tran_i, inp)


class ComplexConv2d(Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=3,
        stride=1,
        padding=0,
        dilation=1,
        groups=1,
        bias=True,
    ):
        super(ComplexConv2d, self).__init__()
        self.conv_r = Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            dilation,
            groups,
            bias,
        )
        self.conv_i = Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            dilation,
            groups,
            bias,
        )

    def forward(self, inp):
        return apply_complex(self.conv_r, self.conv_i, inp)


class ComplexLinear(Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.fc_r = Linear(in_features, out_features)
        self.fc_i = Linear(in_features, out_features)

    def forward(self, inp):
        return apply_complex(self.fc_r, self.fc_i, inp)


class BinaryComplexActivation(Module):
    def __init__(self, grad_mode="bireal"):
        super().__init__()
        self.grad_mode = grad_mode

    def forward(self, inp):
        return complex_binary_activation(inp, grad_mode=self.grad_mode)


class BinaryComplexBitActivation(BinaryComplexActivation):
    """Apply Bi-Real activation, then encode {-1, +1} as {0, 1}."""

    @staticmethod
    def _encode_component(signed, value):
        hard_signed = torch.where(
            value > 0.0,
            torch.ones_like(value),
            -torch.ones_like(value),
        )
        signed_pm_one = signed + (hard_signed - signed).detach()
        return (signed_pm_one + 1.0) * 0.5

    def forward(self, inp):
        signed = super().forward(inp)
        if not torch.is_complex(inp):
            return self._encode_component(signed, inp)
        real = self._encode_component(signed.real, inp.real)
        imag = self._encode_component(signed.imag, inp.imag)
        return torch.complex(real, imag)


class BinaryComplexConv2d(Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=3,
        stride=1,
        padding=0,
        dilation=1,
        groups=1,
        bias=True,
        per_channel=True,
        weight_grad_mode="ste",
        weight_proxy_mode="scaled_ste",
    ):
        super().__init__()
        if weight_proxy_mode not in ("scaled_ste", "bireal"):
            raise ValueError(
                "Unknown binary-weight proxy mode: {}".format(
                    weight_proxy_mode
                )
            )
        self.conv_r = Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            dilation,
            groups,
            bias,
        )
        self.conv_i = Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            dilation,
            groups,
            bias,
        )
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.per_channel = per_channel
        self.weight_grad_mode = weight_grad_mode
        self.weight_proxy_mode = weight_proxy_mode

    def forward(self, inp):
        weight = torch.complex(self.conv_r.weight, self.conv_i.weight)
        weight = complex_binary_weight(
            weight,
            per_channel=self.per_channel,
            grad_mode=self.weight_grad_mode,
            proxy_mode=self.weight_proxy_mode,
        )
        w_r = weight.real
        w_i = weight.imag

        real = F.conv2d(
            inp.real,
            w_r,
            bias=self.conv_r.bias,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
            groups=self.groups,
        )
        real = real - F.conv2d(
            inp.imag,
            w_i,
            bias=self.conv_i.bias,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
            groups=self.groups,
        )

        imag = F.conv2d(
            inp.imag,
            w_r,
            bias=self.conv_r.bias,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
            groups=self.groups,
        )
        imag = imag + F.conv2d(
            inp.real,
            w_i,
            bias=self.conv_i.bias,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
            groups=self.groups,
        )

        return torch.complex(real, imag)


class BinaryComplexLinear(Module):
    def __init__(self, in_features, out_features, bias=True, per_channel=True, weight_grad_mode="ste"):
        super().__init__()
        self.fc_r = Linear(in_features, out_features, bias=bias)
        self.fc_i = Linear(in_features, out_features, bias=bias)
        self.per_channel = per_channel
        self.weight_grad_mode = weight_grad_mode

    def forward(self, inp):
        weight = torch.complex(self.fc_r.weight, self.fc_i.weight)
        weight = complex_binary_weight(
            weight,
            per_channel=self.per_channel,
            grad_mode=self.weight_grad_mode,
        )
        w_r = weight.real
        w_i = weight.imag

        real = F.linear(inp.real, w_r, self.fc_r.bias)
        real = real - F.linear(inp.imag, w_i, self.fc_i.bias)
        imag = F.linear(inp.imag, w_r, self.fc_r.bias)
        imag = imag + F.linear(inp.real, w_i, self.fc_i.bias)

        return torch.complex(real, imag)


class NaiveComplexBatchNorm1d(Module):
    """
    Naive approach to complex batch norm, perform batch norm independently on real and imaginary part.
    """

    def __init__(
        self,
        num_features,
        eps=1e-5,
        momentum=0.1,
        affine=True,
        track_running_stats=True,
    ):
        super(NaiveComplexBatchNorm1d, self).__init__()
        self.bn_r = BatchNorm1d(
            num_features, eps, momentum, affine, track_running_stats
        )
        self.bn_i = BatchNorm1d(
            num_features, eps, momentum, affine, track_running_stats
        )

    def forward(self, inp):
        return self.bn_r(inp.real).type(torch.complex64) + 1j * self.bn_i(
            inp.imag
        ).type(torch.complex64)


class NaiveComplexBatchNorm2d(Module):
    """
    Naive approach to complex batch norm, perform batch norm independently on real and imaginary part.
    """

    def __init__(
        self,
        num_features,
        eps=1e-5,
        momentum=0.1,
        affine=True,
        track_running_stats=True,
    ):
        super(NaiveComplexBatchNorm2d, self).__init__()
        self.bn_r = BatchNorm2d(
            num_features, eps, momentum, affine, track_running_stats
        )
        self.bn_i = BatchNorm2d(
            num_features, eps, momentum, affine, track_running_stats
        )

    def forward(self, inp):
        return self.bn_r(inp.real).type(torch.complex64) + 1j * self.bn_i(
            inp.imag
        ).type(torch.complex64)


class _ComplexBatchNorm(Module):
    running_mean: Optional[torch.Tensor]

    def __init__(
        self,
        num_features,
        eps=1e-5,
        momentum=0.1,
        affine=True,
        track_running_stats=True,
    ):
        super(_ComplexBatchNorm, self).__init__()
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        self.affine = affine
        self.track_running_stats = track_running_stats
        if self.affine:
            self.weight = Parameter(torch.Tensor(num_features, 3))
            self.bias = Parameter(torch.Tensor(num_features, 2))
        else:
            self.register_parameter("weight", None)
            self.register_parameter("bias", None)
        if self.track_running_stats:
            self.register_buffer(
                "running_mean", torch.zeros(
                    num_features, dtype=torch.complex64)
            )
            self.register_buffer("running_covar", torch.zeros(num_features, 3))
            self.running_covar[:, 0] = 1.4142135623730951
            self.running_covar[:, 1] = 1.4142135623730951
            self.register_buffer(
                "num_batches_tracked", torch.tensor(0, dtype=torch.long)
            )
        else:
            self.register_parameter("running_mean", None)
            self.register_parameter("running_covar", None)
            self.register_parameter("num_batches_tracked", None)
        self.reset_parameters()

    def reset_running_stats(self):
        if self.track_running_stats:
            self.running_mean.zero_()
            self.running_covar.zero_()
            self.running_covar[:, 0] = 1.4142135623730951
            self.running_covar[:, 1] = 1.4142135623730951
            self.num_batches_tracked.zero_()

    def reset_parameters(self):
        self.reset_running_stats()
        if self.affine:
            init.constant_(self.weight[:, :2], 1.4142135623730951)
            init.zeros_(self.weight[:, 2])
            init.zeros_(self.bias)


class ComplexBatchNorm2d(_ComplexBatchNorm):
    def forward(self, inp):
        exponential_average_factor = 0.0

        if self.training and self.track_running_stats:
            if self.num_batches_tracked is not None:
                self.num_batches_tracked += 1
                if self.momentum is None:  # use cumulative moving average
                    exponential_average_factor = 1.0 / \
                        float(self.num_batches_tracked)
                else:  # use exponential moving average
                    exponential_average_factor = self.momentum

        if self.training or (not self.track_running_stats):
            # calculate mean of real and imaginary part
            # mean does not support automatic differentiation for outputs with complex dtype.
            mean_r = inp.real.mean([0, 2, 3]).type(torch.complex64)
            mean_i = inp.imag.mean([0, 2, 3]).type(torch.complex64)
            mean = mean_r + 1j * mean_i
        else:
            mean = self.running_mean

        if self.training and self.track_running_stats:
            # update running mean
            with torch.no_grad():
                self.running_mean = (
                    exponential_average_factor * mean
                    + (1 - exponential_average_factor) * self.running_mean
                )

        inp = inp - mean[None, :, None, None]

        if self.training or (not self.track_running_stats):
            # Elements of the covariance matrix (biased for train)
            n = inp.numel() / inp.size(1)
            Crr = 1.0 / n * inp.real.pow(2).sum(dim=[0, 2, 3]) + self.eps
            Cii = 1.0 / n * inp.imag.pow(2).sum(dim=[0, 2, 3]) + self.eps
            Cri = (inp.real.mul(inp.imag)).mean(dim=[0, 2, 3])
        else:
            Crr = self.running_covar[:, 0] + self.eps
            Cii = self.running_covar[:, 1] + self.eps
            Cri = self.running_covar[:, 2]  # +self.eps

        if self.training and self.track_running_stats:
            with torch.no_grad():
                self.running_covar[:, 0] = (
                    exponential_average_factor * Crr * n / (n - 1)  #
                    + (1 - exponential_average_factor) * \
                    self.running_covar[:, 0]
                )

                self.running_covar[:, 1] = (
                    exponential_average_factor * Cii * n / (n - 1)
                    + (1 - exponential_average_factor) *
                    self.running_covar[:, 1]
                )

                self.running_covar[:, 2] = (
                    exponential_average_factor * Cri * n / (n - 1)
                    + (1 - exponential_average_factor) *
                    self.running_covar[:, 2]
                )

        # calculate the inverse square root the covariance matrix
        det = (Crr * Cii - Cri.pow(2)).clamp(min=1e-7)
        s = torch.sqrt(det)
        t = torch.sqrt(Cii + Crr + 2 * s).clamp(min=1e-7)
        inverse_st = 1.0 / (s * t).clamp(min=1e-7)
        Rrr = (Cii + s) * inverse_st
        Rii = (Crr + s) * inverse_st
        Rri = -Cri * inverse_st

        inp = (
            Rrr[None, :, None, None] * inp.real +
            Rri[None, :, None, None] * inp.imag
        ).type(torch.complex64) + 1j * (
            Rii[None, :, None, None] * inp.imag +
            Rri[None, :, None, None] * inp.real
        ).type(
            torch.complex64
        )

        if self.affine:
            inp = (
                self.weight[None, :, 0, None, None] * inp.real
                + self.weight[None, :, 2, None, None] * inp.imag
                + self.bias[None, :, 0, None, None]
            ).type(torch.complex64) + 1j * (
                self.weight[None, :, 2, None, None] * inp.real
                + self.weight[None, :, 1, None, None] * inp.imag
                + self.bias[None, :, 1, None, None]
            ).type(
                torch.complex64
            )
        return inp


class ComplexBatchNorm1d(_ComplexBatchNorm):
    def forward(self, inp):
        exponential_average_factor = 0.0

        if self.training and self.track_running_stats:
            if self.num_batches_tracked is not None:
                self.num_batches_tracked += 1
                if self.momentum is None:  # use cumulative moving average
                    exponential_average_factor = 1.0 / float(self.num_batches_tracked)
                else:  # use exponential moving average
                    exponential_average_factor = self.momentum

        reduce_dims = tuple([0] + list(range(2, inp.dim())))
        view_shape = [1, inp.size(1)] + [1] * (inp.dim() - 2)

        if self.training or (not self.track_running_stats):
            # calculate mean of real and imaginary part
            mean_r = inp.real.mean(dim=reduce_dims).type(torch.complex64)
            mean_i = inp.imag.mean(dim=reduce_dims).type(torch.complex64)
            mean = mean_r + 1j * mean_i
        else:
            mean = self.running_mean

        if self.training and self.track_running_stats:
            # update running mean
            with torch.no_grad():
                self.running_mean = (
                    exponential_average_factor * mean
                    + (1 - exponential_average_factor) * self.running_mean
                )

        inp = inp - mean.view(*view_shape)

        if self.training or (not self.track_running_stats):
            # Elements of the covariance matrix (biased for train)
            n = inp.numel() / inp.size(1)
            Crr = inp.real.pow(2).mean(dim=reduce_dims) + self.eps
            Cii = inp.imag.pow(2).mean(dim=reduce_dims) + self.eps
            Cri = (inp.real.mul(inp.imag)).mean(dim=reduce_dims)
        else:
            Crr = self.running_covar[:, 0] + self.eps
            Cii = self.running_covar[:, 1] + self.eps
            Cri = self.running_covar[:, 2]

        if self.training and self.track_running_stats:
            with torch.no_grad():
                self.running_covar[:, 0] = (
                    exponential_average_factor * Crr * n / (n - 1)
                    + (1 - exponential_average_factor) * self.running_covar[:, 0]
                )
                self.running_covar[:, 1] = (
                    exponential_average_factor * Cii * n / (n - 1)
                    + (1 - exponential_average_factor) * self.running_covar[:, 1]
                )
                self.running_covar[:, 2] = (
                    exponential_average_factor * Cri * n / (n - 1)
                    + (1 - exponential_average_factor) * self.running_covar[:, 2]
                )

        # calculate the inverse square root the covariance matrix
        det = (Crr * Cii - Cri.pow(2)).clamp(min=1e-7)
        s = torch.sqrt(det)
        t = torch.sqrt(Cii + Crr + 2 * s).clamp(min=1e-7)
        inverse_st = 1.0 / (s * t).clamp(min=1e-7)
        Rrr = (Cii + s) * inverse_st
        Rii = (Crr + s) * inverse_st
        Rri = -Cri * inverse_st

        # 将所有的协方差参数拓展为 view_shape 以匹配输入张量的广播
        Rrr_v = Rrr.view(*view_shape)
        Rii_v = Rii.view(*view_shape)
        Rri_v = Rri.view(*view_shape)

        inp = (Rrr_v * inp.real + Rri_v * inp.imag).type(torch.complex64) + 1j * (
            Rii_v * inp.imag + Rri_v * inp.real
        ).type(torch.complex64)

        if self.affine:
            # 可学习仿射变换参数也必须安全广播
            w0 = self.weight[:, 0].view(*view_shape)
            w1 = self.weight[:, 1].view(*view_shape)
            w2 = self.weight[:, 2].view(*view_shape)
            b0 = self.bias[:, 0].view(*view_shape)
            b1 = self.bias[:, 1].view(*view_shape)

            inp = (
                w0 * inp.real + w2 * inp.imag + b0
            ).type(torch.complex64) + 1j * (
                w2 * inp.real + w1 * inp.imag + b1
            ).type(torch.complex64)

        return inp

class ComplexGRUCell(Module):
    """
    A GRU cell for complex-valued inputs
    """

    def __init__(self, input_length, hidden_length):
        super().__init__()
        self.input_length = input_length
        self.hidden_length = hidden_length

        # reset gate components
        self.linear_reset_w1 = ComplexLinear(
            self.input_length, self.hidden_length)
        self.linear_reset_r1 = ComplexLinear(
            self.hidden_length, self.hidden_length)

        self.linear_reset_w2 = ComplexLinear(
            self.input_length, self.hidden_length)
        self.linear_reset_r2 = ComplexLinear(
            self.hidden_length, self.hidden_length)

        # update gate components
        self.linear_gate_w3 = ComplexLinear(
            self.input_length, self.hidden_length)
        self.linear_gate_r3 = ComplexLinear(
            self.hidden_length, self.hidden_length)

        self.activation_gate = ComplexSigmoid()
        self.activation_candidate = ComplexTanh()

    def reset_gate(self, x, h):
        x_1 = self.linear_reset_w1(x)
        h_1 = self.linear_reset_r1(h)
        # gate update
        reset = self.activation_gate(x_1 + h_1)
        return reset

    def update_gate(self, x, h):
        x_2 = self.linear_reset_w2(x)
        h_2 = self.linear_reset_r2(h)
        z = self.activation_gate(h_2 + x_2)
        return z

    def update_component(self, x, h, r):
        x_3 = self.linear_gate_w3(x)
        h_3 = r * self.linear_gate_r3(h)  # element-wise multiplication
        gate_update = self.activation_candidate(x_3 + h_3)
        return gate_update

    def forward(self, x, h):
        # Equation 1. reset gate vector
        r = self.reset_gate(x, h)

        # Equation 2: the update gate - the shared update gate vector z
        z = self.update_gate(x, h)

        # Equation 3: The almost output component
        n = self.update_component(x, h, r)

        # Equation 4: the new hidden state
        h_new = (1 + complex_opposite(z)) * n + \
            z * h  # element-wise multiplication
        return h_new


class ComplexBNGRUCell(Module):
    """
    A BN-GRU cell for complex-valued inputs
    """

    def __init__(self, input_length=10, hidden_length=20):
        super().__init__()
        self.input_length = input_length
        self.hidden_length = hidden_length

        # reset gate components
        self.linear_reset_w1 = ComplexLinear(
            self.input_length, self.hidden_length)
        self.linear_reset_r1 = ComplexLinear(
            self.hidden_length, self.hidden_length)

        self.linear_reset_w2 = ComplexLinear(
            self.input_length, self.hidden_length)
        self.linear_reset_r2 = ComplexLinear(
            self.hidden_length, self.hidden_length)

        # update gate components
        self.linear_gate_w3 = ComplexLinear(
            self.input_length, self.hidden_length)
        self.linear_gate_r3 = ComplexLinear(
            self.hidden_length, self.hidden_length)

        self.activation_gate = ComplexSigmoid()
        self.activation_candidate = ComplexTanh()

        self.bn = ComplexBatchNorm2d(1)

    def reset_gate(self, x, h):
        x_1 = self.linear_reset_w1(x)
        h_1 = self.linear_reset_r1(h)
        # gate update
        reset = self.activation_gate(self.bn(x_1) + self.bn(h_1))
        return reset

    def update_gate(self, x, h):
        x_2 = self.linear_reset_w2(x)
        h_2 = self.linear_reset_r2(h)
        z = self.activation_gate(self.bn(h_2) + self.bn(x_2))
        return z

    def update_component(self, x, h, r):
        x_3 = self.linear_gate_w3(x)
        # element-wise multiplication
        h_3 = r * self.bn(self.linear_gate_r3(h))
        gate_update = self.activation_candidate(self.bn(self.bn(x_3) + h_3))
        return gate_update

    def forward(self, x, h):
        # Equation 1. reset gate vector
        r = self.reset_gate(x, h)

        # Equation 2: the update gate - the shared update gate vector z
        z = self.update_gate(x, h)

        # Equation 3: The almost output component
        n = self.update_component(x, h, r)

        # Equation 4: the new hidden state


class ComplexGRU(Module):
    def __init__(self, input_size, hidden_size, num_layers=1, bias=True,
                 batch_first=False, dropout=0, bidirectional=False):
        super().__init__()

        self.gru_re = GRU(input_size=input_size, hidden_size=hidden_size,
                          num_layers=num_layers, bias=bias,
                          batch_first=batch_first, dropout=dropout,
                          bidirectional=bidirectional)
        self.gru_im = GRU(input_size=input_size, hidden_size=hidden_size,
                          num_layers=num_layers, bias=bias,
                          batch_first=batch_first, dropout=dropout,
                          bidirectional=bidirectional)

    def forward(self, x):
        real, state_real = self._forward_real(x)
        imaginary, state_imag = self._forward_imaginary(x)

        output = torch.complex(real, imaginary)
        state = torch.complex(state_real, state_imag)

        return output, state

    def forward(self, x):
        r2r_out = self.gru_re(x.real)[0]
        r2i_out = self.gru_im(x.real)[0]
        i2r_out = self.gru_re(x.imag)[0]
        i2i_out = self.gru_im(x.imag)[0]
        real_out = r2r_out - i2i_out
        imag_out = i2r_out + r2i_out

        return torch.complex(real_out, imag_out), None

    def _forward_real(self, x):
        real_real, h_real = self.gru_re(x.real)
        imag_imag, h_imag = self.gru_im(x.imag)
        real = real_real - imag_imag

        return real, torch.complex(h_real, h_imag)

    def _forward_imaginary(self, x):
        imag_real, h_real = self.gru_re(x.imag)
        real_imag, h_imag = self.gru_im(x.real)
        imaginary = imag_real + real_imag

        return imaginary, torch.complex(h_real, h_imag)


class ComplexLSTM(Module):
    def __init__(self, input_size, hidden_size, num_layers=1, bias=True,
                 batch_first=False, dropout=0, bidirectional=False):
        super().__init__()
        self.num_layer = num_layers
        self.hidden_size = hidden_size
        self.batch_dim = 0 if batch_first else 1
        self.bidirectional = bidirectional

        self.lstm_re = LSTM(input_size=input_size, hidden_size=hidden_size,
                            num_layers=num_layers, bias=bias,
                            batch_first=batch_first, dropout=dropout,
                            bidirectional=bidirectional)
        self.lstm_im = LSTM(input_size=input_size, hidden_size=hidden_size,
                            num_layers=num_layers, bias=bias,
                            batch_first=batch_first, dropout=dropout,
                            bidirectional=bidirectional)

    def forward(self, x):
        real, state_real = self._forward_real(x)
        imaginary, state_imag = self._forward_imaginary(x)

        output = torch.complex(real, imaginary)

        return output, (state_real, state_imag)

    def _forward_real(self, x):
        h_real, h_imag, c_real, c_imag = self._init_state(
            self._get_batch_size(x), x.is_cuda)
        real_real, (h_real, c_real) = self.lstm_re(x.real, (h_real, c_real))
        imag_imag, (h_imag, c_imag) = self.lstm_im(x.imag, (h_imag, c_imag))
        real = real_real - imag_imag
        return real, ((h_real, c_real), (h_imag, c_imag))

    def _forward_imaginary(self, x):
        h_real, h_imag, c_real, c_imag = self._init_state(
            self._get_batch_size(x), x.is_cuda)
        imag_real, (h_real, c_real) = self.lstm_re(x.imag, (h_real, c_real))
        real_imag, (h_imag, c_imag) = self.lstm_im(x.real, (h_imag, c_imag))
        imaginary = imag_real + real_imag

        return imaginary, ((h_real, c_real), (h_imag, c_imag))

    def _init_state(self, batch_size, to_gpu=False):
        dim_0 = 2 if self.bidirectional else 1
        dims = (dim_0, batch_size, self.hidden_size)

        h_real, h_imag, c_real, c_imag = [
            torch.zeros(dims) for i in range(4)]

        if to_gpu:
            h_real, h_imag, c_real, c_imag = [
                t.cuda() for t in [h_real, h_imag, c_real, c_imag]]

        return h_real, h_imag, c_real, c_imag

    def _get_batch_size(self, x):
        return x.size(self.batch_dim)
        h_new = (1 + complex_opposite(z)) * n + \
            z * h  # element-wise multiplication

        return h_new


def _initialize_bimodal_(
    tensor,
    negative_mean=-1.0,
    negative_std=0.1,
    positive_mean=1.0,
    positive_std=0.1,
):
    """Initialize LUT logits with the real Bi-Real LUT recipe."""
    with torch.no_grad():
        positive = torch.rand_like(tensor) >= 0.5
        means = torch.where(
            positive,
            torch.full_like(tensor, positive_mean),
            torch.full_like(tensor, negative_mean),
        )
        stds = torch.where(
            positive,
            torch.full_like(tensor, positive_std),
            torch.full_like(tensor, negative_std),
        )
        tensor.copy_(torch.normal(means, stds))


def hard_binary_table(logits):
    """Use a hard Boolean table in forward and identity STE in backward."""
    hard = (logits >= 0.0).to(logits.dtype)
    return logits + (hard - logits).detach()


def annealed_binary_table(logits, temperature):
    """Map LUT logits to continuous 0/1 entries for temperature annealing."""
    if temperature <= 0.0:
        raise ValueError("LUT temperature must be positive")
    return 0.5 * (torch.tanh(logits * float(temperature)) + 1.0)


class LUTBffbConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, group_num=1, init_cfg=[-1, 0.1, 1, 0.1]):
        super(LUTBffbConv2d, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.groups = group_num
        self.K = kernel_size
        self.stride = stride
        self.padding = padding

        self.lut_num = (in_channels // group_num) * kernel_size * kernel_size // 6

        self.weight = nn.Parameter(torch.randn(self.lut_num, 64, out_channels))
        bimodal_initialization(self.weight, init_cfg=init_cfg)

        self.register_buffer('offsets', None)
        self.register_buffer('shifts', None)
        self.register_buffer('tau', torch.tensor([1.0], dtype=torch.float32))
        self.offsets_padded_W = -1

    def _precompute_offsets(self, device, padded_W):
        packed_C = (self.in_channels + 31) // 32
        ic_per_g = self.in_channels // self.groups
        K_sq = self.K * self.K

        l = torch.arange(self.lut_num, device=device).unsqueeze(1)
        i = torch.arange(6, device=device).unsqueeze(0)

        group_id = (l * self.groups) // self.lut_num
        ic_start = group_id * ic_per_g

        initial_idx = (l * 6 + i) % (ic_per_g * K_sq)
        ic = ic_start + (initial_idx // K_sq)
        sp = initial_idx % K_sq

        dy = sp // self.K
        dx = sp % self.K
        c_word = ic // 32
        c_shift = ic % 32

        abs_offset = (dy * padded_W + dx) * packed_C + c_word

        self.offsets = abs_offset.flatten().to(torch.int32)
        self.shifts = c_shift.flatten().to(torch.int32)

    def forward(self, x_bin, x_float):
        padded_W = x_bin.shape[3] + 2 * self.padding
        if self.offsets is None or self.offsets_padded_W != padded_W:
            self._precompute_offsets(x_bin.device, padded_W)
            self.offsets_padded_W = padded_W

        w_q = binary_gumbel_softmax(self.weight, tau=1, hard=True)

        return LUTBffbConvFunction.apply(
            x_bin, x_float, w_q,
            self.offsets, self.shifts, self.groups,
            self.K, self.stride, self.padding, self.tau
        )

class LUTBafwConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, group_num=1, init_cfg=[-1, 0.2, 1, 0.1]):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.groups = group_num

        self.lut_num = math.ceil((in_channels // group_num) * kernel_size * kernel_size / 6)

        self.register_buffer('tau', torch.tensor([1.0], dtype=torch.float32))
        self.register_buffer('hard', torch.tensor([0], dtype=torch.float32))

        self.lut_weight = nn.Parameter(torch.empty(self.lut_num, 64, out_channels))
        bimodal_initialization(self.lut_weight, init_cfg=init_cfg)

        self.register_buffer('offsets', None)
        self.register_buffer('shifts', None)
        self.offsets_padded_W = -1

    def _precompute_offsets(self, device, padded_W):
        packed_C = (self.in_channels + 31) // 32
        ic_per_g = self.in_channels // self.groups
        K_sq = self.kernel_size * self.kernel_size

        l = torch.arange(self.lut_num, device=device).unsqueeze(1)
        i = torch.arange(6, device=device).unsqueeze(0)

        group_id = (l * self.groups) // self.lut_num
        ic_start = group_id * ic_per_g

        initial_idx = (l * 6 + i) % (ic_per_g * K_sq)
        ic = ic_start + (initial_idx // K_sq)
        sp = initial_idx % K_sq

        dy = sp // self.kernel_size
        dx = sp % self.kernel_size
        c_word = ic // 32
        c_shift = ic % 32

        abs_offset = (dy * padded_W + dx) * packed_C + c_word

        self.offsets = abs_offset.flatten().to(torch.int32)
        self.shifts = c_shift.flatten().to(torch.int32)

    def forward(self, x):
        padded_W = x.shape[3] + 2 * self.padding
        if self.offsets is None or self.offsets_padded_W != padded_W:
            self._precompute_offsets(x.device, padded_W)
            self.offsets_padded_W = padded_W

        w_soft = binary_gumbel_softmax(self.lut_weight, tau=self.tau, hard=self.hard)

        return LUTBafwConvFunction.apply(
            x, w_soft,
            self.offsets, self.shifts,
            self.padding, self.stride, self.tau
        )

class LUTFPConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, lut_k=6, kernel_size=3, stride=1, padding=1, group_num=1, init_cfg=[-1, 0.2, 1, 0.1]):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.groups = group_num

        oc_per_g = out_channels // group_num
        if oc_per_g % 32 != 0 and group_num > 1:
            raise ValueError(f"由于 TILE_OC=32 约束，每组通道数({oc_per_g})必须是32的倍数")

        self.lut_num = math.ceil((in_channels // group_num) * kernel_size * kernel_size / 6)

        self.register_buffer('tau', torch.tensor([1.0], dtype=torch.float32))
        self.register_buffer('hard', torch.tensor([0], dtype=torch.float32))
        self.weight = nn.Parameter(torch.empty(self.lut_num, 64, out_channels))
        bimodal_initialization(self.weight, init_cfg=init_cfg)

    def reset_parameters(self):
        nn.init.kaiming_normal_(self.weight, mode='fan_out', nonlinearity='relu')

    def forward(self, x):
        if self.out_channels % 8 != 0:
            raise ValueError("out_channels 必须是 8 的倍数以支持向量化访存")

        w_q = binary_gumbel_softmax(self.weight, tau=self.tau, hard=self.hard)
        return LUTConvFP32Function.apply(
            x, w_q, self.groups, self.kernel_size, self.stride, self.padding
        )

    def extra_repr(self):
        return (f"in_channels={self.in_channels}, out_channels={self.out_channels}, "
                f"kernel_size={self.kernel_size}, groups={self.groups}, lut_num={self.lut_num}")

class LUTBinaryConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, group_num=1, init_cfg=[-1, 0.1, 1, 0.1]):
        super(LUTBinaryConv2d, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.groups = group_num
        self.K = kernel_size
        self.stride = stride
        self.padding = padding

        if in_channels % group_num:
            raise ValueError("in_channels must be divisible by group_num")
        if out_channels % group_num:
            raise ValueError("out_channels must be divisible by group_num")

        self.logical_positions = (
            (in_channels // group_num) * kernel_size * kernel_size
        )
        self.lut_num = math.ceil(self.logical_positions / 6)

        self.weight = nn.Parameter(
            torch.randn(out_channels, self.lut_num, 64)
        )
        bimodal_initialization(self.weight, init_cfg=init_cfg)

    def forward(self, x):
        x = F.pad(
            x,
            (self.padding, self.padding, self.padding, self.padding),
        )
        batch_size, _, padded_h, padded_w = x.shape
        out_h = (padded_h - self.K) // self.stride + 1
        out_w = (padded_w - self.K) // self.stride + 1

        patches = x.unfold(2, self.K, self.stride).unfold(
            3,
            self.K,
            self.stride,
        )
        patches = patches.permute(0, 2, 3, 1, 4, 5).contiguous()
        patches = patches.view(
            batch_size,
            out_h * out_w,
            self.groups,
            self.logical_positions,
        )

        total_positions = self.lut_num * 6
        repeats = (total_positions - 1) // self.logical_positions + 1
        patches = patches.repeat(1, 1, 1, repeats)[..., :total_positions]
        patches = patches.view(
            batch_size,
            out_h * out_w,
            self.groups,
            self.lut_num,
            6,
        )

        w_q = binary_gumbel_softmax(self.weight, tau=1, hard=True)
        output = LUT6Function.apply(
            patches.contiguous(),
            w_q.contiguous(),
            self.groups,
        )
        output = output.sum(dim=-1).permute(0, 2, 1).contiguous()
        return output.view(batch_size, self.out_channels, out_h, out_w)


class PairLUT4ComplexConv2d(Module):
    """Map grouped binary complex activations through two LUT4/LUT6 banks."""

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=3,
        stride=1,
        padding=0,
        dilation=1,
        init_cfg=(-1.0, 0.1, 1.0, 0.1),
        parameterization="independent",
        categorical_temperature=1.0,
        lut_inputs=4,
        activation_encoding="standard",
        dominance_grad_mode="stop",
        dominance_ste_margin=1.0,
    ):
        super().__init__()
        if not isinstance(kernel_size, int):
            raise TypeError("PairLUT4ComplexConv2d requires an integer kernel_size")
        if dilation != 1:
            raise ValueError("PairLUT4ComplexConv2d currently supports dilation=1")
        if parameterization not in (
            "independent",
            "categorical",
            "categorical_residual",
        ):
            raise ValueError(
                "Unknown PairLUT4 parameterization: {}".format(parameterization)
            )
        if categorical_temperature <= 0:
            raise ValueError("categorical_temperature must be positive")
        if lut_inputs not in (4, 6):
            raise ValueError("PairLUT4 lut_inputs must be 4 or 6")
        if activation_encoding not in ("standard", "dominance"):
            raise ValueError(
                "Unknown PairLUT4 activation encoding: {}".format(
                    activation_encoding
                )
            )
        if activation_encoding == "dominance":
            if lut_inputs != 6:
                raise ValueError("Dominance encoding requires lut_inputs=6")
            if parameterization not in (
                "categorical",
                "categorical_residual",
            ):
                raise ValueError(
                    "Dominance encoding requires categorical parameterization"
                )
        if parameterization == "categorical_residual" and (
            activation_encoding != "dominance" or lut_inputs != 6
        ):
            raise ValueError(
                "Categorical residual parameterizations require dominance LUT6"
            )
        if dominance_ste_margin <= 0:
            raise ValueError("dominance_ste_margin must be positive")
        if dominance_grad_mode not in ("stop", "ste"):
            raise ValueError(
                "Unknown dominance gradient mode: {}".format(
                    dominance_grad_mode
                )
            )

        self.in_channels = int(in_channels)
        self.out_channels = int(out_channels)
        self.kernel_size = int(kernel_size)
        self.stride = int(stride)
        self.padding = int(padding)
        self.parameterization = parameterization
        self.categorical_temperature = float(categorical_temperature)
        self.lut_inputs = int(lut_inputs)
        self.activation_encoding = activation_encoding
        self.dominance_ste_margin = float(dominance_ste_margin)
        self.dominance_grad_mode = dominance_grad_mode
        self.LUT_INPUTS = self.lut_inputs
        self.complex_inputs_per_lut = (
            2 if self.activation_encoding == "dominance" else self.lut_inputs // 2
        )
        self.table_size = 1 << self.lut_inputs
        self.logical_positions = (
            self.in_channels * self.kernel_size * self.kernel_size
        )
        self.group_num = math.ceil(
            self.logical_positions / self.complex_inputs_per_lut
        )
        self.pair_num = self.group_num

        if self.parameterization == "independent":
            self.weight = nn.Parameter(
                torch.empty(
                    2 * self.out_channels,
                    self.group_num,
                    self.table_size,
                )
            )
            bimodal_initialization(self.weight, init_cfg=init_cfg)
        elif self.parameterization == "categorical":
            self.weight = nn.Parameter(
                torch.empty(
                    self.out_channels,
                    self.group_num,
                    self.table_size,
                    4,
                )
            )
            nn.init.normal_(self.weight, mean=0.0, std=0.01)
        else:
            self.weight = nn.Parameter(
                torch.zeros(
                    self.out_channels,
                    self.group_num,
                    self.table_size,
                    4,
                )
            )
            self.register_buffer(
                "dominance_base",
                torch.zeros(self.out_channels, self.group_num, 16, 4),
            )
            self.register_buffer(
                "dominance_residual_alpha",
                torch.tensor(0.1, dtype=torch.float32),
            )

        state_ids = torch.arange(self.table_size, dtype=torch.long)
        state_bits = torch.stack(
            [
                ((state_ids >> shift) & 1).to(torch.float32)
                for shift in reversed(range(self.lut_inputs))
            ],
            dim=-1,
        )
        self.register_buffer("state_bits", state_bits)
        if self.parameterization == "categorical_residual":
            old_addresses = (
                state_bits[:, 0].long() * 8
                + state_bits[:, 1].long() * 4
                + state_bits[:, 3].long() * 2
                + state_bits[:, 4].long()
            )
            self.register_buffer("dominance_old_addresses", old_addresses)
            self.register_buffer(
                "dominance_assignment",
                F.one_hot(old_addresses, num_classes=16).to(torch.float32),
            )
        self.register_buffer(
            "output_codebook",
            torch.tensor(
                [[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]]
            ),
        )

    @torch.no_grad()
    def initialize_from_binary_complex_weights(
        self,
        weight_real,
        weight_imag,
        logit_magnitude=1.0,
    ):
        """Compile Phase 2 binary complex weights into hard LUT categories."""
        if self.activation_encoding != "standard":
            raise RuntimeError(
                "Dominance LUT6 tables must keep their random initialization"
            )
        expected_shape = (
            self.out_channels,
            self.in_channels,
            self.kernel_size,
            self.kernel_size,
        )
        if tuple(weight_real.shape) != expected_shape:
            raise ValueError(
                "Unexpected real-weight shape {}; expected {}".format(
                    tuple(weight_real.shape), expected_shape
                )
            )
        if tuple(weight_imag.shape) != expected_shape:
            raise ValueError(
                "Unexpected imaginary-weight shape {}; expected {}".format(
                    tuple(weight_imag.shape), expected_shape
                )
            )
        if logit_magnitude <= 0:
            raise ValueError("logit_magnitude must be positive")

        device = self.weight.device
        dtype = self.weight.dtype
        weight_real = weight_real.to(device=device, dtype=dtype).sign()
        weight_imag = weight_imag.to(device=device, dtype=dtype).sign()

        group_ids = torch.arange(self.group_num, device=device).unsqueeze(1)
        offsets = torch.arange(
            self.complex_inputs_per_lut, device=device
        ).unsqueeze(0)
        positions = group_ids * self.complex_inputs_per_lut + offsets
        valid = positions < self.logical_positions
        positions = positions.remainder(self.logical_positions)
        kernel_area = self.kernel_size * self.kernel_size
        channels = torch.div(positions, kernel_area, rounding_mode="floor")
        spatial = positions.remainder(kernel_area)
        kernel_y = torch.div(
            spatial, self.kernel_size, rounding_mode="floor"
        )
        kernel_x = spatial.remainder(self.kernel_size)

        wr = weight_real[:, channels, kernel_y, kernel_x]
        wi = weight_imag[:, channels, kernel_y, kernel_x]
        valid = valid.to(dtype=dtype).unsqueeze(0)
        wr = wr * valid
        wi = wi * valid

        signed_states = self.state_bits.to(device=device, dtype=dtype) * 2.0 - 1.0
        states = signed_states.view(
            self.table_size,
            self.complex_inputs_per_lut,
            2,
        )
        xr = states[..., 0].view(
            1, 1, self.table_size, self.complex_inputs_per_lut
        )
        xi = states[..., 1].view(
            1, 1, self.table_size, self.complex_inputs_per_lut
        )
        wr = wr.unsqueeze(2)
        wi = wi.unsqueeze(2)
        output_real = (xr * wr - xi * wi).sum(dim=-1)
        output_imag = (xi * wr + xr * wi).sum(dim=-1)

        real_bits = output_real >= 0.0
        imag_bits = output_imag >= 0.0
        if self.parameterization == "independent":
            magnitude = self.weight.new_tensor(float(logit_magnitude))
            real_logits = torch.where(real_bits, magnitude, -magnitude)
            imag_logits = torch.where(imag_bits, magnitude, -magnitude)
            self.weight[: self.out_channels].copy_(real_logits)
            self.weight[self.out_channels :].copy_(imag_logits)
        else:
            target_classes = 2 * real_bits.long() + imag_bits.long()
            self.weight.zero_()
            self.weight.scatter_(
                dim=-1,
                index=target_classes.unsqueeze(-1),
                value=0.25,
            )

    def _dominance_bit(self, source):
        # delta = source.real.abs() - source.imag.abs()
        delta = torch.sign(source.imag)*source.real - torch.sign(source.real)*source.imag
        hard = (delta >= 0.0).to(delta.dtype)
        if self.dominance_grad_mode == "stop":
            return hard.detach()
        delta = binary_gumbel_softmax(delta, tau=1, hard=True)
        return delta

    def _group_patches(self, inp, dominance_source=None):
        real = F.unfold(
            inp.real,
            kernel_size=self.kernel_size,
            dilation=1,
            padding=self.padding,
            stride=self.stride,
        )
        imag = F.unfold(
            inp.imag,
            kernel_size=self.kernel_size,
            dilation=1,
            padding=self.padding,
            stride=self.stride,
        )
        batch_size, _, locations = real.shape
        if self.activation_encoding == "dominance":
            if dominance_source is None:
                raise ValueError(
                    "Dominance encoding requires the pre-binary activation"
                )
            if not torch.is_complex(dominance_source):
                raise TypeError("dominance_source must be complex")
            if dominance_source.shape != inp.shape:
                raise ValueError(
                    "dominance_source shape {} does not match input {}".format(
                        tuple(dominance_source.shape), tuple(inp.shape)
                    )
                )
            dominance = F.unfold(
                self._dominance_bit(dominance_source),
                kernel_size=self.kernel_size,
                dilation=1,
                padding=self.padding,
                stride=self.stride,
            )
            values = torch.stack((real, imag, dominance), dim=-1)
        else:
            values = torch.stack((real, imag), dim=-1)
        values = values.permute(0, 2, 1, 3).contiguous()
        remainder = self.logical_positions % self.complex_inputs_per_lut
        if remainder:
            padding = self.complex_inputs_per_lut - remainder
            values = torch.cat((values, values[:, :, :padding, :]), dim=2)
        return values.view(
            batch_size,
            locations,
            self.group_num,
            self.lut_inputs,
        )

    def _pair_patches(self, inp, dominance_source=None):
        """Backward-compatible name for the generalized LUT input grouping."""
        return self._group_patches(inp, dominance_source=dominance_source)

    def set_dominance_residual_alpha(self, alpha):
        if self.parameterization != "categorical_residual":
            return
        self.dominance_residual_alpha.fill_(float(alpha))

    def _categorical_logits(self):
        if self.parameterization != "categorical_residual":
            return self.weight
        addresses = self.dominance_old_addresses
        expanded_base = self.dominance_base.index_select(2, addresses)
        assignment = self.dominance_assignment.to(self.weight.dtype)
        residual_mean = torch.einsum(
            "ogsc,sa->ogac", self.weight, assignment
        ) / 4.0
        centered_residual = self.weight - residual_mean.index_select(
            2, addresses
        )
        return expanded_base + self.dominance_residual_alpha * centered_residual

    def _categorical_table(self):
        probabilities = F.softmax(
            self._categorical_logits() / self.categorical_temperature,
            dim=-1,
        )
        hard_selection = F.one_hot(
            probabilities.argmax(dim=-1),
            num_classes=4,
        ).to(probabilities.dtype)
        selection = probabilities + (hard_selection - probabilities).detach()
        table = torch.matmul(selection, self.output_codebook)
        return table.permute(3, 0, 1, 2).reshape(
            2 * self.out_channels,
            self.group_num,
            self.table_size,
        )

    def materialize_table(self):
        """Return the two hard LUT banks with the configured STE proxy."""
        if self.parameterization == "independent":
            return hard_binary_table(self.weight)
        return self._categorical_table()

    def _reference_forward(self, patches, table):
        bits = self.state_bits.to(device=patches.device, dtype=patches.dtype)
        basis = torch.where(
            bits.view(
                1, 1, 1, self.table_size, self.lut_inputs
            ) > 0.5,
            patches.unsqueeze(-2),
            1.0 - patches.unsqueeze(-2),
        ).prod(dim=-1)
        return torch.einsum("brps,ops->brop", basis, table).sum(dim=-1)

    @staticmethod
    def _expand_lut4_for_lut6(table):
        return table.unsqueeze(-1).expand(-1, -1, -1, 4).reshape(
            table.size(0),
            table.size(1),
            64,
        )

    def forward(self, inp, dominance_source=None):
        if not torch.is_complex(inp):
            raise TypeError("PairLUT4ComplexConv2d requires a complex input")
        patches = self._group_patches(
            inp,
            dominance_source=dominance_source,
        )
        table = self.materialize_table()

        if patches.is_cuda:
            if self.lut_inputs == 4:
                dummy = patches.new_zeros(*patches.shape[:-1], 2)
                lut6_inputs = torch.cat((patches, dummy), dim=-1)
                lut6_table = self._expand_lut4_for_lut6(table)
            else:
                lut6_inputs = patches
                lut6_table = table
            output = LUT6Function.apply(
                lut6_inputs.unsqueeze(2).contiguous(),
                lut6_table.contiguous(),
                1,
            ).sum(dim=-1)
        else:
            output = self._reference_forward(patches, table)

        output = output.permute(0, 2, 1).contiguous()
        output_r, output_i = output.split(self.out_channels, dim=1)
        out_h = (
            inp.size(2) + 2 * self.padding - self.kernel_size
        ) // self.stride + 1
        out_w = (
            inp.size(3) + 2 * self.padding - self.kernel_size
        ) // self.stride + 1
        return torch.complex(
            output_r.view(inp.size(0), self.out_channels, out_h, out_w),
            output_i.view(inp.size(0), self.out_channels, out_h, out_w),
        )

    def extra_repr(self):
        return (
            "in_channels={}, out_channels={}, kernel_size={}, stride={}, "
            "padding={}, groups={}, lut_inputs={}, parameterization={}, "
            "activation_encoding={}"
        ).format(
            self.in_channels,
            self.out_channels,
            self.kernel_size,
            self.stride,
            self.padding,
            self.group_num,
            self.lut_inputs,
            self.parameterization,
            self.activation_encoding,
        )


class TripleLUT6ComplexConv2d(Module):
    """Map triples of binary complex activations through real/imag LUT6s."""

    LUT_INPUTS = 6

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=3,
        stride=1,
        padding=0,
        dilation=1,
        init_cfg=(-1.0, 0.1, 1.0, 0.1),
    ):
        super().__init__()
        if not isinstance(kernel_size, int):
            raise TypeError(
                "TripleLUT6ComplexConv2d requires an integer kernel_size"
            )
        if dilation != 1:
            raise ValueError(
                "TripleLUT6ComplexConv2d currently supports dilation=1"
            )

        self.in_channels = int(in_channels)
        self.out_channels = int(out_channels)
        self.kernel_size = int(kernel_size)
        self.stride = int(stride)
        self.padding = int(padding)
        self.groups_per_position = math.ceil(self.in_channels / 3)
        self.lut_num = (
            self.kernel_size
            * self.kernel_size
            * self.groups_per_position
        )

        # The first out_channels tables produce real bits; the remainder
        # produce imaginary bits.
        self.weight = nn.Parameter(
            torch.empty(2 * self.out_channels, self.lut_num, 64)
        )
        bimodal_initialization(self.weight, init_cfg=init_cfg)

        state_ids = torch.arange(64, dtype=torch.long)
        state_bits = torch.stack(
            [
                ((state_ids >> shift) & 1).to(torch.float32)
                for shift in (5, 4, 3, 2, 1, 0)
            ],
            dim=-1,
        )
        self.register_buffer("state_bits", state_bits)

    @torch.no_grad()
    def initialize_from_binary_complex_weights(
        self,
        weight_real,
        weight_imag,
        logit_magnitude=1.0,
    ):
        """Compile a Phase 2 binary complex convolution into LUT4 tables."""
        expected_shape = (
            self.out_channels,
            self.in_channels,
            self.kernel_size,
            self.kernel_size,
        )
        if tuple(weight_real.shape) != expected_shape:
            raise ValueError(
                "Unexpected real-weight shape {}; expected {}".format(
                    tuple(weight_real.shape), expected_shape
                )
            )
        if tuple(weight_imag.shape) != expected_shape:
            raise ValueError(
                "Unexpected imaginary-weight shape {}; expected {}".format(
                    tuple(weight_imag.shape), expected_shape
                )
            )
        if logit_magnitude <= 0:
            raise ValueError("logit_magnitude must be positive")

        device = self.weight.device
        dtype = self.weight.dtype
        weight_real = weight_real.to(device=device, dtype=dtype).sign()
        weight_imag = weight_imag.to(device=device, dtype=dtype).sign()

        pair_ids = torch.arange(self.pair_num, device=device)
        spatial_ids = torch.div(
            pair_ids,
            self.pairs_per_position,
            rounding_mode="floor",
        )
        pair_slots = pair_ids.remainder(self.pairs_per_position)
        channel0 = 2 * pair_slots
        channel1 = channel0 + 1
        channel1_valid = channel1 < self.in_channels
        channel1 = channel1.clamp(max=self.in_channels - 1)
        kernel_y = torch.div(
            spatial_ids,
            self.kernel_size,
            rounding_mode="floor",
        )
        kernel_x = spatial_ids.remainder(self.kernel_size)

        wr0 = weight_real[:, channel0, kernel_y, kernel_x]
        wi0 = weight_imag[:, channel0, kernel_y, kernel_x]
        wr1 = weight_real[:, channel1, kernel_y, kernel_x]
        wi1 = weight_imag[:, channel1, kernel_y, kernel_x]
        valid = channel1_valid.to(dtype=dtype).unsqueeze(0)
        wr1 = wr1 * valid
        wi1 = wi1 * valid

        signed_states = self.state_bits.to(device=device, dtype=dtype) * 2.0 - 1.0
        xr0, xi0, xr1, xi1 = (
            component.view(1, 1, 16)
            for component in signed_states.unbind(dim=-1)
        )
        wr0, wi0, wr1, wi1 = (
            component.unsqueeze(-1)
            for component in (wr0, wi0, wr1, wi1)
        )
        output_real = xr0 * wr0 - xi0 * wi0 + xr1 * wr1 - xi1 * wi1
        output_imag = xi0 * wr0 + xr0 * wi0 + xi1 * wr1 + xr1 * wi1

        magnitude = self.weight.new_tensor(float(logit_magnitude))
        real_logits = torch.where(output_real >= 0.0, magnitude, -magnitude)
        imag_logits = torch.where(output_imag >= 0.0, magnitude, -magnitude)
        self.weight[: self.out_channels].copy_(real_logits)
        self.weight[self.out_channels :].copy_(imag_logits)

    def _triple_patches(self, inp):
        real = F.unfold(
            inp.real,
            kernel_size=self.kernel_size,
            dilation=1,
            padding=self.padding,
            stride=self.stride,
        )
        imag = F.unfold(
            inp.imag,
            kernel_size=self.kernel_size,
            dilation=1,
            padding=self.padding,
            stride=self.stride,
        )
        batch_size, _, locations = real.shape
        kernel_area = self.kernel_size * self.kernel_size
        real = real.view(
            batch_size,
            self.in_channels,
            kernel_area,
            locations,
        ).permute(0, 3, 2, 1)
        imag = imag.view(
            batch_size,
            self.in_channels,
            kernel_area,
            locations,
        ).permute(0, 3, 2, 1)
        values = torch.stack((real, imag), dim=-1).contiguous()

        tail = (-self.in_channels) % 3
        if tail:
            values = torch.cat((values, values[:, :, :, :tail, :]), dim=3)
        return values.view(batch_size, locations, self.lut_num, 6)

    def _reference_forward(self, patches, table):
        bits = self.state_bits.to(device=patches.device, dtype=patches.dtype)
        basis = torch.where(
            bits.view(1, 1, 1, 64, 6) > 0.5,
            patches.unsqueeze(-2),
            1.0 - patches.unsqueeze(-2),
        ).prod(dim=-1)
        return torch.einsum("brls,ols->brol", basis, table).sum(dim=-1)

    def forward(self, inp):
        if not torch.is_complex(inp):
            raise TypeError("TripleLUT6ComplexConv2d requires a complex input")
        patches = self._triple_patches(inp)
        table = hard_binary_table(self.weight)

        if patches.is_cuda:
            lut6_inputs = patches.unsqueeze(2)
            output = LUT6Function.apply(
                lut6_inputs.contiguous(),
                table.contiguous(),
                1,
            ).sum(dim=-1)
        else:
            output = self._reference_forward(patches, table)

        output = output.permute(0, 2, 1).contiguous()
        output_r, output_i = output.split(self.out_channels, dim=1)
        out_h = (
            inp.size(2) + 2 * self.padding - self.kernel_size
        ) // self.stride + 1
        out_w = (
            inp.size(3) + 2 * self.padding - self.kernel_size
        ) // self.stride + 1
        return torch.complex(
            output_r.view(inp.size(0), self.out_channels, out_h, out_w),
            output_i.view(inp.size(0), self.out_channels, out_h, out_w),
        )

    def extra_repr(self):
        return (
            "in_channels={}, out_channels={}, kernel_size={}, stride={}, "
            "padding={}, lut_num={}"
        ).format(
            self.in_channels,
            self.out_channels,
            self.kernel_size,
            self.stride,
            self.padding,
            self.lut_num,
        )


class TwoLUTComplexConv2d(Module):
    """Complex convolution with shared real- and imaginary-weight LUTConvs."""

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=3,
        stride=1,
        padding=0,
        dilation=1,
        group_num=1,
        init_cfg=(-1.0, 0.1, 1.0, 0.1),
    ):
        super().__init__()
        if dilation != 1:
            raise ValueError("TwoLUTComplexConv2d currently supports dilation=1")
        branch_kwargs = {
            "in_channels": in_channels,
            "out_channels": out_channels,
            "kernel_size": kernel_size,
            "stride": stride,
            "padding": padding,
            "group_num": group_num,
            "init_cfg": init_cfg,
        }
        self.conv_r = LUTBinaryConv2d(**branch_kwargs)
        self.conv_i = LUTBinaryConv2d(**branch_kwargs)
        self.stride = stride
        self.padding = padding

    def forward(self, inp):
        rr = self.conv_r(inp.real)
        ii = self.conv_i(inp.imag)
        ri = self.conv_r(inp.imag)
        ir = self.conv_i(inp.real)
        return torch.complex(rr - ii, ri + ir)

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

from .lut_backend import LUTFloatingConvFunction, LUTBinaryConvFunction
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
    binary_scale_weight_complex,
    binary_sign,
)


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
    ):
        super().__init__()
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

    def forward(self, inp):
        weight = torch.complex(self.conv_r.weight, self.conv_i.weight)
        weight = complex_binary_weight(
            weight,
            per_channel=self.per_channel,
            grad_mode=self.weight_grad_mode,
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

class LUTAwareComplexBinaryConv2d(Module):
    """
    硬件感知复数二值卷积层 (Phase 3: LUT-Aware Transition Phase)。
    将局部复数乘法映射为 1-LUT 承载，严格输出 {0, 1} 物理非对称累加结果。
    通过无损布尔多项式展开，彻底解决 unfold 带来的显存 OOM 灾难。
    """
    def __init__(
        self, in_channels, out_channels, kernel_size=3, stride=1, padding=0, 
        dilation=1, groups=1, bias=False, per_channel=True, weight_grad_mode="ste"
    ):
        super().__init__()
        self.stride = stride
        self.padding = padding
        self.kernel_size = kernel_size
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.dilation = dilation
        self.groups = groups
        self.per_channel = per_channel
        self.weight_grad_mode = weight_grad_mode

        # 硬件累加的极限范围 [0, N]
        self.N = in_channels * kernel_size * kernel_size // groups

        # 必须叫 conv_r 和 conv_i，与 BinaryComplexConv2d 100% 对齐，实现无缝 Load
        self.conv_r = Conv2d(in_channels, out_channels, kernel_size, stride, 0, dilation, groups, bias)
        self.conv_i = Conv2d(in_channels, out_channels, kernel_size, stride, 0, dilation, groups, bias)

    def forward(self, inp):
        # =====================================================================
        # 1. 边缘物理对齐补丁 (The Padding Alignment)
        # 强行用 -1.0 填充边缘，使其在布尔逻辑中等效于输入全 0 (物理低电平)
        # =====================================================================
        if self.padding > 0:
            pad_tuple = (self.padding, self.padding, self.padding, self.padding)
            x_r_pad = F.pad(inp.real, pad_tuple, "constant", -1.0)
            x_i_pad = F.pad(inp.imag, pad_tuple, "constant", -1.0)
        else:
            x_r_pad = inp.real
            x_i_pad = inp.imag

        # =====================================================================
        # 2. 原始 BNN 二值化 (提取比例因子 alpha)
        # =====================================================================
        weight_r = self.conv_r.weight
        weight_i = self.conv_i.weight
        complex_w = torch.complex(weight_r, weight_i)
        
        # 复数统一缩放因子 (alpha)
        alpha = binary_scale_weight_complex(complex_w, per_channel=self.per_channel)
        alpha = alpha.view(1, -1, 1, 1)

        w_r_bin = binary_sign(weight_r, grad_mode=self.weight_grad_mode)
        w_i_bin = binary_sign(weight_i, grad_mode=self.weight_grad_mode)
        x_r_bin = binary_sign(x_r_pad, grad_mode="ste")
        x_i_bin = binary_sign(x_i_pad, grad_mode="ste")

        # =====================================================================
        # 3. 原始 BNN 数学理想输出 (用于反向传播的平滑梯度)
        # =====================================================================
        conv_rr = F.conv2d(x_r_bin, w_r_bin, stride=self.stride, dilation=self.dilation, groups=self.groups)
        conv_ii = F.conv2d(x_i_bin, w_i_bin, stride=self.stride, dilation=self.dilation, groups=self.groups)
        conv_ri = F.conv2d(x_r_bin, w_i_bin, stride=self.stride, dilation=self.dilation, groups=self.groups)
        conv_ir = F.conv2d(x_i_bin, w_r_bin, stride=self.stride, dilation=self.dilation, groups=self.groups)
        
        y_orig_r = alpha * (conv_rr - conv_ii)
        y_orig_i = alpha * (conv_ri + conv_ir)

        # =====================================================================
        # 4. 硬件真实的 0/1 LUT 截断结果 (无损 O(1) 空间展开)
        # =====================================================================
        with torch.no_grad():
            # 交叉项：用来捕获阶跃函数的非对称截断误差
            conv_cross = F.conv2d(x_r_bin * x_i_bin, w_r_bin * w_i_bin, stride=self.stride, dilation=self.dilation, groups=self.groups)

            # LUT 查表后的物理累加值，严格在 [0, N] 之间
            y_lut_r = 0.25 * (3 * self.N + conv_rr - conv_ii + conv_cross)
            y_lut_i = 0.25 * (3 * self.N + conv_ri + conv_ir - conv_cross)

        # =====================================================================
        # 5. 零震荡对齐 (Zero-Shock Alignment) & STE 穿透
        # =====================================================================
        # 将 [0, N] 的硬件输出映射回数学期望域，使得 BN 层不会崩溃！
        y_hw_math_r = (y_lut_r - self.N / 2.0) * 4.0 * alpha
        y_hw_math_i = (y_lut_i - self.N / 2.0) * 4.0 * alpha

        # 魔法：前向是包含硬件误差的 LUT 结果，反向是平滑的复数乘加真实梯度！
        y_r = y_hw_math_r.detach() - y_orig_r.detach() + y_orig_r
        y_i = y_hw_math_i.detach() - y_orig_i.detach() + y_orig_i

        # 加入原始偏差
        if self.conv_r.bias is not None:
            y_r += self.conv_r.bias.view(1, -1, 1, 1)
            y_i += self.conv_i.bias.view(1, -1, 1, 1)

        return torch.complex(y_r, y_i)

# =====================================================================
# 1. 核心算子：原汁原味的 BNN 直通估计器 (STE)
# =====================================================================
class SignWithSTE(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        # 前向传播：严格二值化为 0.0 和 1.0 (完美契合查表索引概率)
        return torch.where(x >= 0, 
                           torch.tensor(1.0, dtype=x.dtype, device=x.device), 
                           torch.tensor(0.0, dtype=x.dtype, device=x.device))
    
    @staticmethod
    def backward(ctx, grad_output):
        # 反向传播：梯度直接穿透 (STE 原理)
        # 如果需要更稳定的训练，可以加上 clamp: return grad_output.clamp(-1, 1)
        return grad_output


# =====================================================================
# 2. 核心算子：全局真值表的退火函数
# =====================================================================
def binary_annealing(logits, tau=1.0, hard=False):
    """
    连续退火与直通估计器，专用于将 Logits 挤压为 0~1 的物理逻辑状态。
    tau 越大，曲线越陡峭，越逼近绝对二值。
    """
    y = F.tanh(logits * tau)
    soft_sample = (y + 1.0) / 2.0
    binary_hard = (logits >= 0).to(logits.dtype)
    hard_sample = (binary_hard - logits).detach() + logits

    if torch.is_tensor(hard):
        return torch.where(hard.to(device=logits.device, dtype=torch.bool), hard_sample, soft_sample)
    return hard_sample if hard else soft_sample


# =====================================================================
# 3. 终极硬件感知复数 LUT 卷积层 (5-Phase Architecture)
# =====================================================================
class ComplexLUTConv2d(Module):
    """
    Phase 3: LUT-Aware BNN (真值表固定，仅用 STE 训练空间权重)
    Phase 4: Joint Annealing (真值表解锁退火，空间权重继续 STE)
    Phase 5: Hard Binary (物理部署，全二值纯逻辑门运行)

    lut_sets controls how many independently learnable real/imag LUT pairs are
    available when LUT allocation is layer-level. With channel-level allocation,
    lut_sets_per_channel controls how many LUT pairs each output channel owns,
    so the physical LUT count is out_channels * lut_sets_per_channel.
    """
    def __init__(
        self, in_channels, out_channels, kernel_size=3, stride=1, padding=1,
        groups=1, phase=3, lut_sets=1, lut_allocation="layer", lut_sets_per_channel=1
    ):
        super().__init__()
        if lut_sets < 1:
            raise ValueError("lut_sets must be at least 1")
        if lut_sets_per_channel < 1:
            raise ValueError("lut_sets_per_channel must be at least 1")
        if lut_allocation not in ("layer", "channel"):
            raise ValueError("lut_allocation must be 'layer' or 'channel'")

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.groups = groups
        self.phase = phase
        self.lut_allocation = lut_allocation
        self.layer_lut_sets = lut_sets
        self.lut_sets_per_channel = lut_sets_per_channel
        self.lut_sets = lut_sets if lut_allocation == "layer" else out_channels * lut_sets_per_channel
        
        # 折叠魔法：16状态LUT吸收固定权重后，底层CUDA内核只需处理 K=2
        self.LUT_K = 2 
        self.lut_num = in_channels * kernel_size * kernel_size // groups

        # ---------------------------------------------------------------------
        # A. 经典复数空间权重 (潜权重 Logits)
        # ---------------------------------------------------------------------
        self.weight_r = Parameter(torch.Tensor(out_channels, in_channels // groups, kernel_size, kernel_size))
        self.weight_i = Parameter(torch.Tensor(out_channels, in_channels // groups, kernel_size, kernel_size))
        nn.init.kaiming_uniform_(self.weight_r, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.weight_i, a=math.sqrt(5))

        # ---------------------------------------------------------------------
        # B. 可配置分配粒度的 LUT (转换为 Logits)
        # ---------------------------------------------------------------------
        # 标准复数乘法的布尔逻辑 (16种状态)
        init_lut_r = torch.tensor([1, 1, 0, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 0, 1, 1], dtype=torch.float32)
        init_lut_i = torch.tensor([1, 1, 1, 0, 1, 0, 1, 1, 1, 1, 0, 1, 0, 1, 1, 1], dtype=torch.float32)

        # 🌟 将 0/1 映射为极端的 Logits (-5.0 和 5.0)，保证退火初期的绝对记忆
        logit_lut_r = torch.where(init_lut_r == 1, 5.0, -5.0)
        logit_lut_i = torch.where(init_lut_i == 1, 5.0, -5.0)

        self.lut_r = Parameter(logit_lut_r.repeat(self.lut_sets, 1))
        self.lut_i = Parameter(logit_lut_i.repeat(self.lut_sets, 1))

        # 只有在 Phase 4 (联合优化) 时，LUT 才是可学习的参数
        if self.phase != 4:
            self.lut_r.requires_grad = False
            self.lut_i.requires_grad = False

        # ---------------------------------------------------------------------
        # C. 预计算硬件寻址连接 (Hardware Offsets) & 状态寄存器
        # ---------------------------------------------------------------------
        conn_c = torch.arange(in_channels).repeat_interleave(kernel_size * kernel_size)
        conn_dy = (torch.arange(kernel_size * kernel_size) // kernel_size).repeat(in_channels)
        conn_dx = (torch.arange(kernel_size * kernel_size) % kernel_size).repeat(in_channels)

        # 构建 K=2 交叉采样 (偶数取 X_r, 奇数取 X_i)
        flat_c = torch.empty(self.lut_num * 2, dtype=torch.int32)
        flat_c[0::2] = conn_c              
        flat_c[1::2] = conn_c + in_channels 

        flat_dy = torch.empty(self.lut_num * 2, dtype=torch.int32)
        flat_dy[0::2], flat_dy[1::2] = conn_dy, conn_dy

        flat_dx = torch.empty(self.lut_num * 2, dtype=torch.int32)
        flat_dx[0::2], flat_dx[1::2] = conn_dx, conn_dx

        self.register_buffer('flat_c', flat_c)
        self.register_buffer('flat_dy', flat_dy)
        self.register_buffer('flat_dx', flat_dx)
        self.register_buffer('shifts', flat_c % 32)

        if lut_allocation == "layer":
            # Layer-level allocation: output channels share a fixed pool of LUT
            # sets round-robin. This is the original multi-LUT behavior.
            output_lut_set_ids = torch.arange(out_channels, dtype=torch.long) % lut_sets
            lut_set_ids = output_lut_set_ids.unsqueeze(0).expand(self.lut_num, -1).clone()
        else:
            # Channel-level allocation: each output channel owns
            # lut_sets_per_channel LUT sets. Within a channel, logical LUTs from
            # different input/kernel positions use that channel-local pool
            # round-robin.
            channel_base = torch.arange(out_channels, dtype=torch.long).unsqueeze(0) * lut_sets_per_channel
            channel_local_ids = (torch.arange(self.lut_num, dtype=torch.long) % lut_sets_per_channel).unsqueeze(1)
            lut_set_ids = channel_base + channel_local_ids
        self.register_buffer('lut_set_ids', lut_set_ids)
        
        # 退火超参数控制台
        self.register_buffer("tau", torch.tensor(1.0))
        self.register_buffer("hard", torch.tensor(0.0 if self.phase != 5 else 1.0))

    def forward(self, inp):
        x_r, x_i = inp.real, inp.imag
        x_cat = torch.cat([x_r, x_i], dim=1) # [B, 2*in_C, H, W]
        B, _, H, W = x_cat.shape
        
        padded_W = W + 2 * self.padding
        packed_C = (2 * self.in_channels + 31) // 32
        
        # 动态计算绝对内存偏移 (支持任意分辨率输入)
        offsets = (self.flat_dy * padded_W + self.flat_dx) * packed_C + (self.flat_c // 32)

        complex_w = torch.complex(self.weight_r, self.weight_i)
        alpha = binary_scale_weight_complex(complex_w, per_channel=True).view(1, -1, 1, 1)

        # 展平空间权重 Logits
        wr = self.weight_r.view(self.out_channels, self.lut_num).t() 
        wi = self.weight_i.view(self.out_channels, self.lut_num).t()

        current_tau = self.tau
        is_hard = self.hard

        if self.phase in [3, 4]:
            # =================================================================
            # Phase 3/4 核心：异构向前传播 (Heterogeneous Forward Pass)
            # =================================================================
            
            # 1. 空间权重坚守阵地：原生纯二值化 + STE，保留前序特征提取能力
            pr = SignWithSTE.apply(wr)
            pi = SignWithSTE.apply(wi)
            
            # 由于 pr/pi 是纯 One-Hot (非0即1)，这里变成了纯粹的离散路由选择
            p00 = (1 - pr) * (1 - pi)
            p01 = (1 - pr) * pi
            p10 = pr * (1 - pi)
            p11 = pr * pi

            # 2. 多组 LUT 探索空间：退火映射
            L_r = binary_annealing(self.lut_r, tau=current_tau, hard=is_hard)
            L_i = binary_annealing(self.lut_i, tau=current_tau, hard=is_hard)

            # 3. 概率折叠：将 16 状态 LUT 吸纳权重，坍缩为 K=2 的 4 状态浮点表
            selected_r = L_r[self.lut_set_ids]
            selected_i = L_i[self.lut_set_ids]

            v00_r = p00 * selected_r[..., 0] + p01 * selected_r[..., 1] + p10 * selected_r[..., 2] + p11 * selected_r[..., 3]
            v01_r = p00 * selected_r[..., 4] + p01 * selected_r[..., 5] + p10 * selected_r[..., 6] + p11 * selected_r[..., 7]
            v10_r = p00 * selected_r[..., 8] + p01 * selected_r[..., 9] + p10 * selected_r[..., 10] + p11 * selected_r[..., 11]
            v11_r = p00 * selected_r[..., 12] + p01 * selected_r[..., 13] + p10 * selected_r[..., 14] + p11 * selected_r[..., 15]
            w_folded_r = torch.stack([v00_r, v01_r, v10_r, v11_r], dim=1)

            v00_i = p00 * selected_i[..., 0] + p01 * selected_i[..., 1] + p10 * selected_i[..., 2] + p11 * selected_i[..., 3]
            v01_i = p00 * selected_i[..., 4] + p01 * selected_i[..., 5] + p10 * selected_i[..., 6] + p11 * selected_i[..., 7]
            v10_i = p00 * selected_i[..., 8] + p01 * selected_i[..., 9] + p10 * selected_i[..., 10] + p11 * selected_i[..., 11]
            v11_i = p00 * selected_i[..., 12] + p01 * selected_i[..., 13] + p10 * selected_i[..., 14] + p11 * selected_i[..., 15]
            w_folded_i = torch.stack([v00_i, v01_i, v10_i, v11_i], dim=1)

            # 上一层传递过来的激活是 [-1, 1]，映射为浮点多项式期望的 [0, 1] 概率
            x_prob = (x_cat + 1.0) / 2.0
            float_offsets = (self.flat_dy * padded_W + self.flat_dx) * (2 * self.in_channels) + self.flat_c
            
            # 交给底层的 C++ 极速浮点多线性插值引擎；offsets 决定每个 LUT 的 K 个输入连接。
            out_r = LUTFloatingConvFunction.apply(x_prob, w_folded_r, float_offsets, self.groups, self.LUT_K, self.kernel_size, self.stride, self.padding)
            out_i = LUTFloatingConvFunction.apply(x_prob, w_folded_i, float_offsets, self.groups, self.LUT_K, self.kernel_size, self.stride, self.padding)

        else:
            # =================================================================
            # Phase 5: Hard Binary (物理部署，彻底转换为位运算)
            # =================================================================
            # 空间权重纯粹硬截断
            wr_bit = (wr > 0).long()
            wi_bit = (wi > 0).long()
            base_idx = wr_bit * 2 + wi_bit
            
            # LUT 也进行硬截断
            L_r_int = (self.lut_r >= 0).long()
            L_i_int = (self.lut_i >= 0).long()

            # 将被吸收后的 4状态 K=2 真值表打包为 int64 喂给底层
            w_packed_r = (
                (L_r_int[self.lut_set_ids, 12 + base_idx] << 3)
                | (L_r_int[self.lut_set_ids, 8 + base_idx] << 2)
                | (L_r_int[self.lut_set_ids, 4 + base_idx] << 1)
                | L_r_int[self.lut_set_ids, base_idx]
            )
            w_packed_i = (
                (L_i_int[self.lut_set_ids, 12 + base_idx] << 3)
                | (L_i_int[self.lut_set_ids, 8 + base_idx] << 2)
                | (L_i_int[self.lut_set_ids, 4 + base_idx] << 1)
                | L_i_int[self.lut_set_ids, base_idx]
            )

            # 交给底层的 C++ __ballot_sync 按位同或并发位运算引擎
            out_r = LUTBinaryConvFunction.apply(x_cat, w_packed_r, offsets, self.shifts, self.groups, self.LUT_K, self.stride, self.padding)
            out_i = LUTBinaryConvFunction.apply(x_cat, w_packed_i, offsets, self.shifts, self.groups, self.LUT_K, self.stride, self.padding)

        # 零震荡对齐，并恢复 phase3 中使用的复数权重缩放因子。
        out_r = (out_r - (self.lut_num / 2.0)) * 4.0 * alpha
        out_i = (out_i - (self.lut_num / 2.0)) * 4.0 * alpha

        return torch.complex(out_r, out_i)
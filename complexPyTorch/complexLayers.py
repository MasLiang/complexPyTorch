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
    LUTFloatingConvFunction,
    LUTFloatingShadowInputGradFunction,
    LUTBinaryConvFunction,
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
    binary_scale_weight_complex,
    binary_sign,
)


def physical_magnitude_threshold(raw_threshold):
    """Map an unconstrained parameter to a positive magnitude threshold."""
    return F.softplus(raw_threshold)

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


_C8_CODEBOOK_MODES = ("roots", "octants")
_C8_GRAD_MODES = (
    "softmax",
    "bireal_ste",
    "semantic_ste",
    "semantic_phase_ste",
)


def _c8_unit_codebook(mode="roots"):
    if mode not in _C8_CODEBOOK_MODES:
        raise ValueError(
            "C8 codebook mode must be one of {}".format(_C8_CODEBOOK_MODES)
        )
    inv_sqrt2 = 1.0 / math.sqrt(2.0)
    if mode == "roots":
        return torch.tensor(
            [
                [-inv_sqrt2, -inv_sqrt2],
                [-1.0, 0.0],
                [-inv_sqrt2, inv_sqrt2],
                [0.0, 1.0],
                [inv_sqrt2, inv_sqrt2],
                [1.0, 0.0],
                [inv_sqrt2, -inv_sqrt2],
                [0.0, -1.0],
            ],
            dtype=torch.float32,
        )

    high = math.cos(math.pi / 8.0)
    low = math.sin(math.pi / 8.0)
    return torch.tensor(
        [
            [-high, -low],
            [-high, low],
            [-low, high],
            [low, high],
            [high, low],
            [high, -low],
            [low, -high],
            [-low, -high],
        ],
        dtype=torch.float32,
    )


def _c8_gray_bits():
    phase_indices = torch.arange(8, dtype=torch.long)
    gray_addresses = phase_indices ^ (phase_indices >> 1)
    return torch.stack(
        [
            (gray_addresses >> 2) & 1,
            (gray_addresses >> 1) & 1,
            gray_addresses & 1,
        ],
        dim=1,
    ).to(torch.float32)


class C8ComplexActivation(Module):
    """Hard nearest-C8 activation with a selectable QAT backward path."""

    def __init__(
        self,
        beta=2.0,
        codebook_mode="roots",
        grad_mode="softmax",
    ):
        super().__init__()
        if beta <= 0.0:
            raise ValueError("C8 beta must be positive")
        if codebook_mode not in _C8_CODEBOOK_MODES:
            raise ValueError(
                "C8 codebook mode must be one of {}".format(
                    _C8_CODEBOOK_MODES
                )
            )
        if grad_mode not in _C8_GRAD_MODES:
            raise ValueError(
                "C8 gradient mode must be one of {}".format(_C8_GRAD_MODES)
            )
        if (
            grad_mode in ("semantic_ste", "semantic_phase_ste")
            and codebook_mode != "octants"
        ):
            raise ValueError(
                "C8 {} requires the sign-preserving octants codebook".format(
                    grad_mode
                )
            )
        self.c8_codebook_mode = codebook_mode
        self.c8_grad_mode = grad_mode
        codebook = _c8_unit_codebook(codebook_mode)
        gray_bits = _c8_gray_bits()
        self.register_buffer("c8_codebook", codebook)
        self.register_buffer("c8_gray_bits", gray_bits)

        semantic_states = torch.arange(8, dtype=torch.long)
        semantic_state_bits = torch.stack(
            [
                (semantic_states >> 2) & 1,
                (semantic_states >> 1) & 1,
                semantic_states & 1,
            ],
            dim=1,
        ).to(torch.float32)
        semantic_to_gray = torch.zeros(8, 3, dtype=torch.float32)
        if codebook_mode == "octants":
            sign_r = (codebook[:, 0] >= 0).to(torch.long)
            sign_i = (codebook[:, 1] >= 0).to(torch.long)
            dominance = (
                codebook[:, 0].abs() >= codebook[:, 1].abs()
            ).to(torch.long)
            semantic_address = sign_r * 4 + sign_i * 2 + dominance
            semantic_to_gray[semantic_address] = gray_bits
        self.register_buffer("c8_semantic_state_bits", semantic_state_bits)
        self.register_buffer("c8_semantic_to_gray", semantic_to_gray)
        self.register_buffer("c8_beta", torch.tensor(float(beta)))

    @torch.no_grad()
    def set_beta(self, beta):
        beta = float(beta)
        if beta <= 0.0:
            raise ValueError("C8 beta must be positive")
        self.c8_beta.fill_(beta)

    def _phase_scores(self, inp):
        if not torch.is_complex(inp):
            raise TypeError("C8ComplexActivation expects a complex tensor")
        magnitude = torch.sqrt(
            inp.real.square() + inp.imag.square()
        ).clamp_min(1e-6)
        unit_r = inp.real / magnitude
        unit_i = inp.imag / magnitude
        codebook = self.c8_codebook.to(dtype=unit_r.dtype)
        return (
            unit_r.unsqueeze(-1) * codebook[:, 0]
            + unit_i.unsqueeze(-1) * codebook[:, 1]
        )

    def hard_phase_indices(self, inp):
        return self._phase_scores(inp).argmax(dim=-1)

    def _semantic_octant_proxy(self, inp, phase_normalized=False):
        signed_r = binary_sign(inp.real, grad_mode="bireal")
        signed_i = binary_sign(inp.imag, grad_mode="bireal")

        beta = self.c8_beta.to(dtype=inp.real.dtype)
        dominance_score = inp.real.abs() - inp.imag.abs()
        if phase_normalized:
            epsilon = max(torch.finfo(inp.real.dtype).eps, 1e-6)
            magnitude = torch.sqrt(
                inp.real.square()
                + inp.imag.square()
                + epsilon * epsilon
            )
            dominance_score = dominance_score / magnitude
        soft_dominance = torch.sigmoid(beta * dominance_score)
        hard_dominance = (inp.real.abs() >= inp.imag.abs()).to(
            dtype=inp.real.dtype
        )
        dominance = soft_dominance + (
            hard_dominance - soft_dominance
        ).detach()

        codebook = self.c8_codebook.to(dtype=inp.real.dtype)
        high = codebook.abs().amax()
        low = codebook.abs().amin()
        magnitude_r = low + (high - low) * dominance
        magnitude_i = high - (high - low) * dominance
        return torch.stack(
            (signed_r * magnitude_r, signed_i * magnitude_i),
            dim=-1,
        )

    def _hard_soft_values(self, inp):
        scores = self._phase_scores(inp)
        hard_index = scores.argmax(dim=-1)
        codebook = self.c8_codebook.to(dtype=scores.dtype)
        hard_value = codebook[hard_index]
        if self.c8_grad_mode == "softmax":
            beta = self.c8_beta.to(dtype=scores.dtype)
            proxy_value = torch.softmax(scores * beta, dim=-1) @ codebook
        elif self.c8_grad_mode == "bireal_ste":
            proxy_value = torch.stack(
                (
                    binary_sign(inp.real, grad_mode="bireal"),
                    binary_sign(inp.imag, grad_mode="bireal"),
                ),
                dim=-1,
            )
        else:
            proxy_value = self._semantic_octant_proxy(
                inp,
                phase_normalized=(
                    self.c8_grad_mode == "semantic_phase_ste"
                ),
            )
        value = proxy_value + (hard_value - proxy_value).detach()
        return value, hard_index, scores

    def phase_code_bits(self, inp):
        scores = self._phase_scores(inp)
        hard_index = scores.argmax(dim=-1)
        bits = self.c8_gray_bits.to(dtype=scores.dtype)
        hard_bits = bits[hard_index]

        if self.c8_grad_mode in ("semantic_ste", "semantic_phase_ste"):
            signed_r = binary_sign(inp.real, grad_mode="bireal")
            signed_i = binary_sign(inp.imag, grad_mode="bireal")
            sign_r = (signed_r + 1.0) / 2.0
            sign_i = (signed_i + 1.0) / 2.0

            beta = self.c8_beta.to(dtype=scores.dtype)
            dominance_score = inp.real.abs() - inp.imag.abs()
            if self.c8_grad_mode == "semantic_phase_ste":
                epsilon = max(torch.finfo(inp.real.dtype).eps, 1e-6)
                magnitude = torch.sqrt(
                    inp.real.square()
                    + inp.imag.square()
                    + epsilon * epsilon
                )
                dominance_score = dominance_score / magnitude
            dominance = torch.sigmoid(beta * dominance_score)

            semantic_probs = torch.stack(
                (sign_r, sign_i, dominance),
                dim=-1,
            ).unsqueeze(-2)
            state_bits = self.c8_semantic_state_bits.to(
                dtype=scores.dtype
            )
            state_probabilities = torch.where(
                state_bits.bool(),
                semantic_probs,
                1.0 - semantic_probs,
            ).prod(dim=-1)
            soft_bits = state_probabilities @ self.c8_semantic_to_gray.to(
                dtype=scores.dtype
            )
        else:
            beta = self.c8_beta.to(dtype=scores.dtype)
            soft_bits = torch.softmax(scores * beta, dim=-1) @ bits

        qat_bits = soft_bits + (hard_bits - soft_bits).detach()
        signed_bits = qat_bits * 2.0 - 1.0
        return torch.cat(
            [signed_bits[..., bit] for bit in range(3)],
            dim=1,
        )

    def semantic_code_bits(self, inp):
        """Return hard [sign_r, sign_i, dominance] bits with QAT gradients."""
        if self.c8_codebook_mode != "octants":
            raise RuntimeError(
                "semantic_code_bits requires the sign-preserving octants codebook"
            )
        if self.c8_grad_mode not in ("semantic_ste", "semantic_phase_ste"):
            raise RuntimeError(
                "semantic_code_bits requires semantic_ste or semantic_phase_ste"
            )

        proxy_r = binary_sign(inp.real, grad_mode="bireal")
        proxy_i = binary_sign(inp.imag, grad_mode="bireal")
        hard_r = torch.where(inp.real >= 0.0, 1.0, -1.0)
        hard_i = torch.where(inp.imag >= 0.0, 1.0, -1.0)
        signed_r = proxy_r + (hard_r - proxy_r).detach()
        signed_i = proxy_i + (hard_i - proxy_i).detach()

        dominance_score = inp.real.abs() - inp.imag.abs()
        if self.c8_grad_mode == "semantic_phase_ste":
            epsilon = max(torch.finfo(inp.real.dtype).eps, 1e-6)
            magnitude = torch.sqrt(
                inp.real.square() + inp.imag.square() + epsilon * epsilon
            )
            dominance_score = dominance_score / magnitude
        beta = self.c8_beta.to(dtype=inp.real.dtype)
        soft_dominance = torch.sigmoid(beta * dominance_score)
        hard_dominance = (inp.real.abs() >= inp.imag.abs()).to(inp.real.dtype)
        dominance = soft_dominance + (
            hard_dominance - soft_dominance
        ).detach()
        signed_dominance = dominance * 2.0 - 1.0

        return torch.cat((signed_r, signed_i, signed_dominance), dim=1)

    def forward(self, inp):
        value, _, _ = self._hard_soft_values(inp)
        return torch.complex(value[..., 0], value[..., 1])


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


class LUT5AwareComplexQATConv2d(Module):
    """Phase3.5 activation-QAT bridge from a binary conv to a hard LUT5.

    The phase bit selects the dominant complex component. ``phase_strength=0``
    reproduces ``BinaryComplexConv2d`` exactly. At ``phase_strength=1`` a hard
    phase bit maps the dominant component magnitude to 2 and the other to 0,
    so every local complex product is +/-2 and can be represented exactly by
    the phase-conditioned 5-input truth table.
    """

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
            0,
            dilation,
            groups,
            bias,
        )
        self.conv_i = Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            0,
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
        self.lut_inputs = 5

        self.register_buffer("phase_strength", torch.tensor(0.0))
        self.register_buffer("phase_beta", torch.tensor(1.0))

    @torch.no_grad()
    def set_qat_state(self, strength, beta):
        strength = float(strength)
        beta = float(beta)
        if not 0.0 <= strength <= 1.0:
            raise ValueError("phase_strength must be between 0 and 1")
        if beta <= 0.0:
            raise ValueError("phase_beta must be positive")
        self.phase_strength.fill_(strength)
        self.phase_beta.fill_(beta)

    def forward(self, inp, phase_source=None):
        if phase_source is None:
            raise ValueError(
                "LUT5AwareComplexQATConv2d requires the pre-binary complex activation"
            )

        phase_score = phase_source.real.abs() - phase_source.imag.abs()
        phase_soft = torch.sigmoid(self.phase_beta * phase_score)
        phase_hard = (phase_score >= 0).to(phase_soft.dtype)
        # Deployment-exact hard forward with a smooth comparator backward.
        phase_bit = phase_soft + (phase_hard - phase_soft).detach()
        phase_sign = phase_bit * 2.0 - 1.0

        strength = self.phase_strength.to(dtype=inp.real.dtype)
        q_r = inp.real * (1.0 + strength * phase_sign)
        q_i = inp.imag * (1.0 - strength * phase_sign)

        if isinstance(self.padding, tuple):
            pad_h, pad_w = self.padding
        else:
            pad_h = pad_w = self.padding
        if pad_h > 0 or pad_w > 0:
            pad = (pad_w, pad_w, pad_h, pad_h)

            # Interpolate edge padding from Phase2 zero-padding to the LUT5
            # physical low-bit state. At strength=1 this is exactly (0, -2).
            pad_r = strength * (strength - 1.0)
            pad_i = -strength * (1.0 + strength)
            q_r = F.pad(q_r - pad_r, pad, "constant", 0.0) + pad_r
            q_i = F.pad(q_i - pad_i, pad, "constant", 0.0) + pad_i

        weight = torch.complex(self.conv_r.weight, self.conv_i.weight)
        weight = complex_binary_weight(
            weight,
            per_channel=self.per_channel,
            grad_mode=self.weight_grad_mode,
        )
        w_r = weight.real
        w_i = weight.imag
        conv_kwargs = {
            "stride": self.stride,
            "padding": 0,
            "dilation": self.dilation,
            "groups": self.groups,
        }

        real = F.conv2d(q_r, w_r, bias=self.conv_r.bias, **conv_kwargs)
        real = real - F.conv2d(q_i, w_i, bias=self.conv_i.bias, **conv_kwargs)
        imag = F.conv2d(q_i, w_r, bias=self.conv_r.bias, **conv_kwargs)
        imag = imag + F.conv2d(q_r, w_i, bias=self.conv_i.bias, **conv_kwargs)
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
        dilation=1, groups=1, bias=False, per_channel=True, weight_grad_mode="ste",
        lut_inputs=4,
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
        self.lut_inputs = lut_inputs
        if lut_inputs not in (4, 5):
            raise ValueError("lut_inputs must be 4 or 5")
        if lut_inputs == 5:
            # Phase 3 can start from the exact LUT4 operation and continuously
            # introduce the fixed phase-conditioned LUT5 truth table.
            self.register_buffer("phase_mix", torch.tensor(1.0))

        # 硬件累加的极限范围 [0, N]
        self.N = in_channels * kernel_size * kernel_size // groups

        # 必须叫 conv_r 和 conv_i，与 BinaryComplexConv2d 100% 对齐，实现无缝 Load
        self.conv_r = Conv2d(in_channels, out_channels, kernel_size, stride, 0, dilation, groups, bias)
        self.conv_i = Conv2d(in_channels, out_channels, kernel_size, stride, 0, dilation, groups, bias)

    def set_phase_mix(self, value):
        if self.lut_inputs != 5:
            raise ValueError("phase_mix is only available when lut_inputs=5")
        value = float(value)
        if not 0.0 <= value <= 1.0:
            raise ValueError("phase_mix must be between 0 and 1")
        self.phase_mix.fill_(value)

    def forward(self, inp, phase_source=None):
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

        phase_bit = None
        if self.lut_inputs == 5:
            if phase_source is None:
                raise ValueError(
                    "lut_inputs=5 requires the pre-binary complex activation"
                )
            # The sign bits already encode the quadrant. This comparison splits
            # each quadrant into real-dominant and imaginary-dominant sectors.
            phase_score = (
                phase_source.real.abs() - phase_source.imag.abs()
            )
            phase_bit = SignWithSTE.apply(phase_score)
            if self.padding > 0:
                # Phase 4 pads every physical input bit with low logic level.
                phase_bit = F.pad(phase_bit, pad_tuple, "constant", 0.0)

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
            conv_cross = F.conv2d(
                x_r_bin * x_i_bin,
                w_r_bin * w_i_bin,
                stride=self.stride,
                dilation=self.dilation,
                groups=self.groups,
            )
            y_lut4_r = 0.25 * (
                3 * self.N + conv_rr - conv_ii + conv_cross
            )
            y_lut4_i = 0.25 * (
                3 * self.N + conv_ri + conv_ir - conv_cross
            )

        if self.lut_inputs == 4:
            y_lut_r = y_lut4_r
            y_lut_i = y_lut4_i
        else:
            # Let d=+1 when |x_r|>=|x_i| and d=-1 otherwise. For real
            # multiplication A-B, a binary tie A=B is resolved as A*d.
            # For imaginary multiplication C+D, a tie C=-D is resolved
            # as C*d. The polynomial below is the exact 32-entry LUT5.
            xr, xi = x_r_bin.detach(), x_i_bin.detach()
            wr, wi = w_r_bin.detach(), w_i_bin.detach()
            phase_sign = phase_bit * 2.0 - 1.0
            conv_kwargs = {
                "stride": self.stride,
                "dilation": self.dilation,
                "groups": self.groups,
            }
            conv_rr_hw = F.conv2d(xr, wr, **conv_kwargs)
            conv_ii_hw = F.conv2d(xi, wi, **conv_kwargs)
            conv_ri_hw = F.conv2d(xr, wi, **conv_kwargs)
            conv_ir_hw = F.conv2d(xi, wr, **conv_kwargs)
            conv_phase_rr = F.conv2d(phase_sign * xr, wr, **conv_kwargs)
            conv_phase_ii = F.conv2d(phase_sign * xi, wi, **conv_kwargs)
            conv_phase_ri = F.conv2d(phase_sign * xr, wi, **conv_kwargs)
            conv_phase_ir = F.conv2d(phase_sign * xi, wr, **conv_kwargs)
            y_lut5_r = 0.25 * (
                2 * self.N
                + conv_rr_hw
                - conv_ii_hw
                + conv_phase_rr
                + conv_phase_ii
            )
            y_lut5_i = 0.25 * (
                2 * self.N
                + conv_ri_hw
                + conv_ir_hw
                + conv_phase_ri
                - conv_phase_ir
            )
            y_lut_r = torch.lerp(y_lut4_r, y_lut5_r, self.phase_mix)
            y_lut_i = torch.lerp(y_lut4_i, y_lut5_i, self.phase_mix)

        # =====================================================================
        # 5. 零震荡对齐 (Zero-Shock Alignment) & STE 穿透
        # =====================================================================
        # 将 [0, N] 的硬件输出映射回数学期望域，使得 BN 层不会崩溃！
        alignment_alpha = alpha.detach() if self.lut_inputs == 5 else alpha
        y_hw_math_r = (y_lut_r - self.N / 2.0) * 4.0 * alignment_alpha
        y_hw_math_i = (y_lut_i - self.N / 2.0) * 4.0 * alignment_alpha

        # Keep the original BNN surrogate for weights, with an extra STE path
        # through the phase-dominance comparator for LUT5.
        y_r = y_hw_math_r.detach() - y_orig_r.detach() + y_orig_r
        y_i = y_hw_math_i.detach() - y_orig_i.detach() + y_orig_i
        if self.lut_inputs == 5:
            y_r = y_r + y_hw_math_r - y_hw_math_r.detach()
            y_i = y_i + y_hw_math_i - y_hw_math_i.detach()

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
        hard = hard.to(device=logits.device, dtype=logits.dtype)
        hard_ratio = hard.clamp(0.0, 1.0)
        return soft_sample + hard_ratio * (hard_sample - soft_sample)
    if isinstance(hard, float) and 0.0 < hard < 1.0:
        return soft_sample + hard * (hard_sample - soft_sample)
    return hard_sample if hard else soft_sample




def init_complex_lut_weight_(weight, comp_init):
    if comp_init == "complex_independent":
        nn.init.kaiming_uniform_(weight, a=math.sqrt(5))
        return
    if comp_init == "bimodal":
        signs = torch.empty_like(weight).bernoulli_(0.5).mul_(2.0).sub_(1.0)
        noise = torch.randn_like(weight) * math.sqrt(0.1)
        weight.data.copy_(signs + noise)
        return
    raise ValueError("comp_init must be 'complex_independent' or 'bimodal'")


_PURE_MLP_TEMPLATE_CACHE = {}


def _fit_pure_mlp_template(addresses, target_logits, hidden_width):
    """Fit one deterministic MLP template to the analytic C8 operation."""
    cache_key = (
        int(hidden_width),
        tuple(round(float(value), 7) for value in target_logits.reshape(-1)),
    )
    cached = _PURE_MLP_TEMPLATE_CACHE.get(cache_key)
    if cached is not None:
        return {
            key: value.clone() if torch.is_tensor(value) else value
            for key, value in cached.items()
        }

    addresses = addresses.detach().cpu().to(torch.float32)
    target_logits = target_logits.detach().cpu().to(torch.float32)
    if hidden_width >= addresses.shape[0]:
        detector_weight = torch.zeros(
            hidden_width,
            addresses.shape[1],
            dtype=torch.float32,
        )
        detector_bias = torch.zeros(hidden_width, dtype=torch.float32)
        detector_weight[: addresses.shape[0]].copy_(addresses)
        detector_bias[: addresses.shape[0]].fill_(
            -(addresses.shape[1] - 1.0)
        )
        hidden = torch.tanh(
            F.linear(addresses, detector_weight, detector_bias)
        )
        design = torch.cat(
            [hidden, torch.ones(hidden.shape[0], 1)],
            dim=1,
        )
        solution = torch.linalg.lstsq(
            design.to(torch.float64),
            target_logits.to(torch.float64),
        ).solution.to(torch.float32)
        output_weight = solution[:-1].t().contiguous()
        output_bias = solution[-1].contiguous()
        prediction = F.linear(hidden, output_weight, output_bias)
        best = {
            "hidden_weight": detector_weight,
            "hidden_bias": detector_bias,
            "output_weight": output_weight,
            "output_bias": output_bias,
            "mse": float(F.mse_loss(prediction, target_logits).item()),
            "max_abs_error": float(
                (prediction - target_logits).abs().max().item()
            ),
            "hard_mismatches": int(
                ((prediction >= 0) != (target_logits >= 0)).sum().item()
            ),
        }
        _PURE_MLP_TEMPLATE_CACHE[cache_key] = best
        return {
            key: value.clone() if torch.is_tensor(value) else value
            for key, value in best.items()
        }

    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(7)
        hidden_weight = Parameter(
            torch.empty(hidden_width, addresses.shape[1])
        )
        hidden_bias = Parameter(torch.zeros(hidden_width))
        output_weight = Parameter(torch.empty(2, hidden_width))
        output_bias = Parameter(torch.zeros(2))
        nn.init.xavier_uniform_(hidden_weight)
        nn.init.xavier_uniform_(output_weight)
        optimizer = torch.optim.Adam(
            [hidden_weight, hidden_bias, output_weight, output_bias],
            lr=0.03,
        )

        best = None
        best_loss = float("inf")
        for _ in range(1000):
            optimizer.zero_grad(set_to_none=True)
            hidden = torch.tanh(
                F.linear(addresses, hidden_weight, hidden_bias)
            )
            prediction = F.linear(hidden, output_weight, output_bias)
            loss = F.mse_loss(prediction, target_logits)
            loss.backward()
            optimizer.step()

            with torch.no_grad():
                hidden = torch.tanh(
                    F.linear(addresses, hidden_weight, hidden_bias)
                )
                prediction = F.linear(hidden, output_weight, output_bias)
                current_loss = float(
                    F.mse_loss(prediction, target_logits).item()
                )
                if current_loss < best_loss:
                    best_loss = current_loss
                    best = {
                        "hidden_weight": hidden_weight.detach().clone(),
                        "hidden_bias": hidden_bias.detach().clone(),
                        "output_weight": output_weight.detach().clone(),
                        "output_bias": output_bias.detach().clone(),
                        "mse": current_loss,
                        "max_abs_error": float(
                            (prediction - target_logits).abs().max().item()
                        ),
                        "hard_mismatches": int(
                            (
                                (prediction >= 0)
                                != (target_logits >= 0)
                            ).sum().item()
                        ),
                    }

    if best is None:
        raise RuntimeError("Failed to initialize the pure LUT5 MLP")
    _PURE_MLP_TEMPLATE_CACHE[cache_key] = best
    return {
        key: value.clone() if torch.is_tensor(value) else value
        for key, value in best.items()
    }


# =====================================================================
# 3. 终极硬件感知复数 LUT 卷积层 (5-Phase Architecture)
# =====================================================================
class ComplexLUTConv2d(Module):
    """
    Phase 3: LUT-Aware BNN (真值表固定，仅用 STE 训练空间权重)
    Phase 4: Joint Annealing (真值表解锁退火，空间权重继续 STE)
    Phase 5: Hard Binary (物理部署，全二值纯逻辑门运行)

    lut_inputs=4 uses the original physical LUT inputs [x_r, x_i, w_r, w_i]
    and folds the two weight bits into a runtime K=2 LUT. The standard
    lut_inputs=5 normally adds the phase-dominance bit
    abs(x_r)>=abs(x_i). The isolated magnitude_ste mode instead adds a
    learned-threshold magnitude bit and uses a backward-only asymmetric LUT
    shadow so that a duplicated LUT4 initialization does not kill its
    gradient. Phase 3.6 supplies a three-bit C8 activation address; all LUT5
    paths fold the two complex-weight bits into a runtime K=3 LUT.
    """
    def __init__(
        self, in_channels, out_channels, kernel_size=3, stride=1, padding=1,
        groups=1, phase=3, lut_sets=1, lut_allocation="layer", lut_sets_per_channel=1,
        lut_inputs=4, lut_logit_init=2.0, lut_init_mode="binary",
        lut_sign_flip_prob=0.0, comp_init="complex_independent",
        lut5_init_strategy="phase_conditioned", c8_codebook_mode="octants",
        lut_init_tau=1.0, lut_parameterization="direct",
        neural_lut_hidden=8, neural_lut_residual_scale=1.0,
        lut_extra_bit="phase", magnitude_threshold_init=1.0,
        magnitude_bit_beta=2.0, magnitude_shadow_epsilon=0.05,
    ):
        super().__init__()
        if lut_sets < 1:
            raise ValueError("lut_sets must be at least 1")
        if lut_sets_per_channel < 1:
            raise ValueError("lut_sets_per_channel must be at least 1")
        if lut_allocation not in ("layer", "channel"):
            raise ValueError("lut_allocation must be 'layer' or 'channel'")
        if lut_inputs not in (4, 5):
            raise ValueError("lut_inputs must be 4 or 5")
        if lut_extra_bit not in ("phase", "magnitude_ste"):
            raise ValueError(
                "lut_extra_bit must be 'phase' or 'magnitude_ste'"
            )
        if lut_extra_bit == "magnitude_ste" and lut_inputs != 5:
            raise ValueError("magnitude_ste requires lut_inputs=5")
        if lut_extra_bit == "magnitude_ste" and phase != 4:
            raise ValueError("magnitude_ste is an isolated Phase 4 mode")
        if (
            lut_extra_bit == "magnitude_ste"
            and magnitude_threshold_init <= 0.0
        ):
            raise ValueError("magnitude_threshold_init must be positive")
        if magnitude_bit_beta <= 0.0:
            raise ValueError("magnitude_bit_beta must be positive")
        if not 0.0 <= magnitude_shadow_epsilon < 0.5:
            raise ValueError(
                "magnitude_shadow_epsilon must be in [0, 0.5)"
            )
        if lut_init_mode not in ("binary", "raw", "random"):
            raise ValueError("lut_init_mode must be 'binary', 'raw', or 'random'")
        if lut5_init_strategy not in (
            "phase_conditioned",
            "duplicate_lut4",
            "c8_product",
            "semantic_c8_product",
        ):
            raise ValueError(
                "lut5_init_strategy must be 'phase_conditioned', "
                "'duplicate_lut4', 'c8_product', or "
                "'semantic_c8_product'"
            )
        if c8_codebook_mode not in _C8_CODEBOOK_MODES:
            raise ValueError(
                "c8_codebook_mode must be one of {}".format(
                    _C8_CODEBOOK_MODES
                )
            )
        if lut_init_tau <= 0.0:
            raise ValueError("lut_init_tau must be positive")
        if lut_parameterization not in ("direct", "neural", "mlp"):
            raise ValueError(
                "lut_parameterization must be 'direct', 'neural', or 'mlp'"
            )
        if neural_lut_hidden < 1:
            raise ValueError("neural_lut_hidden must be at least 1")
        if neural_lut_residual_scale <= 0.0:
            raise ValueError("neural_lut_residual_scale must be positive")
        if (
            lut_parameterization in ("neural", "mlp")
            and lut5_init_strategy != "c8_product"
        ):
            raise ValueError(
                "neural/MLP parameterization requires c8_product "
                "initialization"
            )
        if lut5_init_strategy in ("c8_product", "semantic_c8_product"):
            if lut_inputs != 5:
                raise ValueError(
                    "C8 product initialization requires lut_inputs=5"
                )
            if lut_init_mode != "raw":
                raise ValueError(
                    "C8 product initialization requires lut_init_mode='raw'"
                )
        if (
            lut5_init_strategy == "semantic_c8_product"
            and c8_codebook_mode != "octants"
        ):
            raise ValueError(
                "semantic_c8_product requires c8_codebook_mode='octants'"
            )
        if (
            lut_extra_bit == "magnitude_ste"
            and lut5_init_strategy != "duplicate_lut4"
        ):
            raise ValueError(
                "magnitude_ste requires duplicate_lut4 initialization"
            )
        if not 0.0 <= lut_sign_flip_prob <= 1.0:
            raise ValueError("lut_sign_flip_prob must be between 0 and 1")
        if comp_init not in ("complex_independent", "bimodal"):
            raise ValueError("comp_init must be 'complex_independent' or 'bimodal'")

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.groups = groups
        self.phase = phase
        self.lut_inputs = lut_inputs
        self.lut_logit_init = lut_logit_init
        self.lut_init_mode = lut_init_mode
        self.lut_sign_flip_prob = lut_sign_flip_prob
        self.comp_init = comp_init
        self.lut5_init_strategy = lut5_init_strategy
        self.c8_codebook_mode = c8_codebook_mode
        self.lut_init_tau = float(lut_init_tau)
        self.lut_parameterization = lut_parameterization
        self.neural_lut_hidden = int(neural_lut_hidden)
        self.neural_lut_residual_scale = float(
            neural_lut_residual_scale
        )
        self.lut_extra_bit = lut_extra_bit
        self.magnitude_threshold_init = float(magnitude_threshold_init)
        self.magnitude_bit_beta = float(magnitude_bit_beta)
        self.magnitude_shadow_epsilon = float(
            magnitude_shadow_epsilon
        )
        self.lut_allocation = lut_allocation
        self.layer_lut_sets = lut_sets
        self.lut_sets_per_channel = lut_sets_per_channel
        self.lut_sets = lut_sets if lut_allocation == "layer" else out_channels * lut_sets_per_channel

        # Weight bits are folded into the truth table, leaving x_r/x_i at
        # runtime for LUT4 and x_r/x_i/phase_dominance at runtime for LUT5.
        self.LUT_K = lut_inputs - 2
        self.runtime_lut_size = 1 << self.LUT_K
        self.physical_lut_size = 1 << lut_inputs
        self.runtime_channels = self.LUT_K * in_channels
        self.lut_num = in_channels * kernel_size * kernel_size // groups

        # ---------------------------------------------------------------------
        # A. 经典复数空间权重 (潜权重 Logits)
        # ---------------------------------------------------------------------
        self.weight_r = Parameter(torch.Tensor(out_channels, in_channels // groups, kernel_size, kernel_size))
        self.weight_i = Parameter(torch.Tensor(out_channels, in_channels // groups, kernel_size, kernel_size))
        init_complex_lut_weight_(self.weight_r, comp_init)
        init_complex_lut_weight_(self.weight_i, comp_init)

        # ---------------------------------------------------------------------
        # B. 可配置分配粒度的 LUT (转换为 Logits)
        # ---------------------------------------------------------------------
        state_idx4 = torch.arange(16)
        x_r_bit = (state_idx4 >> 3) & 1
        x_i_bit = (state_idx4 >> 2) & 1
        w_r_bit = (state_idx4 >> 1) & 1
        w_i_bit = state_idx4 & 1
        x_r_sign = x_r_bit.to(torch.float32) * 2.0 - 1.0
        x_i_sign = x_i_bit.to(torch.float32) * 2.0 - 1.0
        w_r_sign = w_r_bit.to(torch.float32) * 2.0 - 1.0
        w_i_sign = w_i_bit.to(torch.float32) * 2.0 - 1.0

        raw_lut_r4 = x_r_sign * w_r_sign - x_i_sign * w_i_sign
        raw_lut_i4 = x_r_sign * w_i_sign + x_i_sign * w_r_sign
        init_lut_r4 = (raw_lut_r4 >= 0).to(torch.float32)
        init_lut_i4 = (raw_lut_i4 >= 0).to(torch.float32)

        if lut5_init_strategy in ("c8_product", "semantic_c8_product"):
            state_idx = torch.arange(32)
            activation_address = state_idx >> 2
            weight_r = (
                ((state_idx >> 1) & 1).to(torch.float32) * 2.0
            ) - 1.0
            weight_i = (
                (state_idx & 1).to(torch.float32) * 2.0
            ) - 1.0

            if lut5_init_strategy == "semantic_c8_product":
                sign_r = (
                    ((activation_address >> 2) & 1).to(torch.float32) * 2.0
                ) - 1.0
                sign_i = (
                    ((activation_address >> 1) & 1).to(torch.float32) * 2.0
                ) - 1.0
                dominance = (activation_address & 1).to(torch.float32)
                codebook = _c8_unit_codebook("octants")
                high = codebook.abs().amax()
                low = codebook.abs().amin()
                magnitude_r = low + (high - low) * dominance
                magnitude_i = high - (high - low) * dominance
                decoded = torch.stack(
                    (sign_r * magnitude_r, sign_i * magnitude_i),
                    dim=1,
                )
            else:
                phase_index = torch.arange(8, dtype=torch.long)
                gray_address = phase_index ^ (phase_index >> 1)
                address_to_phase = torch.empty(8, dtype=torch.long)
                address_to_phase[gray_address] = phase_index
                decoded = _c8_unit_codebook(c8_codebook_mode)[
                    address_to_phase[activation_address]
                ]
            raw_lut_r = decoded[:, 0] * weight_r - decoded[:, 1] * weight_i
            raw_lut_i = decoded[:, 0] * weight_i + decoded[:, 1] * weight_r

            epsilon = 1e-6
            logit_lut_r = torch.atanh(
                (raw_lut_r / 2.0).clamp(-1.0 + epsilon, 1.0 - epsilon)
            ) / self.lut_init_tau
            logit_lut_i = torch.atanh(
                (raw_lut_i / 2.0).clamp(-1.0 + epsilon, 1.0 - epsilon)
            ) / self.lut_init_tau
        elif lut_init_mode == "random":
            logit_lut_r = torch.where(
                torch.rand(self.physical_lut_size) < 0.5,
                torch.full((self.physical_lut_size,), -lut_logit_init),
                torch.full((self.physical_lut_size,), lut_logit_init),
            )
            logit_lut_i = torch.where(
                torch.rand(self.physical_lut_size) < 0.5,
                torch.full((self.physical_lut_size,), -lut_logit_init),
                torch.full((self.physical_lut_size,), lut_logit_init),
            )
        else:
            if lut_init_mode == "raw":
                # Keep zero-valued complex multiply cases at logit 0. Soft LUT training
                # starts from 0.5 there, while hard mode still uses the existing >=0 tie.
                logit_lut_r4 = raw_lut_r4 * (lut_logit_init / 2.0)
                logit_lut_i4 = raw_lut_i4 * (lut_logit_init / 2.0)
            else:
                logit_lut_r4 = torch.where(init_lut_r4 == 1, lut_logit_init, -lut_logit_init)
                logit_lut_i4 = torch.where(init_lut_i4 == 1, lut_logit_init, -lut_logit_init)

            if lut_inputs == 5 and lut5_init_strategy == "duplicate_lut4":
                state_idx = torch.arange(32)
                # Physical LUT5 address: [x_r, x_i, d, w_r, w_i]. Removing
                # d recovers the LUT4 address and gives identical initial
                # slices. In raw mode, LUT4 tie entries remain exactly zero.
                state_idx4 = ((state_idx & 0b11000) >> 1) | (
                    state_idx & 0b00011
                )
                logit_lut_r = logit_lut_r4[state_idx4]
                logit_lut_i = logit_lut_i4[state_idx4]
            elif lut_inputs == 5:
                state_idx = torch.arange(32)
                x_r = (((state_idx >> 4) & 1).to(torch.float32) * 2.0) - 1.0
                x_i = (((state_idx >> 3) & 1).to(torch.float32) * 2.0) - 1.0
                phase_sign = (
                    ((state_idx >> 2) & 1).to(torch.float32) * 2.0
                ) - 1.0
                w_r = (((state_idx >> 1) & 1).to(torch.float32) * 2.0) - 1.0
                w_i = ((state_idx & 1).to(torch.float32) * 2.0) - 1.0
                real_a = x_r * w_r
                real_b = x_i * w_i
                imag_a = x_r * w_i
                imag_b = x_i * w_r
                raw_phase_r = torch.where(
                    real_a != real_b,
                    real_a - real_b,
                    real_a * phase_sign,
                )
                raw_phase_i = torch.where(
                    imag_a == imag_b,
                    imag_a + imag_b,
                    imag_a * phase_sign,
                )
                if lut_init_mode == "raw":
                    logit_lut_r = raw_phase_r * (lut_logit_init / 2.0)
                    logit_lut_i = raw_phase_i * (lut_logit_init / 2.0)
                else:
                    logit_lut_r = torch.where(
                        raw_phase_r > 0, lut_logit_init, -lut_logit_init
                    )
                    logit_lut_i = torch.where(
                        raw_phase_i > 0, lut_logit_init, -lut_logit_init
                    )
            else:
                logit_lut_r = logit_lut_r4
                logit_lut_i = logit_lut_i4

        logit_lut_r = logit_lut_r.repeat(self.lut_sets, 1).clone()
        logit_lut_i = logit_lut_i.repeat(self.lut_sets, 1).clone()
        unperturbed_lut_r = logit_lut_r.clone()
        unperturbed_lut_i = logit_lut_i.clone()

        if lut_sign_flip_prob > 0.0:
            flip_r = torch.rand_like(logit_lut_r) < lut_sign_flip_prob
            flip_i = torch.rand_like(logit_lut_i) < lut_sign_flip_prob
            logit_lut_r = torch.where(flip_r, -logit_lut_r, logit_lut_r)
            logit_lut_i = torch.where(flip_i, -logit_lut_i, logit_lut_i)

        if self.lut_parameterization == "mlp":
            self.register_parameter("lut_r", None)
            self.register_parameter("lut_i", None)
            self.register_buffer(
                "lut_unperturbed_reference_r",
                None,
                persistent=False,
            )
            self.register_buffer(
                "lut_unperturbed_reference_i",
                None,
                persistent=False,
            )
        else:
            self.lut_r = Parameter(logit_lut_r)
            self.lut_i = Parameter(logit_lut_i)
            self.register_buffer(
                "lut_unperturbed_reference_r",
                unperturbed_lut_r,
                persistent=False,
            )
            self.register_buffer(
                "lut_unperturbed_reference_i",
                unperturbed_lut_i,
                persistent=False,
            )

        if self.lut_extra_bit == "magnitude_ste":
            threshold_raw = math.log(
                math.expm1(self.magnitude_threshold_init)
            )
            self.mag_threshold_raw = Parameter(
                torch.full(
                    (self.in_channels,),
                    threshold_raw,
                    dtype=torch.float32,
                )
            )
        else:
            self.register_parameter("mag_threshold_raw", None)

        self.mlp_init_mse = None
        self.mlp_init_max_abs_error = None
        self.mlp_init_hard_mismatches = None
        if self.lut_parameterization in ("neural", "mlp"):
            state = torch.arange(self.physical_lut_size)
            shifts = torch.arange(
                self.lut_inputs - 1,
                -1,
                -1,
            )
            addresses = (
                ((state.unsqueeze(1) >> shifts.unsqueeze(0)) & 1)
                .to(torch.float32)
                .mul_(2.0)
                .sub_(1.0)
            )
            self.register_buffer(
                "lut_generator_addresses",
                addresses,
                persistent=False,
            )
            if self.lut_parameterization == "mlp":
                template = _fit_pure_mlp_template(
                    addresses,
                    torch.stack(
                        [logit_lut_r[0], logit_lut_i[0]],
                        dim=1,
                    ),
                    self.neural_lut_hidden,
                )
                self.lut_generator_hidden_weight = Parameter(
                    template["hidden_weight"].unsqueeze(0).repeat(
                        self.lut_sets, 1, 1
                    )
                )
                self.lut_generator_hidden_bias = Parameter(
                    template["hidden_bias"].unsqueeze(0).repeat(
                        self.lut_sets, 1
                    )
                )
                self.lut_generator_output_weight = Parameter(
                    template["output_weight"].unsqueeze(0).repeat(
                        self.lut_sets, 1, 1
                    )
                )
                self.lut_generator_output_bias = Parameter(
                    template["output_bias"].unsqueeze(0).repeat(
                        self.lut_sets, 1
                    )
                )
                self.mlp_init_mse = template["mse"]
                self.mlp_init_max_abs_error = template["max_abs_error"]
                self.mlp_init_hard_mismatches = template[
                    "hard_mismatches"
                ]
            else:
                self.lut_generator_hidden_weight = Parameter(
                    torch.empty(
                        self.lut_sets,
                        self.neural_lut_hidden,
                        self.lut_inputs,
                    )
                )
                self.lut_generator_hidden_bias = Parameter(
                    torch.zeros(self.lut_sets, self.neural_lut_hidden)
                )
                self.lut_generator_output_weight = Parameter(
                    torch.zeros(
                        self.lut_sets,
                        2,
                        self.neural_lut_hidden,
                    )
                )
                self.lut_generator_output_bias = Parameter(
                    torch.zeros(self.lut_sets, 2)
                )
                for weight in self.lut_generator_hidden_weight:
                    nn.init.xavier_uniform_(weight)
        else:
            self.register_buffer(
                "lut_generator_addresses",
                None,
                persistent=False,
            )
            self.register_parameter("lut_generator_hidden_weight", None)
            self.register_parameter("lut_generator_hidden_bias", None)
            self.register_parameter("lut_generator_output_weight", None)
            self.register_parameter("lut_generator_output_bias", None)

        # Phase 5 reuses direct LUT parameters as differentiable proposal
        # scores. Residual neural LUTs keep the analytic table fixed and train
        # only the generator; pure MLPs have no LUT parameters at all.
        if (
            self.lut_r is not None
            and (
                self.phase not in (4, 5)
                or self.lut_parameterization == "neural"
            )
        ):
            self.lut_r.requires_grad = False
            self.lut_i.requires_grad = False

        # ---------------------------------------------------------------------
        # C. 预计算硬件寻址连接 (Hardware Offsets) & 状态寄存器
        # ---------------------------------------------------------------------
        conn_c = torch.arange(in_channels).repeat_interleave(kernel_size * kernel_size)
        conn_dy = (torch.arange(kernel_size * kernel_size) // kernel_size).repeat(in_channels)
        conn_dx = (torch.arange(kernel_size * kernel_size) % kernel_size).repeat(in_channels)

        flat_c = torch.empty(self.lut_num * self.LUT_K, dtype=torch.int32)
        flat_dy = torch.empty(self.lut_num * self.LUT_K, dtype=torch.int32)
        flat_dx = torch.empty(self.lut_num * self.LUT_K, dtype=torch.int32)
        for input_idx in range(self.LUT_K):
            flat_c[input_idx::self.LUT_K] = conn_c + input_idx * in_channels
            flat_dy[input_idx::self.LUT_K] = conn_dy
            flat_dx[input_idx::self.LUT_K] = conn_dx

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
        self.register_buffer("lut_search_base_r", torch.zeros_like(logit_lut_r), persistent=False)
        self.register_buffer("lut_search_base_i", torch.zeros_like(logit_lut_i), persistent=False)
        self.register_buffer("lut_search_origin_r", torch.zeros_like(logit_lut_r), persistent=False)
        self.register_buffer("lut_search_origin_i", torch.zeros_like(logit_lut_i), persistent=False)
        self.register_buffer(
            "lut_search_eligible_r",
            torch.ones_like(logit_lut_r, dtype=torch.bool),
            persistent=False,
        )
        self.register_buffer(
            "lut_search_eligible_i",
            torch.ones_like(logit_lut_i, dtype=torch.bool),
            persistent=False,
        )
        self.register_buffer(
            "lut_search_cooldown_r",
            torch.zeros_like(logit_lut_r, dtype=torch.int16),
            persistent=False,
        )
        self.register_buffer(
            "lut_search_cooldown_i",
            torch.zeros_like(logit_lut_i, dtype=torch.int16),
            persistent=False,
        )
        self.lut_flip_search_active = False
        self.lut_flip_score_temperature = 1.0
        self.lut_flip_initial_score = 0.0
        self.lut_search_anchor_slice = None
        self.lut_search_ties_only = False

    def effective_lut_logits(self):
        if self.lut_parameterization == "direct":
            return self.lut_r, self.lut_i

        hidden = torch.tanh(
            torch.einsum(
                "shd,nd->snh",
                self.lut_generator_hidden_weight,
                self.lut_generator_addresses,
            )
            + self.lut_generator_hidden_bias.unsqueeze(1)
        )
        generated = (
            torch.einsum(
                "soh,snh->sno",
                self.lut_generator_output_weight,
                hidden,
            )
            + self.lut_generator_output_bias.unsqueeze(1)
        )
        if self.lut_parameterization == "mlp":
            return generated[:, :, 0], generated[:, :, 1]

        residual = generated * self.neural_lut_residual_scale
        return (
            self.lut_r + residual[:, :, 0],
            self.lut_i + residual[:, :, 1],
        )

    def mlp_operation_values(self):
        if self.lut_parameterization != "mlp":
            raise RuntimeError(
                "mlp_operation_values requires pure MLP parameterization"
            )
        score_r, score_i = self.effective_lut_logits()

        def project(scores):
            soft = (torch.tanh(scores) + 1.0) / 2.0
            hard = (scores >= 0).to(scores.dtype)
            hard_ste = soft + (hard - soft).detach()
            hard_ratio = self.hard.to(
                device=scores.device,
                dtype=scores.dtype,
            ).clamp(0.0, 1.0)
            return soft + hard_ratio * (hard_ste - soft)

        return project(score_r), project(score_i)

    @torch.no_grad()
    def export_hard_lut_tables(self):
        lut_r, lut_i = self.effective_lut_logits()
        return (lut_r >= 0).to(torch.uint8), (lut_i >= 0).to(torch.uint8)


    @torch.no_grad()
    def begin_lut_flip_search(
        self,
        init_flip_prob=0.05,
        score_temperature=0.25,
        anchor_slice=None,
        ties_only=False,
    ):
        if self.lut_parameterization != "direct":
            raise RuntimeError(
                "LUT flip search requires direct table parameterization"
            )
        if self.phase not in (4, 5):
            raise RuntimeError("LUT flip search is only supported in Phase 4 or 5")
        if not 0.0 < init_flip_prob < 0.5:
            raise ValueError("init_flip_prob must be between 0 and 0.5")
        if score_temperature <= 0.0:
            raise ValueError("score_temperature must be positive")
        if anchor_slice is not None:
            if self.phase != 5 or self.lut_inputs != 5:
                raise ValueError(
                    "an anchor slice requires Phase 5 with lut_inputs=5"
                )
            if anchor_slice not in (0, 1):
                raise ValueError("anchor_slice must be 0 or 1")

        self.lut_search_base_r.copy_((self.lut_r >= 0).to(self.lut_r.dtype))
        self.lut_search_base_i.copy_((self.lut_i >= 0).to(self.lut_i.dtype))
        self.lut_search_origin_r.copy_(self.lut_search_base_r)
        self.lut_search_origin_i.copy_(self.lut_search_base_i)
        self.lut_search_eligible_r.fill_(True)
        self.lut_search_eligible_i.fill_(True)
        if anchor_slice is not None:
            state_idx = torch.arange(
                self.physical_lut_size,
                device=self.lut_r.device,
            )
            phase_bit = (state_idx >> 2) & 1
            trainable_slice = phase_bit != int(anchor_slice)
            eligible_r = trainable_slice
            eligible_i = trainable_slice

            if ties_only:
                x_r = (((state_idx >> 4) & 1) * 2 - 1)
                x_i = (((state_idx >> 3) & 1) * 2 - 1)
                w_r = (((state_idx >> 1) & 1) * 2 - 1)
                w_i = ((state_idx & 1) * 2 - 1)
                real_tie = (x_r * w_r) == (x_i * w_i)
                imag_tie = (x_r * w_i) != (x_i * w_r)
                eligible_r = eligible_r & real_tie
                eligible_i = eligible_i & imag_tie

            self.lut_search_eligible_r.copy_(
                eligible_r.unsqueeze(0).expand_as(self.lut_search_eligible_r)
            )
            self.lut_search_eligible_i.copy_(
                eligible_i.unsqueeze(0).expand_as(self.lut_search_eligible_i)
            )

        self.lut_search_cooldown_r.zero_()
        self.lut_search_cooldown_i.zero_()
        self.lut_flip_score_temperature = float(score_temperature)
        self.lut_flip_initial_score = self.lut_flip_score_temperature * math.log(
            init_flip_prob / (1.0 - init_flip_prob)
        )
        self.lut_r.fill_(self.lut_flip_initial_score)
        self.lut_i.fill_(self.lut_flip_initial_score)
        self.hard.zero_()
        self.lut_flip_search_active = True
        self.lut_search_anchor_slice = anchor_slice
        self.lut_search_ties_only = bool(ties_only)

    def lut_flip_probabilities(self):
        if not self.lut_flip_search_active:
            raise RuntimeError("LUT flip search is not active")
        temperature = self.lut_flip_score_temperature
        prob_r = torch.sigmoid(self.lut_r / temperature)
        prob_i = torch.sigmoid(self.lut_i / temperature)
        return (
            prob_r * self.lut_search_eligible_r.to(prob_r.dtype),
            prob_i * self.lut_search_eligible_i.to(prob_i.dtype),
        )

    def lut_flip_sparsity(self):
        prob_r, prob_i = self.lut_flip_probabilities()
        eligible = torch.cat(
            [
                self.lut_search_eligible_r.reshape(-1),
                self.lut_search_eligible_i.reshape(-1),
            ]
        )
        probabilities = torch.cat([prob_r.reshape(-1), prob_i.reshape(-1)])
        if not bool(eligible.any()):
            return probabilities.sum() * 0.0
        return probabilities[eligible].mean()

    @torch.no_grad()
    def advance_lut_flip_cooldown(self):
        self.lut_search_cooldown_r.sub_(1).clamp_min_(0)
        self.lut_search_cooldown_i.sub_(1).clamp_min_(0)

    @torch.no_grad()
    def microcommit_lut_flip_entries(self, real_indices, imag_indices, cooldown_commits=2):
        if not self.lut_flip_search_active:
            raise RuntimeError("LUT flip search is not active")
        if cooldown_commits < 0:
            raise ValueError("cooldown_commits must be non-negative")

        committed = {}
        for name, indices, base, score, cooldown, eligible in (
            (
                "real",
                real_indices,
                self.lut_search_base_r,
                self.lut_r,
                self.lut_search_cooldown_r,
                self.lut_search_eligible_r,
            ),
            (
                "imag",
                imag_indices,
                self.lut_search_base_i,
                self.lut_i,
                self.lut_search_cooldown_i,
                self.lut_search_eligible_i,
            ),
        ):
            indices = torch.as_tensor(indices, dtype=torch.long, device=score.device).reshape(-1)
            committed[name] = indices
            if indices.numel() == 0:
                continue
            if not bool(eligible.reshape(-1)[indices].all()):
                raise ValueError(
                    "attempted to modify a frozen LUT5 anchor entry"
                )
            base_flat = base.reshape(-1)
            score_flat = score.reshape(-1)
            cooldown_flat = cooldown.reshape(-1)
            base_flat[indices] = 1.0 - base_flat[indices]
            score_flat[indices] = self.lut_flip_initial_score
            cooldown_flat[indices] = cooldown_commits
        return committed

    @torch.no_grad()
    def finalize_lut_flip_search(self, logit_abs_value=None):
        if not self.lut_flip_search_active:
            raise RuntimeError("LUT flip search is not active")

        logit_abs_value = (
            self.lut_logit_init
            if logit_abs_value is None
            else logit_abs_value
        )
        logit_abs_value = max(abs(float(logit_abs_value)), 1.0)
        net_real_flips = int((self.lut_search_base_r != self.lut_search_origin_r).sum().item())
        net_imag_flips = int((self.lut_search_base_i != self.lut_search_origin_i).sum().item())
        eligible_real_entries = int(self.lut_search_eligible_r.sum().item())
        eligible_imag_entries = int(self.lut_search_eligible_i.sum().item())
        frozen_real_diff = int(
            (
                (self.lut_search_base_r != self.lut_search_origin_r)
                & ~self.lut_search_eligible_r
            ).sum().item()
        )
        frozen_imag_diff = int(
            (
                (self.lut_search_base_i != self.lut_search_origin_i)
                & ~self.lut_search_eligible_i
            ).sum().item()
        )
        self.lut_r.copy_(
            torch.where(
                self.lut_search_base_r > 0.5,
                logit_abs_value,
                -logit_abs_value,
            )
        )
        self.lut_i.copy_(
            torch.where(
                self.lut_search_base_i > 0.5,
                logit_abs_value,
                -logit_abs_value,
            )
        )
        self.hard.fill_(1.0)
        self.lut_flip_search_active = False
        return {
            "net_real_flips": net_real_flips,
            "net_imag_flips": net_imag_flips,
            "entries": int(self.lut_r.numel() + self.lut_i.numel()),
            "eligible_real_entries": eligible_real_entries,
            "eligible_imag_entries": eligible_imag_entries,
            "frozen_real_diff": frozen_real_diff,
            "frozen_imag_diff": frozen_imag_diff,
            "anchor_slice": self.lut_search_anchor_slice,
            "ties_only": self.lut_search_ties_only,
        }

    @torch.no_grad()
    def commit_lut_flip_search(self, threshold=0.8, logit_abs_value=None):
        if not self.lut_flip_search_active:
            raise RuntimeError("LUT flip search is not active")
        if not 0.5 < threshold < 1.0:
            raise ValueError("threshold must be between 0.5 and 1.0")

        prob_r, prob_i = self.lut_flip_probabilities()
        flip_r = prob_r >= threshold
        flip_i = prob_i >= threshold
        real_indices = torch.nonzero(flip_r.reshape(-1), as_tuple=False).reshape(-1)
        imag_indices = torch.nonzero(flip_i.reshape(-1), as_tuple=False).reshape(-1)
        self.microcommit_lut_flip_entries(real_indices, imag_indices, cooldown_commits=0)
        final_stats = self.finalize_lut_flip_search(logit_abs_value)
        return {
            "real_flips": int(real_indices.numel()),
            "imag_flips": int(imag_indices.numel()),
            "entries": final_stats["entries"],
            "mean_flip_prob": float(torch.cat([prob_r.reshape(-1), prob_i.reshape(-1)]).mean().item()),
            "max_flip_prob": float(torch.maximum(prob_r.max(), prob_i.max()).item()),
        }

    def _make_x_cat(self, inp, phase_source=None, activation_bits=None):
        x_r, x_i = inp.real, inp.imag
        if activation_bits is not None:
            if self.lut_inputs != 5:
                raise ValueError(
                    "explicit activation_bits require lut_inputs=5"
                )
            expected_shape = (
                inp.shape[0],
                self.runtime_channels,
                inp.shape[2],
                inp.shape[3],
            )
            if tuple(activation_bits.shape) != expected_shape:
                raise ValueError(
                    "activation_bits must have shape {}, got {}".format(
                        expected_shape,
                        tuple(activation_bits.shape),
                    )
                )
            return activation_bits
        if self.lut_inputs == 4:
            return torch.cat([x_r, x_i], dim=1)
        if phase_source is None:
            raise ValueError(
                "lut_inputs=5 requires the pre-binary complex activation"
            )
        if self.lut_extra_bit == "magnitude_ste":
            extra_bit = self._magnitude_bit(phase_source)
        else:
            phase_score = (
                phase_source.real.abs() - phase_source.imag.abs()
            )
            extra_bit = SignWithSTE.apply(phase_score)
        phase_sign = extra_bit * 2.0 - 1.0
        if self.phase == 5:
            phase_sign = phase_sign.detach()
        return torch.cat([x_r, x_i, phase_sign], dim=1)

    def _fold_lut_with_weight_probs(self, selected_lut, p00, p01, p10, p11):
        weight_probs = torch.stack([p00, p01, p10, p11], dim=-1)
        grouped = selected_lut.view(self.lut_num, self.out_channels, self.runtime_lut_size, 4)
        return (grouped * weight_probs.unsqueeze(2)).sum(dim=-1).permute(0, 2, 1).contiguous()

    def _fold_lut_int(self, lut_int, base_idx):
        selected = lut_int[self.lut_set_ids].permute(0, 2, 1).contiguous()
        runtime_states = torch.arange(self.runtime_lut_size, dtype=torch.long, device=base_idx.device).view(1, -1, 1)
        gather_idx = runtime_states * 4 + base_idx.unsqueeze(1)
        return torch.gather(selected, dim=1, index=gather_idx).to(torch.float32)

    def physical_mag_threshold(self):
        if self.lut_extra_bit != "magnitude_ste":
            raise RuntimeError(
                "physical_mag_threshold requires magnitude_ste"
            )
        return physical_magnitude_threshold(self.mag_threshold_raw)

    def _magnitude_bit(self, phase_source):
        threshold = self.physical_mag_threshold().view(1, -1, 1, 1)
        score = torch.abs(phase_source) - threshold
        soft_bit = torch.sigmoid(self.magnitude_bit_beta * score)
        hard_bit = (score >= 0.0).to(score.dtype)
        return soft_bit + (hard_bit - soft_bit).detach()

    def _magnitude_shadow_operation(self, operation):
        """Return a same-sign table with asymmetric d slices for grad_x."""
        if self.lut_extra_bit != "magnitude_ste":
            return operation.detach()

        operation = operation.detach()
        state = torch.arange(
            self.physical_lut_size,
            device=operation.device,
        )
        d_mask = (((state >> 2) & 1) == 1).view(1, -1)
        slice0_index = state & ~0b00100
        slice0_values = operation.index_select(1, slice0_index)
        toward_opposite = torch.where(
            slice0_values >= 0.5,
            -torch.ones_like(slice0_values),
            torch.ones_like(slice0_values),
        )
        candidate = (
            operation
            + self.magnitude_shadow_epsilon
            * d_mask.to(operation.dtype)
            * toward_opposite
        ).clamp(0.0, 1.0)
        below_half = 0.5 - torch.finfo(operation.dtype).eps
        candidate = torch.where(
            operation >= 0.5,
            candidate.clamp_min(0.5),
            candidate.clamp_max(below_half),
        )
        return torch.where(d_mask, candidate, operation)

    def forward(self, inp, phase_source=None, activation_bits=None):
        x_cat = self._make_x_cat(
            inp,
            phase_source=phase_source,
            activation_bits=activation_bits,
        )
        B, _, H, W = x_cat.shape

        padded_W = W + 2 * self.padding
        packed_C = (self.runtime_channels + 31) // 32

        # 动态计算绝对内存偏移 (支持任意分辨率输入)
        offsets = (self.flat_dy * padded_W + self.flat_dx) * packed_C + (self.flat_c // 32)

        complex_w = torch.complex(self.weight_r, self.weight_i)
        alpha = binary_scale_weight_complex(complex_w, per_channel=True).view(1, -1, 1, 1)

        # 展平空间权重 Logits
        wr = self.weight_r.view(self.out_channels, self.lut_num).t()
        wi = self.weight_i.view(self.out_channels, self.lut_num).t()

        current_tau = self.tau
        is_hard = self.hard
        effective_lut_r, effective_lut_i = self.effective_lut_logits()

        use_trainable_hard_path = (
            self.phase in (3, 3.6, 4)
            or (self.lut_flip_search_active and self.training)
            or (self.phase == 5 and self.training)
        )
        if use_trainable_hard_path:
            # =================================================================
            # Phase 3/4 核心：异构向前传播 (Heterogeneous Forward Pass)
            # =================================================================
            pr = SignWithSTE.apply(wr)
            pi = SignWithSTE.apply(wi)

            p00 = (1 - pr) * (1 - pi)
            p01 = (1 - pr) * pi
            p10 = pr * (1 - pi)
            p11 = pr * pi

            if self.lut_parameterization == "mlp":
                L_r, L_i = self.mlp_operation_values()
            elif self.lut_flip_search_active:
                flip_prob_r, flip_prob_i = self.lut_flip_probabilities()
                soft_r = self.lut_search_base_r + (1.0 - 2.0 * self.lut_search_base_r) * flip_prob_r
                soft_i = self.lut_search_base_i + (1.0 - 2.0 * self.lut_search_base_i) * flip_prob_i
                # Keep the pretrained hard operation in forward while using the
                # continuous flip probabilities as the surrogate backward path.
                L_r = soft_r + (self.lut_search_base_r - soft_r).detach()
                L_i = soft_i + (self.lut_search_base_i - soft_i).detach()
            else:
                L_r = binary_annealing(
                    effective_lut_r,
                    tau=current_tau,
                    hard=is_hard,
                )
                L_i = binary_annealing(
                    effective_lut_i,
                    tau=current_tau,
                    hard=is_hard,
                )

            selected_r = L_r[self.lut_set_ids]
            selected_i = L_i[self.lut_set_ids]
            w_folded_r = self._fold_lut_with_weight_probs(selected_r, p00, p01, p10, p11)
            w_folded_i = self._fold_lut_with_weight_probs(selected_i, p00, p01, p10, p11)

            use_magnitude_shadow = (
                self.training
                and self.lut_extra_bit == "magnitude_ste"
                and self.magnitude_shadow_epsilon > 0.0
            )
            if use_magnitude_shadow:
                shadow_r = self._magnitude_shadow_operation(L_r)[
                    self.lut_set_ids
                ]
                shadow_i = self._magnitude_shadow_operation(L_i)[
                    self.lut_set_ids
                ]
                shadow_w_folded_r = self._fold_lut_with_weight_probs(
                    shadow_r,
                    p00.detach(),
                    p01.detach(),
                    p10.detach(),
                    p11.detach(),
                )
                shadow_w_folded_i = self._fold_lut_with_weight_probs(
                    shadow_i,
                    p00.detach(),
                    p01.detach(),
                    p10.detach(),
                    p11.detach(),
                )

            # 上一层传递过来的激活是 [-1, 1]，映射为浮点多项式期望的 [0, 1] 概率
            x_prob = (x_cat + 1.0) / 2.0
            float_offsets = (self.flat_dy * padded_W + self.flat_dx) * self.runtime_channels + self.flat_c

            if use_magnitude_shadow:
                out_r = LUTFloatingShadowInputGradFunction.apply(
                    x_prob, w_folded_r, shadow_w_folded_r, float_offsets,
                    self.groups, self.LUT_K, self.kernel_size,
                    self.stride, self.padding,
                )
                out_i = LUTFloatingShadowInputGradFunction.apply(
                    x_prob, w_folded_i, shadow_w_folded_i, float_offsets,
                    self.groups, self.LUT_K, self.kernel_size,
                    self.stride, self.padding,
                )
            else:
                out_r = LUTFloatingConvFunction.apply(
                    x_prob, w_folded_r, float_offsets, self.groups,
                    self.LUT_K, self.kernel_size, self.stride, self.padding,
                )
                out_i = LUTFloatingConvFunction.apply(
                    x_prob, w_folded_i, float_offsets, self.groups,
                    self.LUT_K, self.kernel_size, self.stride, self.padding,
                )

        else:
            # =================================================================
            # Phase 5: Hard Binary (物理部署，彻底转换为位运算)
            # =================================================================
            wr_bit = (wr > 0).long()
            wi_bit = (wi > 0).long()
            base_idx = wr_bit * 2 + wi_bit

            if self.lut_flip_search_active:
                L_r_int = self.lut_search_base_r.long()
                L_i_int = self.lut_search_base_i.long()
            else:
                L_r_int = (effective_lut_r >= 0).long()
                L_i_int = (effective_lut_i >= 0).long()

            w_folded_r = self._fold_lut_int(L_r_int, base_idx)
            w_folded_i = self._fold_lut_int(L_i_int, base_idx)

            out_r = LUTBinaryConvFunction.apply(x_cat, w_folded_r, offsets, self.shifts, self.groups, self.LUT_K, self.kernel_size, self.stride, self.padding)
            out_i = LUTBinaryConvFunction.apply(x_cat, w_folded_i, offsets, self.shifts, self.groups, self.LUT_K, self.kernel_size, self.stride, self.padding)

        # 零震荡对齐，并恢复 phase3 中使用的复数权重缩放因子。
        out_r = (out_r - (self.lut_num / 2.0)) * 4.0 * alpha
        out_i = (out_i - (self.lut_num / 2.0)) * 4.0 * alpha

        return torch.complex(out_r, out_i)



class C8LUTAwareComplexQATConv2d(Module):
    """Analytic Phase-3-style training for a deployable C8 LUT5 operation.

    Forward decodes a hard C8 activation, applies the fixed binary complex
    multiply, truncates each local real/imaginary result with >= 0, and only
    then accumulates. No LUT parameters or LUT backend are used in this phase;
    the fixed operation can be enumerated into LUT5 tables afterwards.
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=3,
        stride=1,
        padding=1,
        groups=1,
        lut_sets=1,
        lut_allocation="layer",
        lut_sets_per_channel=1,
        lut_logit_init=2.0,
        comp_init="complex_independent",
        per_channel=True,
        weight_grad_mode="ste",
        c8_codebook_mode="roots",
    ):
        super().__init__()
        if in_channels % groups != 0 or out_channels % groups != 0:
            raise ValueError(
                "in_channels and out_channels must be divisible by groups"
            )

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.groups = groups
        self.phase = 3.6
        self.lut_inputs = 5
        self.lut_sets = lut_sets
        self.lut_allocation = lut_allocation
        self.lut_sets_per_channel = lut_sets_per_channel
        self.lut_logit_init = lut_logit_init
        self.comp_init = comp_init
        self.per_channel = per_channel
        self.weight_grad_mode = weight_grad_mode
        self.c8_codebook_mode = c8_codebook_mode
        self.dilation = 1

        kernel_h, kernel_w = (
            kernel_size
            if isinstance(kernel_size, tuple)
            else (kernel_size, kernel_size)
        )
        self.N = (in_channels // groups) * kernel_h * kernel_w
        self.conv_r = Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            0,
            self.dilation,
            groups,
            False,
        )
        self.conv_i = Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            0,
            self.dilation,
            groups,
            False,
        )

        # Phase order starts in the south-west and walks clockwise, keeping
        # adjacent physical directions adjacent in the cyclic Gray code.
        codebook = _c8_unit_codebook(c8_codebook_mode)
        phase_indices = torch.arange(8, dtype=torch.long)
        gray_addresses = phase_indices ^ (phase_indices >> 1)
        gray_bits = torch.stack(
            [
                (gray_addresses >> 2) & 1,
                (gray_addresses >> 1) & 1,
                gray_addresses & 1,
            ],
            dim=1,
        ).to(torch.float32)
        diagonal_indices = torch.tensor([0, 2, 4, 6], dtype=torch.long)
        address_to_phase = torch.empty(8, dtype=torch.long)
        address_to_phase[gray_addresses] = phase_indices

        self.register_buffer("c8_codebook", codebook)
        self.register_buffer("c8_gray_bits", gray_bits)
        self.register_buffer("c8_diagonal_indices", diagonal_indices)
        self.register_buffer("c8_address_to_phase", address_to_phase)
        self.register_buffer("c8_progress", torch.tensor(0.0))
        self.register_buffer("c8_beta", torch.tensor(2.0))
        self.register_buffer(
            "c8_max_axis_advantage",
            torch.tensor(1.0 - math.cos(math.pi / 4.0)),
        )

    @torch.no_grad()
    def set_qat_state(self, progress, beta):
        progress = float(progress)
        beta = float(beta)
        if not 0.0 <= progress <= 1.0:
            raise ValueError("C8 progress must be between 0 and 1")
        if beta <= 0.0:
            raise ValueError("C8 beta must be positive")
        self.c8_progress.fill_(progress)
        self.c8_beta.fill_(beta)

    def _phase_scores(self, phase_source):
        magnitude = torch.sqrt(
            phase_source.real.square() + phase_source.imag.square()
        ).clamp_min(1e-6)
        unit_r = phase_source.real / magnitude
        unit_i = phase_source.imag / magnitude
        codebook = self.c8_codebook.to(dtype=unit_r.dtype)
        return (
            unit_r.unsqueeze(-1) * codebook[:, 0]
            + unit_i.unsqueeze(-1) * codebook[:, 1]
        )

    def _c4_phase_indices(self, phase_source):
        positive_r = phase_source.real >= 0
        positive_i = phase_source.imag >= 0
        return torch.where(
            positive_r,
            torch.where(positive_i, 4, 6),
            torch.where(positive_i, 2, 0),
        )

    def _hard_phase_indices_from_scores(self, phase_source, scores):
        if self.c8_codebook_mode == "octants":
            return scores.argmax(dim=-1)

        c4_index = self._c4_phase_indices(phase_source)
        c4_score = torch.gather(
            scores,
            -1,
            c4_index.unsqueeze(-1),
        ).squeeze(-1)
        c8_score, c8_index = scores.max(dim=-1)
        advantage = c8_score - c4_score
        threshold = (
            (1.0 - self.c8_progress.to(dtype=scores.dtype))
            * self.c8_max_axis_advantage.to(dtype=scores.dtype)
        )
        use_new_axis = (
            (c8_index.remainder(2) == 1)
            & (advantage > threshold)
        )
        return torch.where(use_new_axis, c8_index, c4_index)

    def hard_phase_indices(self, phase_source):
        scores = self._phase_scores(phase_source)
        return self._hard_phase_indices_from_scores(phase_source, scores)

    def phase_code_bits(self, phase_source):
        scores = self._phase_scores(phase_source)
        hard_index = self._hard_phase_indices_from_scores(phase_source, scores)
        hard_bits = self.c8_gray_bits[hard_index].to(dtype=scores.dtype)

        beta = self.c8_beta.to(dtype=scores.dtype)
        all_probs = torch.softmax(scores * beta, dim=-1)
        soft_c8 = all_probs @ self.c8_gray_bits.to(dtype=scores.dtype)

        if self.c8_codebook_mode == "octants":
            bits = soft_c8 + (hard_bits - soft_c8).detach()
            signed_bits = bits * 2.0 - 1.0
            return torch.cat(
                [signed_bits[..., bit] for bit in range(3)],
                dim=1,
            )

        diagonal_scores = scores.index_select(
            -1,
            self.c8_diagonal_indices,
        )
        diagonal_probs = torch.softmax(diagonal_scores * beta, dim=-1)
        diagonal_bits = self.c8_gray_bits.index_select(
            0,
            self.c8_diagonal_indices,
        ).to(dtype=scores.dtype)
        soft_c4 = diagonal_probs @ diagonal_bits

        progress = self.c8_progress.to(dtype=scores.dtype)
        soft_bits = torch.lerp(soft_c4, soft_c8, progress)
        bits = soft_bits + (hard_bits - soft_bits).detach()
        signed_bits = bits * 2.0 - 1.0
        return torch.cat(
            [signed_bits[..., bit] for bit in range(3)],
            dim=1,
        )

    def _decoded_activation_from_scores(self, scores, hard_index):
        codebook = self.c8_codebook.to(dtype=scores.dtype)
        hard_decoded = codebook[hard_index]

        beta = self.c8_beta.to(dtype=scores.dtype)
        all_probs = torch.softmax(scores * beta, dim=-1)
        soft_c8 = all_probs @ codebook

        if self.c8_codebook_mode == "octants":
            return soft_c8 + (hard_decoded - soft_c8).detach()

        diagonal_scores = scores.index_select(
            -1,
            self.c8_diagonal_indices,
        )
        diagonal_probs = torch.softmax(diagonal_scores * beta, dim=-1)
        diagonal_codebook = codebook.index_select(
            0,
            self.c8_diagonal_indices,
        )
        soft_c4 = diagonal_probs @ diagonal_codebook

        progress = self.c8_progress.to(dtype=scores.dtype)
        soft_decoded = torch.lerp(soft_c4, soft_c8, progress)
        return soft_decoded + (hard_decoded - soft_decoded).detach()

    def decoded_activation(self, phase_source):
        scores = self._phase_scores(phase_source)
        hard_index = self._hard_phase_indices_from_scores(phase_source, scores)
        decoded = self._decoded_activation_from_scores(scores, hard_index)
        return decoded[..., 0], decoded[..., 1]

    def _padding_tuple(self):
        if isinstance(self.padding, tuple):
            pad_h, pad_w = self.padding
        else:
            pad_h = pad_w = self.padding
        return pad_w, pad_w, pad_h, pad_h

    def _group_major_features(self, features):
        stacked = torch.stack(features, dim=2)
        batch, channels, feature_count, height, width = stacked.shape
        channels_per_group = channels // self.groups
        return (
            stacked.reshape(
                batch,
                self.groups,
                channels_per_group,
                feature_count,
                height,
                width,
            )
            .permute(0, 1, 3, 2, 4, 5)
            .reshape(batch, feature_count * channels, height, width)
        )

    def _local_comparator_features(self, hard_index):
        decoded = self.c8_codebook[hard_index]
        decoded_r = decoded[..., 0]
        decoded_i = decoded[..., 1]
        difference = decoded_r - decoded_i
        total = decoded_r + decoded_i

        # These four comparator maps exactly factor the local complex multiply
        # for the four binary complex-weight states.
        feature_a = (difference >= 0).to(decoded_r.dtype)
        feature_b = (difference <= 0).to(decoded_r.dtype)
        feature_c = (total >= 0).to(decoded_r.dtype)
        feature_d = (total <= 0).to(decoded_r.dtype)
        return self._group_major_features(
            (feature_a, feature_b, feature_c, feature_d)
        )

    def _local_comparator_routes(self, dtype):
        positive_r = self.conv_r.weight >= 0
        positive_i = self.conv_i.weight >= 0
        state_pp = (positive_r & positive_i).to(dtype)
        state_nn = (~positive_r & ~positive_i).to(dtype)
        state_pn = (positive_r & ~positive_i).to(dtype)
        state_np = (~positive_r & positive_i).to(dtype)

        # Feature order is [A, B, C, D].
        route_r = torch.stack(
            (state_pp, state_nn, state_pn, state_np),
            dim=1,
        ).flatten(1, 2)
        route_i = torch.stack(
            (state_np, state_pn, state_pp, state_nn),
            dim=1,
        ).flatten(1, 2)
        return route_r, route_i

    @torch.no_grad()
    def _hard_local_sums_from_indices(self, hard_index):
        if self.padding != 0:
            hard_index = F.pad(
                hard_index,
                self._padding_tuple(),
                "constant",
                0,
            )

        feature_input = self._local_comparator_features(hard_index)
        route_r, route_i = self._local_comparator_routes(feature_input.dtype)

        conv_args = {
            "stride": self.stride,
            "padding": 0,
            "dilation": self.dilation,
            "groups": self.groups,
        }
        return (
            F.conv2d(feature_input, route_r, **conv_args),
            F.conv2d(feature_input, route_i, **conv_args),
        )

    @torch.no_grad()
    def _hard_local_sums_zero_padding_from_indices(self, hard_index):
        feature_input = self._local_comparator_features(hard_index)
        route_r, route_i = self._local_comparator_routes(feature_input.dtype)
        conv_args = {
            "stride": self.stride,
            "padding": self.padding,
            "dilation": self.dilation,
            "groups": self.groups,
        }
        hard_sum_r = F.conv2d(feature_input, route_r, **conv_args)
        hard_sum_i = F.conv2d(feature_input, route_i, **conv_args)

        # A padded location is absent hardware work, not a valid C8 code. The
        # centering term therefore uses the in-bounds products at each output.
        spatial_ones = torch.ones(
            (1, 1, hard_index.shape[-2], hard_index.shape[-1]),
            dtype=feature_input.dtype,
            device=feature_input.device,
        )
        kernel_ones = torch.ones(
            (1, 1, *self.conv_r.weight.shape[-2:]),
            dtype=feature_input.dtype,
            device=feature_input.device,
        )
        valid_local_count = F.conv2d(
            spatial_ones,
            kernel_ones,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
        ) * (self.in_channels // self.groups)
        return hard_sum_r, hard_sum_i, valid_local_count

    @torch.no_grad()
    def hard_local_sums(self, phase_source):
        return self._hard_local_sums_from_indices(
            self.hard_phase_indices(phase_source)
        )

    def _binary_weight_state(self):
        weight_r = self.conv_r.weight
        weight_i = self.conv_i.weight
        complex_weight = torch.complex(weight_r, weight_i)
        alpha = binary_scale_weight_complex(
            complex_weight,
            per_channel=self.per_channel,
        ).view(1, -1, 1, 1)
        weight_r_bin = binary_sign(
            weight_r,
            grad_mode=self.weight_grad_mode,
        )
        weight_i_bin = binary_sign(
            weight_i,
            grad_mode=self.weight_grad_mode,
        )
        return weight_r_bin, weight_i_bin, alpha

    def _ordinary_binary_output(
        self,
        decoded_r,
        decoded_i,
        weight_r_bin,
        weight_i_bin,
        alpha,
        padding_mode,
    ):
        if padding_mode == "low_code" and self.padding != 0:
            padding = self._padding_tuple()
            decoded_r = F.pad(
                decoded_r,
                padding,
                "constant",
                float(self.c8_codebook[0, 0].item()),
            )
            decoded_i = F.pad(
                decoded_i,
                padding,
                "constant",
                float(self.c8_codebook[0, 1].item()),
            )
            conv_padding = 0
        elif padding_mode == "zero":
            conv_padding = self.padding
        else:
            conv_padding = 0

        conv_args = {
            "stride": self.stride,
            "padding": conv_padding,
            "dilation": self.dilation,
            "groups": self.groups,
        }
        conv_rr = F.conv2d(decoded_r, weight_r_bin, **conv_args)
        conv_ii = F.conv2d(decoded_i, weight_i_bin, **conv_args)
        conv_ri = F.conv2d(decoded_r, weight_i_bin, **conv_args)
        conv_ir = F.conv2d(decoded_i, weight_r_bin, **conv_args)
        return alpha * (conv_rr - conv_ii), alpha * (conv_ri + conv_ir)

    def _forward_decoded(self, decoded_r, decoded_i, hard_index):
        weight_r_bin, weight_i_bin, alpha = self._binary_weight_state()
        surrogate_r, surrogate_i = self._ordinary_binary_output(
            decoded_r,
            decoded_i,
            weight_r_bin,
            weight_i_bin,
            alpha,
            padding_mode="low_code",
        )

        hard_sum_r, hard_sum_i = self._hard_local_sums_from_indices(
            hard_index.detach()
        )
        hardware_r = (hard_sum_r - self.N / 2.0) * 4.0 * alpha
        hardware_i = (hard_sum_i - self.N / 2.0) * 4.0 * alpha
        output_r = hardware_r.detach() - surrogate_r.detach() + surrogate_r
        output_i = hardware_i.detach() - surrogate_i.detach() + surrogate_i
        return torch.complex(output_r, output_i)

    def _forward_local_compare_transition(
        self,
        decoded_r,
        decoded_i,
        hard_index,
        progress,
        padding_mode,
    ):
        weight_r_bin, weight_i_bin, alpha = self._binary_weight_state()
        phase2_r, phase2_i = self._ordinary_binary_output(
            decoded_r,
            decoded_i,
            weight_r_bin,
            weight_i_bin,
            alpha,
            padding_mode="zero",
        )

        if padding_mode == "low_code":
            endpoint_surrogate_r, endpoint_surrogate_i = (
                self._ordinary_binary_output(
                    decoded_r,
                    decoded_i,
                    weight_r_bin,
                    weight_i_bin,
                    alpha,
                    padding_mode="low_code",
                )
            )
            hard_sum_r, hard_sum_i = self._hard_local_sums_from_indices(
                hard_index.detach()
            )
            local_count = self.N
        else:
            endpoint_surrogate_r, endpoint_surrogate_i = phase2_r, phase2_i
            hard_sum_r, hard_sum_i, local_count = (
                self._hard_local_sums_zero_padding_from_indices(
                    hard_index.detach()
                )
            )

        hardware_r = (hard_sum_r - local_count / 2.0) * 4.0 * alpha
        hardware_i = (hard_sum_i - local_count / 2.0) * 4.0 * alpha
        progress = progress.to(dtype=phase2_r.dtype)
        forward_r = torch.lerp(phase2_r, hardware_r, progress)
        forward_i = torch.lerp(phase2_i, hardware_i, progress)
        surrogate_r = torch.lerp(
            phase2_r,
            endpoint_surrogate_r,
            progress,
        )
        surrogate_i = torch.lerp(
            phase2_i,
            endpoint_surrogate_i,
            progress,
        )
        output_r = forward_r.detach() - surrogate_r.detach() + surrogate_r
        output_i = forward_i.detach() - surrogate_i.detach() + surrogate_i
        return torch.complex(output_r, output_i)

    def forward(self, inp, phase_source=None):
        if phase_source is None:
            raise ValueError(
                "C8 LUT-aware QAT requires the pre-binary complex activation"
            )
        if inp.shape != phase_source.shape:
            raise ValueError(
                "binary activation and phase_source shapes must match"
            )

        scores = self._phase_scores(phase_source)
        hard_index = self._hard_phase_indices_from_scores(
            phase_source,
            scores,
        )
        decoded = self._decoded_activation_from_scores(scores, hard_index)
        return self._forward_decoded(
            decoded[..., 0],
            decoded[..., 1],
            hard_index,
        )


class C8LUTAwareComplexBinaryConv2d(C8LUTAwareComplexQATConv2d):
    """Fixed C8 local-comparator convolution used by Phase3.1."""

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=3,
        stride=1,
        padding=1,
        dilation=1,
        groups=1,
        bias=False,
        per_channel=True,
        weight_grad_mode="ste",
        beta=2.0,
        c8_codebook_mode="roots",
        phase3p1_padding_mode="low_code",
    ):
        if dilation != 1:
            raise ValueError("C8 local-comparator convolution requires dilation=1")
        if bias:
            raise ValueError("C8 local-comparator convolution does not support bias")
        if phase3p1_padding_mode not in ("low_code", "zero"):
            raise ValueError(
                "Phase3.1 padding mode must be 'low_code' or 'zero'"
            )
        super().__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            groups=groups,
            per_channel=per_channel,
            weight_grad_mode=weight_grad_mode,
            c8_codebook_mode=c8_codebook_mode,
        )
        self.phase = 3.1
        self.phase3p1_padding_mode = phase3p1_padding_mode
        self.register_buffer(
            "c8_local_compare_progress",
            torch.tensor(1.0),
        )
        self.set_qat_state(progress=1.0, beta=beta)

    @torch.no_grad()
    def set_local_compare_progress(self, progress):
        progress = float(progress)
        if not 0.0 <= progress <= 1.0:
            raise ValueError(
                "Phase3.1 local-compare progress must be between 0 and 1"
            )
        self.c8_local_compare_progress.fill_(progress)

    def forward(self, inp):
        if not torch.is_complex(inp):
            raise TypeError(
                "C8 local-comparator convolution expects a complex tensor"
            )
        hard_index = self.hard_phase_indices(inp.detach())
        progress = float(self.c8_local_compare_progress.item())
        if progress >= 1.0 and self.phase3p1_padding_mode == "low_code":
            return self._forward_decoded(
                inp.real,
                inp.imag,
                hard_index,
            )
        return self._forward_local_compare_transition(
            inp.real,
            inp.imag,
            hard_index,
            self.c8_local_compare_progress,
            self.phase3p1_padding_mode,
        )

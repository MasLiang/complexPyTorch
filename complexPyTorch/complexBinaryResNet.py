#!/usr/bin/env python3
"""Complex CIFAR Bi-Real network with a two-LUTConv Phase 3."""

import torch
import torch.nn as nn

from .complexFunctions import complex_avg_pool2d
from .complexLayers import (
    BinaryComplexActivation,
    BinaryComplexBitActivation,
    BinaryComplexConv2d,
    ComplexAvgPool2d,
    ComplexBatchNorm2d,
    ComplexConv2d,
    ComplexReLU,
    PairLUT4ComplexConv2d,
    TripleLUT6ComplexConv2d,
    TwoLUTComplexConv2d,
    NaiveComplexBatchNorm2d,
)
from .complexResNet import (
    LearnImagBlock,
    _SPECTRAL_SCHEMES,
    apply_spectral_pooling,
)


ACTIVE_PHASES = (1, 2, 3)
_PHASE3_OPERATORS = ("triple_lut6", "pair_lut4", "shared_lut6")
_BN_MODES = ("covariance", "naive", "none")
_SHORTCUT_MODES = ("fp", "option_a")


def _same_padding(kernel_size):
    if isinstance(kernel_size, tuple):
        return tuple(size // 2 for size in kernel_size)
    return kernel_size // 2


def _make_complex_batch_norm(num_features, mode, eps=1e-4):
    if mode == "covariance":
        return ComplexBatchNorm2d(num_features, eps=eps)
    if mode == "naive":
        return NaiveComplexBatchNorm2d(num_features, eps=eps)
    if mode == "none":
        return nn.Identity()
    raise ValueError(
        "Unknown complex BatchNorm mode {!r}; expected one of {}".format(
            mode,
            _BN_MODES,
        )
    )


class ComplexOptionAShortcut(nn.Module):
    """Parameter-free spatial decimation and symmetric channel zero-padding."""

    def __init__(self, in_channels, out_channels, stride):
        super().__init__()
        if out_channels < in_channels:
            raise ValueError("Option-A shortcut cannot reduce channel count")
        self.stride = int(stride)
        channel_padding = int(out_channels) - int(in_channels)
        self.pad_before = channel_padding // 2
        self.pad_after = channel_padding - self.pad_before

    def forward(self, inp):
        output = inp[..., :: self.stride, :: self.stride]
        if not (self.pad_before or self.pad_after):
            return output
        shape = list(output.shape)
        before_shape = shape.copy()
        after_shape = shape.copy()
        before_shape[1] = self.pad_before
        after_shape[1] = self.pad_after
        return torch.cat(
            (
                output.new_zeros(before_shape),
                output,
                output.new_zeros(after_shape),
            ),
            dim=1,
        )


class BiRealComplexResidualBlock(nn.Module):
    """Activation -> operator -> BN -> residual, as in CIFAR Bi-Real."""

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=3,
        stride=1,
        projection=False,
        spectral_pool_scheme="none",
        spectral_pool_gamma=0.0,
        per_channel=True,
        weight_grad_mode="ste",
        act_grad_mode="bireal",
        is_binary=True,
        phase=2,
        post_bn_mode="covariance",
        phase3_operator="pair_lut4",
        pair_lut_parameterization="independent",
        pair_lut_inputs=4,
        pair_lut_encoding="standard",
        dominance_grad_mode="stop",
        dominance_ste_margin=1.0,
        shortcut_mode="fp",
    ):
        super().__init__()
        padding = _same_padding(kernel_size)
        self.projection = bool(projection)
        self.spectral_pool_scheme = spectral_pool_scheme
        self.spectral_pool_gamma = spectral_pool_gamma
        if shortcut_mode not in _SHORTCUT_MODES:
            raise ValueError(
                "Unknown shortcut mode {!r}; expected one of {}".format(
                    shortcut_mode, _SHORTCUT_MODES
                )
            )
        self.shortcut_mode = shortcut_mode

        if is_binary:
            if phase == 3:
                self.act = BinaryComplexBitActivation(
                    grad_mode=act_grad_mode
                )
                operator_class = {
                    "triple_lut6": TripleLUT6ComplexConv2d,
                    "pair_lut4": PairLUT4ComplexConv2d,
                    "shared_lut6": TwoLUTComplexConv2d,
                }.get(phase3_operator)
                if operator_class is None:
                    raise ValueError(
                        "Unknown Phase 3 operator: {}".format(phase3_operator)
                    )
                operator_kwargs = {
                    "stride": stride,
                    "padding": padding,
                }
                if operator_class is PairLUT4ComplexConv2d:
                    operator_kwargs["parameterization"] = (
                        pair_lut_parameterization
                    )
                    operator_kwargs["lut_inputs"] = pair_lut_inputs
                    operator_kwargs["activation_encoding"] = pair_lut_encoding
                    operator_kwargs["dominance_grad_mode"] = dominance_grad_mode
                    operator_kwargs["dominance_ste_margin"] = dominance_ste_margin
                self.conv = operator_class(
                    in_channels,
                    out_channels,
                    kernel_size,
                    **operator_kwargs,
                )
            else:
                self.act = BinaryComplexActivation(
                    grad_mode=act_grad_mode
                )
                self.conv = BinaryComplexConv2d(
                    in_channels,
                    out_channels,
                    kernel_size,
                    stride=stride,
                    padding=padding,
                    bias=False,
                    per_channel=per_channel,
                    weight_grad_mode=weight_grad_mode,
                    weight_proxy_mode="bireal",
                )
        else:
            self.act = ComplexReLU()
            self.conv = ComplexConv2d(
                in_channels,
                out_channels,
                kernel_size,
                stride=stride,
                padding=padding,
                bias=False,
            )

        self.bn_post = _make_complex_batch_norm(
            out_channels,
            post_bn_mode,
            eps=1e-4,
        )

        if projection or stride != 1 or in_channels != out_channels:
            if self.shortcut_mode == "option_a":
                self.proj = ComplexOptionAShortcut(
                    in_channels, out_channels, stride
                )
                return
            projection_layers = []
            if stride != 1:
                projection_layers.append(
                    ComplexAvgPool2d(kernel_size=2, stride=stride)
                )
            projection_layers.extend(
                [
                    ComplexConv2d(
                        in_channels,
                        out_channels,
                        kernel_size=1,
                        stride=1,
                        padding=0,
                        bias=False,
                    ),
                    _make_complex_batch_norm(
                        out_channels,
                        post_bn_mode,
                        eps=1e-4,
                    ),
                ]
            )
            self.proj = nn.Sequential(*projection_layers)
        else:
            self.proj = None

    def forward(self, inp):
        identity = inp
        activation_source = inp
        output = self.act(inp)

        if self.projection and self.spectral_pool_scheme == "proj":
            output = apply_spectral_pooling(
                output,
                self.spectral_pool_gamma,
            )
            activation_source = apply_spectral_pooling(
                activation_source,
                self.spectral_pool_gamma,
            )

        if (
            isinstance(self.conv, PairLUT4ComplexConv2d)
            and self.conv.activation_encoding == "dominance"
        ):
            output = self.conv(
                output,
                dominance_source=activation_source,
            )
        else:
            output = self.conv(output)
        output = self.bn_post(output)

        if self.proj is not None:
            if self.spectral_pool_scheme == "proj":
                identity = apply_spectral_pooling(
                    identity,
                    self.spectral_pool_gamma,
                )
            identity = self.proj(identity)
        return output + identity


class BinaryComplexResNet(nn.Module):
    """Three-stage complex Bi-Real network for CIFAR-sized inputs."""

    def __init__(
        self,
        in_channels=3,
        num_blocks=3,
        start_filters=16,
        num_classes=10,
        spectral_pool_scheme="none",
        spectral_pool_gamma=0.0,
        per_channel=True,
        weight_grad_mode="ste",
        act_grad_mode="bireal",
        binary_stem=False,
        is_sar_input=True,
        is_binary=None,
        phase=2,
        post_bn_mode="covariance",
        phase3_operator="pair_lut4",
        pair_lut_parameterization="independent",
        pair_lut_inputs=4,
        pair_lut_encoding="standard",
        dominance_grad_mode="stop",
        dominance_ste_margin=1.0,
        shortcut_mode="fp",
    ):
        super().__init__()
        if phase not in ACTIVE_PHASES:
            raise ValueError("Only Phase 1, Phase 2, and Phase 3 are active")
        if phase3_operator not in _PHASE3_OPERATORS:
            raise ValueError(
                "Unknown Phase 3 operator {!r}; expected one of {}".format(
                    phase3_operator, _PHASE3_OPERATORS
                )
            )
        if pair_lut_parameterization not in (
            "independent",
            "categorical",
            "categorical_residual",
        ):
            raise ValueError(
                "Unknown PairLUT4 parameterization: {}".format(
                    pair_lut_parameterization
                )
            )
        if pair_lut_inputs not in (4, 6):
            raise ValueError("PairLUT4 inputs must be 4 or 6")
        if pair_lut_encoding not in ("standard", "dominance"):
            raise ValueError(
                "Unknown PairLUT4 activation encoding: {}".format(
                    pair_lut_encoding
                )
            )
        if pair_lut_encoding == "dominance" and (
            pair_lut_inputs != 6
            or pair_lut_parameterization not in (
                "categorical",
                "categorical_residual",
            )
        ):
            raise ValueError(
                "Dominance encoding requires categorical LUT6"
            )
        if spectral_pool_scheme not in _SPECTRAL_SCHEMES:
            raise ValueError(
                "Unknown spectral_pool_scheme: {}".format(
                    spectral_pool_scheme
                )
            )
        if post_bn_mode not in _BN_MODES:
            raise ValueError(
                "Unknown complex BatchNorm mode {!r}; expected one of {}".format(
                    post_bn_mode,
                    _BN_MODES,
                )
            )

        self.actual_blocks_per_stage = int(num_blocks) * 2
        self.spectral_pool_scheme = spectral_pool_scheme
        self.spectral_pool_gamma = spectral_pool_gamma
        self.is_sar_input = bool(is_sar_input)
        self.is_binary = phase >= 2 if is_binary is None else bool(is_binary)
        self.phase = int(phase)
        self.phase3_operator = phase3_operator
        self.pair_lut_parameterization = pair_lut_parameterization
        self.pair_lut_inputs = int(pair_lut_inputs)
        self.pair_lut_encoding = pair_lut_encoding
        self.dominance_grad_mode = dominance_grad_mode
        self.dominance_ste_margin = float(dominance_ste_margin)
        self.post_bn_mode = post_bn_mode
        if shortcut_mode not in _SHORTCUT_MODES:
            raise ValueError(
                "Unknown shortcut mode {!r}; expected one of {}".format(
                    shortcut_mode, _SHORTCUT_MODES
                )
            )
        self.shortcut_mode = shortcut_mode

        if not self.is_sar_input:
            self.learn_imag = LearnImagBlock(
                in_channels,
                in_channels,
                in_channels,
                kernel_size=1,
            )

        if binary_stem:
            self.conv1 = BinaryComplexConv2d(
                in_channels,
                start_filters,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False,
                per_channel=per_channel,
                weight_grad_mode=weight_grad_mode,
                weight_proxy_mode="bireal",
            )
        else:
            self.conv1 = ComplexConv2d(
                in_channels,
                start_filters,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False,
            )
        self.bn1 = _make_complex_batch_norm(
            start_filters,
            post_bn_mode,
            eps=1e-4,
        )

        channels = start_filters
        self.stage2 = self._make_stage(
            channels,
            channels,
            self.actual_blocks_per_stage,
            stride=1,
            per_channel=per_channel,
            weight_grad_mode=weight_grad_mode,
            act_grad_mode=act_grad_mode,
        )

        stride3 = 1 if spectral_pool_scheme == "nodownsample" else 2
        self.stage3 = self._make_stage(
            channels,
            channels * 2,
            self.actual_blocks_per_stage,
            stride=stride3,
            per_channel=per_channel,
            weight_grad_mode=weight_grad_mode,
            act_grad_mode=act_grad_mode,
        )
        channels *= 2

        stride4 = 1 if spectral_pool_scheme == "nodownsample" else 2
        self.stage4 = self._make_stage(
            channels,
            channels * 2,
            self.actual_blocks_per_stage,
            stride=stride4,
            per_channel=per_channel,
            weight_grad_mode=weight_grad_mode,
            act_grad_mode=act_grad_mode,
        )
        channels *= 2

        self.final_channels = channels
        self.fc = nn.Linear(self.final_channels * 2, num_classes)

    def _make_stage(
        self,
        in_channels,
        out_channels,
        num_blocks,
        stride,
        per_channel,
        weight_grad_mode,
        act_grad_mode,
    ):
        common_kwargs = {
            "spectral_pool_scheme": self.spectral_pool_scheme,
            "spectral_pool_gamma": self.spectral_pool_gamma,
            "per_channel": per_channel,
            "weight_grad_mode": weight_grad_mode,
            "act_grad_mode": act_grad_mode,
            "is_binary": self.is_binary,
            "phase": self.phase,
            "post_bn_mode": self.post_bn_mode,
            "phase3_operator": self.phase3_operator,
            "pair_lut_parameterization": self.pair_lut_parameterization,
            "pair_lut_inputs": self.pair_lut_inputs,
            "pair_lut_encoding": self.pair_lut_encoding,
            "dominance_grad_mode": self.dominance_grad_mode,
            "dominance_ste_margin": self.dominance_ste_margin,
            "shortcut_mode": self.shortcut_mode,
        }
        layers = [
            BiRealComplexResidualBlock(
                in_channels,
                out_channels,
                stride=stride,
                projection=(stride != 1 or in_channels != out_channels),
                **common_kwargs,
            )
        ]
        for _ in range(1, num_blocks):
            layers.append(
                BiRealComplexResidualBlock(
                    out_channels,
                    out_channels,
                    stride=1,
                    projection=False,
                    **common_kwargs,
                )
            )
        return nn.ModuleList(layers)

    def _maybe_stage_pool(self, inp, block_index):
        if (
            self.spectral_pool_scheme == "stagemiddle"
            and block_index == self.actual_blocks_per_stage // 2
        ):
            return apply_spectral_pooling(
                inp,
                self.spectral_pool_gamma,
            )
        return inp

    def forward(self, inp):
        if not self.is_sar_input and not torch.is_complex(inp):
            inp = torch.complex(inp, self.learn_imag(inp))

        output = self.bn1(self.conv1(inp))
        for index, block in enumerate(self.stage2):
            output = self._maybe_stage_pool(block(output), index)

        if self.spectral_pool_scheme == "nodownsample":
            output = apply_spectral_pooling(
                output,
                self.spectral_pool_gamma,
            )
        for index, block in enumerate(self.stage3):
            output = self._maybe_stage_pool(block(output), index)

        if self.spectral_pool_scheme == "nodownsample":
            output = apply_spectral_pooling(
                output,
                self.spectral_pool_gamma,
            )
        for index, block in enumerate(self.stage4):
            output = self._maybe_stage_pool(block(output), index)

        if self.spectral_pool_scheme == "nodownsample":
            output = apply_spectral_pooling(
                output,
                self.spectral_pool_gamma,
            )
            output = complex_avg_pool2d(output, kernel_size=32)
        else:
            output = complex_avg_pool2d(output, kernel_size=8)

        output = torch.cat([output.real, output.imag], dim=1)
        return self.fc(output.reshape(output.size(0), -1))


def binary_complex_resnet_cifar10(**kwargs):
    return BinaryComplexResNet(
        num_classes=10,
        is_sar_input=False,
        **kwargs
    )


def binary_complex_resnet_sar(**kwargs):
    return BinaryComplexResNet(
        num_classes=10,
        is_sar_input=True,
        **kwargs
    )

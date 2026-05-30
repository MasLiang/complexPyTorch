#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch
import torch.nn as nn

from .complexLayers import ComplexBatchNorm2d, ComplexConv2d
from .complexFunctions import complex_avg_pool2d, complex_relu


_SPECTRAL_SCHEMES = {"none", "proj", "stagemiddle", "nodownsample"}


def _same_padding(kernel_size):
    if isinstance(kernel_size, tuple):
        return tuple(k // 2 for k in kernel_size)
    return kernel_size // 2


def _spectral_mask_1d(size, topf, device, dtype):
    if topf <= 0 or size < 2 * topf:
        return None
    mask = torch.zeros(size, device=device, dtype=dtype)
    mask[:topf] = 1.0
    mask[-topf:] = 1.0
    return mask


def apply_spectral_pooling(x, gamma):
    if gamma <= 0:
        return x
    _, _, height, width = x.shape
    topf_h = int(gamma * height / 2.0)
    topf_w = int(gamma * width / 2.0)
    mask_h = _spectral_mask_1d(height, topf_h, x.device, x.real.dtype)
    mask_w = _spectral_mask_1d(width, topf_w, x.device, x.real.dtype)
    if mask_h is None or mask_w is None:
        return x
    mask = (mask_h[:, None] * mask_w[None, :]).unsqueeze(0).unsqueeze(0)
    x_freq = torch.fft.fft2(x, norm="ortho")
    x_freq = x_freq * mask
    return torch.fft.ifft2(x_freq, norm="ortho")


class LearnImagBlock(nn.Module):
    def __init__(self, in_channels, mid_channels, out_channels, kernel_size):
        super().__init__()
        padding = _same_padding(kernel_size)
        self.bn1 = nn.BatchNorm2d(in_channels, eps=1e-4)
        self.bn2 = nn.BatchNorm2d(mid_channels, eps=1e-4)
        self.act = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(
            in_channels,
            mid_channels,
            kernel_size,
            padding=padding,
            bias=False,
        )
        self.conv2 = nn.Conv2d(
            mid_channels,
            out_channels,
            kernel_size,
            padding=padding,
            bias=False,
        )

    def forward(self, x):
        x = self.bn1(x)
        x = self.act(x)
        x = self.conv1(x)
        x = self.bn2(x)
        x = self.act(x)
        x = self.conv2(x)
        return x


class ComplexResidualBlock(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=3,
        stride=1,
        projection=False,
        spectral_pool_scheme="none",
        spectral_pool_gamma=0.0,
    ):
        super().__init__()
        padding = _same_padding(kernel_size)
        self.projection = projection
        self.spectral_pool_scheme = spectral_pool_scheme
        self.spectral_pool_gamma = spectral_pool_gamma

        self.bn1 = ComplexBatchNorm2d(in_channels, eps=1e-4)
        self.conv1 = ComplexConv2d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            bias=False,
        )
        self.bn2 = ComplexBatchNorm2d(out_channels, eps=1e-4)
        self.conv2 = ComplexConv2d(
            out_channels,
            out_channels,
            kernel_size,
            stride=1,
            padding=padding,
            bias=False,
        )

        if projection:
            self.proj = ComplexConv2d(
                in_channels,
                out_channels,
                kernel_size=1,
                stride=stride,
                padding=0,
                bias=False,
            )
        else:
            self.proj = None

    def forward(self, x):
        identity = x
        out = self.bn1(x)
        out = complex_relu(out)
        if self.projection and self.spectral_pool_scheme == "proj":
            out = apply_spectral_pooling(out, self.spectral_pool_gamma)
        out = self.conv1(out)
        out = self.bn2(out)
        out = complex_relu(out)
        out = self.conv2(out)

        if self.projection:
            if self.spectral_pool_scheme == "proj":
                identity = apply_spectral_pooling(identity, self.spectral_pool_gamma)
            identity = self.proj(identity)
            out = torch.cat([identity, out], dim=1)
        else:
            out = out + identity
        return out


class ComplexResNet(nn.Module):
    def __init__(
        self,
        in_channels=3,
        num_blocks=3,
        start_filters=16,
        num_classes=10,
        spectral_pool_scheme="none",
        spectral_pool_gamma=0.0,
    ):
        super().__init__()
        if spectral_pool_scheme not in _SPECTRAL_SCHEMES:
            raise ValueError("Unknown spectral_pool_scheme: {}".format(spectral_pool_scheme))
        self.num_blocks = num_blocks
        self.spectral_pool_scheme = spectral_pool_scheme
        self.spectral_pool_gamma = spectral_pool_gamma

        self.learn_imag = LearnImagBlock(
            in_channels,
            in_channels,
            in_channels,
            kernel_size=1,
        )

        self.conv1 = ComplexConv2d(
            in_channels,
            start_filters,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
        )
        self.bn1 = ComplexBatchNorm2d(start_filters, eps=1e-4)

        channels = start_filters
        self.stage2 = nn.ModuleList(
            [
                ComplexResidualBlock(
                    channels,
                    channels,
                    projection=False,
                    spectral_pool_scheme=spectral_pool_scheme,
                    spectral_pool_gamma=spectral_pool_gamma,
                )
                for _ in range(num_blocks)
            ]
        )

        stride3 = 1 if spectral_pool_scheme == "nodownsample" else 2
        self.stage3_proj = ComplexResidualBlock(
            channels,
            channels,
            stride=stride3,
            projection=True,
            spectral_pool_scheme=spectral_pool_scheme,
            spectral_pool_gamma=spectral_pool_gamma,
        )
        channels *= 2
        self.stage3 = nn.ModuleList(
            [
                ComplexResidualBlock(
                    channels,
                    channels,
                    projection=False,
                    spectral_pool_scheme=spectral_pool_scheme,
                    spectral_pool_gamma=spectral_pool_gamma,
                )
                for _ in range(num_blocks - 1)
            ]
        )

        stride4 = 1 if spectral_pool_scheme == "nodownsample" else 2
        self.stage4_proj = ComplexResidualBlock(
            channels,
            channels,
            stride=stride4,
            projection=True,
            spectral_pool_scheme=spectral_pool_scheme,
            spectral_pool_gamma=spectral_pool_gamma,
        )
        channels *= 2
        self.stage4 = nn.ModuleList(
            [
                ComplexResidualBlock(
                    channels,
                    channels,
                    projection=False,
                    spectral_pool_scheme=spectral_pool_scheme,
                    spectral_pool_gamma=spectral_pool_gamma,
                )
                for _ in range(num_blocks - 1)
            ]
        )

        self.final_channels = channels
        self.fc = nn.Linear(self.final_channels * 2, num_classes)

    def _maybe_stage_pool(self, x, block_index):
        if self.spectral_pool_scheme == "stagemiddle" and block_index == self.num_blocks // 2:
            return apply_spectral_pooling(x, self.spectral_pool_gamma)
        return x

    def forward(self, x):
        if not torch.is_complex(x):
            imag = self.learn_imag(x)
            x = torch.complex(x, imag)

        x = self.conv1(x)
        x = self.bn1(x)
        x = complex_relu(x)

        for idx, block in enumerate(self.stage2):
            x = block(x)
            x = self._maybe_stage_pool(x, idx)

        x = self.stage3_proj(x)
        if self.spectral_pool_scheme == "nodownsample":
            x = apply_spectral_pooling(x, self.spectral_pool_gamma)
        for idx, block in enumerate(self.stage3):
            x = block(x)
            x = self._maybe_stage_pool(x, idx)

        x = self.stage4_proj(x)
        if self.spectral_pool_scheme == "nodownsample":
            x = apply_spectral_pooling(x, self.spectral_pool_gamma)
        for idx, block in enumerate(self.stage4):
            x = block(x)
            x = self._maybe_stage_pool(x, idx)

        if self.spectral_pool_scheme == "nodownsample":
            x = apply_spectral_pooling(x, self.spectral_pool_gamma)
            x = complex_avg_pool2d(x, kernel_size=32)
        else:
            x = complex_avg_pool2d(x, kernel_size=8)

        x = torch.cat([x.real, x.imag], dim=1)
        x = x.reshape(x.size(0), -1)
        return self.fc(x)


def complex_resnet_cifar10(**kwargs):
    return ComplexResNet(num_classes=10, **kwargs)

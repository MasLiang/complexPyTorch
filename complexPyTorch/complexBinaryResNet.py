#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch
import torch.nn as nn

from .complexFunctions import complex_avg_pool2d, complex_relu
from .complexLayers import (
    BinaryComplexActivation,
    BinaryComplexConv2d,
    ComplexBatchNorm2d,
    ComplexConv2d,
)
from .complexResNet import LearnImagBlock, apply_spectral_pooling, _SPECTRAL_SCHEMES


def _same_padding(kernel_size):
    if isinstance(kernel_size, tuple):
        return tuple(k // 2 for k in kernel_size)
    return kernel_size // 2


class BiRealComplexResidualBlock(nn.Module):
    """
    标准的 Bi-Real 残差块 (单卷积极简拓扑 + 严格预激活)
    结构: Shortcut + (BN -> Sign -> Conv -> BN)
    """
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
    ):
        super().__init__()
        padding = _same_padding(kernel_size)
        self.projection = projection
        self.spectral_pool_scheme = spectral_pool_scheme
        self.spectral_pool_gamma = spectral_pool_gamma

        # 1. 预激活 BN (用于拉平输入 x 的分布)
        self.bn_pre = ComplexBatchNorm2d(in_channels, eps=1e-4)
        
        # 2. 二值化激活 (Sign)
        self.act = BinaryComplexActivation(grad_mode=act_grad_mode)
        
        # 3. 二值复数卷积
        self.conv = BinaryComplexConv2d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            bias=False,
            per_channel=per_channel,
            weight_grad_mode=weight_grad_mode,
        )
        
        # 4. 卷积后 BN (用于消除二值累加造成的尺度爆炸)
        self.bn_post = ComplexBatchNorm2d(out_channels, eps=1e-4)

        # 5. Projection (Shortcut) 模块
        # 当通道数改变或下采样时，使用全精度 1x1 卷积对齐维度
        if projection or stride != 1 or in_channels != out_channels:
            self.proj = nn.Sequential(
                ComplexConv2d(
                    in_channels,
                    out_channels,
                    kernel_size=1,
                    stride=stride,
                    padding=0,
                    bias=False,
                ),
                ComplexBatchNorm2d(out_channels, eps=1e-4)
            )
        else:
            self.proj = None

    def forward(self, x):
        identity = x

        # 主干分支：严格按照 BN -> Act -> Conv -> BN
        out = self.bn_pre(x)
        out = self.act(out)
        
        if self.projection and self.spectral_pool_scheme == "proj":
            out = apply_spectral_pooling(out, self.spectral_pool_gamma)
            
        out = self.conv(out)
        out = self.bn_post(out)

        # Shortcut 分支处理
        if self.proj is not None:
            if self.spectral_pool_scheme == "proj":
                identity = apply_spectral_pooling(identity, self.spectral_pool_gamma)
            identity = self.proj(identity)

        # 尺度安全的残差相加
        return out + identity


class BinaryComplexResNet(nn.Module):
    def __init__(
        self,
        in_channels=3,
        num_blocks=3, # 这里的 num_blocks 是指双卷积Block的数量。代码内会自动 x2 转换为单卷积Bi-Real Block
        start_filters=16,
        num_classes=10,
        spectral_pool_scheme="none",
        spectral_pool_gamma=0.0,
        per_channel=True,
        weight_grad_mode="ste",
        act_grad_mode="bireal",
        binary_stem=False,
        is_sar_input=True, # 新增标志位：如果是真实SAR复数数据，跳过 LearnImagBlock
    ):
        super().__init__()
        if spectral_pool_scheme not in _SPECTRAL_SCHEMES:
            raise ValueError(f"Unknown spectral_pool_scheme: {spectral_pool_scheme}")
        
        self.num_blocks = num_blocks
        self.actual_blocks_per_stage = num_blocks * 2 # 将标准的 2 层块展开为 2 个单层 Bi-Real 块
        self.spectral_pool_scheme = spectral_pool_scheme
        self.spectral_pool_gamma = spectral_pool_gamma
        self.is_sar_input = is_sar_input

        # 仅针对非复数输入(如光学图像)保留虚部学习模块
        if not self.is_sar_input:
            self.learn_imag = LearnImagBlock(in_channels, in_channels, in_channels, kernel_size=1)

        # Stem (第一层卷积)，通常保持全精度以保留底层特征，也可根据 binary_stem 开启二值化
        if binary_stem:
            self.conv1 = BinaryComplexConv2d(
                in_channels, start_filters, kernel_size=3, stride=1, padding=1, bias=False,
                per_channel=per_channel, weight_grad_mode=weight_grad_mode
            )
        else:
            self.conv1 = ComplexConv2d(in_channels, start_filters, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = ComplexBatchNorm2d(start_filters, eps=1e-4)

        # 构建三个特征提取阶段 (Stage 2, 3, 4)
        channels = start_filters
        
        # Stage 2: 不降采样
        self.stage2 = self._make_stage(
            channels, channels, self.actual_blocks_per_stage, stride=1, 
            per_channel=per_channel, weight_grad_mode=weight_grad_mode, act_grad_mode=act_grad_mode
        )
        
        # Stage 3: 降采样，通道数翻倍
        stride3 = 1 if spectral_pool_scheme == "nodownsample" else 2
        self.stage3 = self._make_stage(
            channels, channels * 2, self.actual_blocks_per_stage, stride=stride3, 
            per_channel=per_channel, weight_grad_mode=weight_grad_mode, act_grad_mode=act_grad_mode
        )
        channels *= 2
        
        # Stage 4: 降采样，通道数翻倍
        stride4 = 1 if spectral_pool_scheme == "nodownsample" else 2
        self.stage4 = self._make_stage(
            channels, channels * 2, self.actual_blocks_per_stage, stride=stride4, 
            per_channel=per_channel, weight_grad_mode=weight_grad_mode, act_grad_mode=act_grad_mode
        )
        channels *= 2

        self.final_channels = channels
        self.fc = nn.Linear(self.final_channels * 2, num_classes) # *2 是因为最后会将实部和虚部 concat 起来

    def _make_stage(self, in_channels, out_channels, num_blocks, stride, per_channel, weight_grad_mode, act_grad_mode):
        """辅助函数：构建单个 Stage"""
        layers = []
        # Stage 的第一个 Block 负责处理下采样和维度匹配
        layers.append(
            BiRealComplexResidualBlock(
                in_channels, out_channels, stride=stride, projection=True,
                spectral_pool_scheme=self.spectral_pool_scheme, spectral_pool_gamma=self.spectral_pool_gamma,
                per_channel=per_channel, weight_grad_mode=weight_grad_mode, act_grad_mode=act_grad_mode
            )
        )
        # Stage 的后续 Blocks 保持维度不变
        for _ in range(1, num_blocks):
            layers.append(
                BiRealComplexResidualBlock(
                    out_channels, out_channels, stride=1, projection=False,
                    spectral_pool_scheme=self.spectral_pool_scheme, spectral_pool_gamma=self.spectral_pool_gamma,
                    per_channel=per_channel, weight_grad_mode=weight_grad_mode, act_grad_mode=act_grad_mode
                )
            )
        return nn.ModuleList(layers)

    def _maybe_stage_pool(self, x, block_index):
        # 配合展开后的实际 blocks 数量调整中心点的 pooling
        if self.spectral_pool_scheme == "stagemiddle" and block_index == self.actual_blocks_per_stage // 2:
            return apply_spectral_pooling(x, self.spectral_pool_gamma)
        return x

    def forward(self, x):
        if not self.is_sar_input and not torch.is_complex(x):
            imag = self.learn_imag(x)
            x = torch.complex(x, imag)

        # Stem 前向传播
        x = self.conv1(x)
        x = self.bn1(x)

        # Stage 2
        for idx, block in enumerate(self.stage2):
            x = block(x)
            x = self._maybe_stage_pool(x, idx)

        # Stage 3
        if self.spectral_pool_scheme == "nodownsample":
            x = apply_spectral_pooling(x, self.spectral_pool_gamma)
        for idx, block in enumerate(self.stage3):
            x = block(x)
            x = self._maybe_stage_pool(x, idx)

        # Stage 4
        if self.spectral_pool_scheme == "nodownsample":
            x = apply_spectral_pooling(x, self.spectral_pool_gamma)
        for idx, block in enumerate(self.stage4):
            x = block(x)
            x = self._maybe_stage_pool(x, idx)

        # 全局池化
        if self.spectral_pool_scheme == "nodownsample":
            x = apply_spectral_pooling(x, self.spectral_pool_gamma)
            x = complex_avg_pool2d(x, kernel_size=32)
        else:
            x = complex_avg_pool2d(x, kernel_size=8) # 注意：这里的 kernel_size 必须根据你输入图像的实际尺寸调整！

        # 展平与分类
        x = torch.cat([x.real, x.imag], dim=1)
        x = x.reshape(x.size(0), -1)
        return self.fc(x)


def binary_complex_resnet_cifar10(**kwargs):
    # 如果你用于 SAR 数据测试，可以在初始化时传入 is_sar_input=True
    return BinaryComplexResNet(num_classes=10, is_sar_input=False, **kwargs)

def binary_complex_resnet_sar(**kwargs):
    # 专为复数 SAR 数据预置的接口
    return BinaryComplexResNet(num_classes=10, is_sar_input=True, **kwargs)
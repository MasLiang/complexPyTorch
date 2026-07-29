#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch
import torch.nn as nn

from .complexFunctions import complex_avg_pool2d, complex_relu
from .complexLayers import (
    BinaryComplexActivation,
    BinaryComplexConv2d,
    C8ComplexActivation,
    C8LUTAwareComplexBinaryConv2d,
    C8LUTAwareComplexQATConv2d,
    LUT5AwareComplexQATConv2d,
    LUTAwareComplexBinaryConv2d,
    ComplexLUTConv2d,
    ComplexBatchNorm2d,
    ComplexConv2d,
    ComplexReLU,
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
        is_binary=True,
        phase=2,
        lut_sets=1,
        lut_allocation="layer",
        lut_sets_per_channel=1,
        lut_inputs=4,
        lut_logit_init=2.0,
        lut_init_mode="binary",
        lut_sign_flip_prob=0.0,
        comp_init="complex_independent",
        c8_beta=2.0,
        c8_codebook="roots",
        c8_grad_mode="softmax",
        phase3p1_padding_mode="low_code",
        phase2p1_mode="c8",
        phase3p1_mode="fixed",
        lut_init_tau=1.0,
        neural_lut_hidden=8,
        neural_lut_residual_scale=1.0,
        lut_extra_bit="phase",
        magnitude_threshold_init=1.0,
        magnitude_bit_beta=2.0,
        magnitude_shadow_epsilon=0.05,
    ):
        super().__init__()
        padding = _same_padding(kernel_size)
        self.projection = projection
        self.spectral_pool_scheme = spectral_pool_scheme
        self.spectral_pool_gamma = spectral_pool_gamma
        self.phase = phase
        self.lut_inputs = lut_inputs
        self.lut_logit_init = lut_logit_init
        self.lut_init_mode = lut_init_mode
        self.lut_sign_flip_prob = lut_sign_flip_prob
        self.comp_init = comp_init
        self.c8_beta = c8_beta
        self.c8_codebook = c8_codebook
        self.c8_grad_mode = c8_grad_mode
        self.phase3p1_padding_mode = phase3p1_padding_mode
        self.phase2p1_mode = phase2p1_mode
        self.phase3p1_mode = phase3p1_mode
        self.lut_init_tau = lut_init_tau
        self.neural_lut_hidden = neural_lut_hidden
        self.neural_lut_residual_scale = neural_lut_residual_scale
        self.lut_extra_bit = lut_extra_bit
        self.magnitude_threshold_init = magnitude_threshold_init
        self.magnitude_bit_beta = magnitude_bit_beta
        self.magnitude_shadow_epsilon = magnitude_shadow_epsilon
        self.learned_lut5_phase2p1 = (
            is_binary
            and phase == 2.1
            and phase2p1_mode == "learned_lut5"
        )
        self.learned_lut5_phase3p1 = (
            is_binary
            and phase == 3.1
            and phase3p1_mode in (
                "learned_lut5",
                "semantic_lut5",
                "neural_lut5",
                "pure_mlp",
            )
        )
        self.semantic_lut5_phase3p1 = (
            is_binary and phase == 3.1 and phase3p1_mode == "semantic_lut5"
        )
        self.neural_lut5_phase3p1 = (
            is_binary
            and phase == 3.1
            and phase3p1_mode == "neural_lut5"
        )
        self.pure_mlp_phase3p1 = (
            is_binary
            and phase == 3.1
            and phase3p1_mode == "pure_mlp"
        )
        self.uses_lut5 = (
            is_binary
            and lut_inputs == 5
            and (
                phase in (3, 3.5, 3.6, 4, 5)
                or self.learned_lut5_phase2p1
                or self.learned_lut5_phase3p1
            )
        )

        # 1. 预激活 BN (用于拉平输入 x 的分布)
        self.bn_pre = ComplexBatchNorm2d(in_channels, eps=1e-4)
        
        # 2. Quantized activation and phase-specific convolution.
        if is_binary:
            if phase in (2.1, 3.1) and not self.learned_lut5_phase2p1:
                self.act = C8ComplexActivation(
                    beta=c8_beta,
                    codebook_mode=c8_codebook,
                    grad_mode=c8_grad_mode,
                )
            else:
                self.act = BinaryComplexActivation(grad_mode=act_grad_mode)

            if (
                phase in (4, 5)
                or self.learned_lut5_phase2p1
                or self.learned_lut5_phase3p1
            ):
                self.conv = ComplexLUTConv2d(
                    in_channels, out_channels, kernel_size, stride=stride, padding=padding,
                    phase=(
                        4
                        if (
                            self.learned_lut5_phase2p1
                            or self.learned_lut5_phase3p1
                        )
                        else phase
                    ),
                    lut_sets=lut_sets, lut_allocation=lut_allocation,
                    lut_sets_per_channel=lut_sets_per_channel, lut_inputs=lut_inputs,
                    lut_logit_init=lut_logit_init, lut_init_mode=lut_init_mode,
                    lut_sign_flip_prob=lut_sign_flip_prob,
                    comp_init=comp_init,
                    lut5_init_strategy=(
                        "duplicate_lut4"
                        if lut_extra_bit == "magnitude_ste"
                        else (
                            "semantic_c8_product"
                            if self.semantic_lut5_phase3p1
                            else (
                                "c8_product"
                                if self.learned_lut5_phase3p1
                                else (
                                    "duplicate_lut4"
                                    if self.learned_lut5_phase2p1
                                    else "phase_conditioned"
                                )
                            )
                        )
                    ),
                    c8_codebook_mode=c8_codebook,
                    lut_init_tau=lut_init_tau,
                    lut_parameterization=(
                        "mlp"
                        if self.pure_mlp_phase3p1
                        else (
                            "neural"
                            if self.neural_lut5_phase3p1
                            else "direct"
                        )
                    ),
                    neural_lut_hidden=neural_lut_hidden,
                    neural_lut_residual_scale=neural_lut_residual_scale,
                    lut_extra_bit=lut_extra_bit,
                    magnitude_threshold_init=magnitude_threshold_init,
                    magnitude_bit_beta=magnitude_bit_beta,
                    magnitude_shadow_epsilon=magnitude_shadow_epsilon,
                )
            elif phase == 3.6:
                self.conv = C8LUTAwareComplexQATConv2d(
                    in_channels, out_channels, kernel_size, stride=stride,
                    padding=padding, lut_sets=lut_sets,
                    lut_allocation=lut_allocation,
                    lut_sets_per_channel=lut_sets_per_channel,
                    lut_logit_init=lut_logit_init,
                    comp_init=comp_init,
                    per_channel=per_channel,
                    weight_grad_mode=weight_grad_mode,
                )
            elif phase == 3.5:
                self.conv = LUT5AwareComplexQATConv2d(
                    in_channels, out_channels, kernel_size, stride=stride,
                    padding=padding, bias=False, per_channel=per_channel,
                    weight_grad_mode=weight_grad_mode,
                )
            elif phase == 3.1:
                self.conv = C8LUTAwareComplexBinaryConv2d(
                    in_channels, out_channels, kernel_size, stride=stride,
                    padding=padding, bias=False, per_channel=per_channel,
                    weight_grad_mode=weight_grad_mode, beta=c8_beta,
                    c8_codebook_mode=c8_codebook,
                    phase3p1_padding_mode=phase3p1_padding_mode,
                )
            elif phase == 3:
                self.conv = LUTAwareComplexBinaryConv2d(
                    in_channels, out_channels, kernel_size, stride=stride, padding=padding,
                    bias=False, per_channel=per_channel, weight_grad_mode=weight_grad_mode,
                    lut_inputs=lut_inputs,
                )
            else:
                self.conv = BinaryComplexConv2d(
                    in_channels, out_channels, kernel_size, stride=stride, padding=padding,
                    bias=False, per_channel=per_channel, weight_grad_mode=weight_grad_mode,
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
        phase_source = out if self.uses_lut5 else None
        if self.learned_lut5_phase3p1:
            if self.projection and self.spectral_pool_scheme == "proj":
                phase_source = apply_spectral_pooling(
                    phase_source, self.spectral_pool_gamma
                )
            activation_bits = (
                self.act.semantic_code_bits(phase_source)
                if self.semantic_lut5_phase3p1
                else self.act.phase_code_bits(phase_source)
            )
            out = self.conv(
                phase_source,
                activation_bits=activation_bits,
            )
        else:
            out = self.act(out)
            if self.projection and self.spectral_pool_scheme == "proj":
                out = apply_spectral_pooling(out, self.spectral_pool_gamma)
                if phase_source is not None:
                    phase_source = apply_spectral_pooling(
                        phase_source, self.spectral_pool_gamma
                    )

            if phase_source is not None:
                out = self.conv(out, phase_source=phase_source)
            else:
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
        start_filters=8,
        num_classes=10,
        spectral_pool_scheme="none",
        spectral_pool_gamma=0.0,
        per_channel=True,
        weight_grad_mode="ste",
        act_grad_mode="bireal",
        binary_stem=False,
        is_sar_input=True, # 新增标志位：如果是真实SAR复数数据，跳过 LearnImagBlock
        is_binary=True, # 是否使用二值化卷积和激活，默认为 True；如果为 False，则整个网络退化为全精度复数 ResNet
        phase=2,
        lut_sets=1,
        lut_allocation="layer",
        lut_sets_per_channel=1,
        lut_inputs=4,
        lut_logit_init=2.0,
        lut_init_mode="binary",
        lut_sign_flip_prob=0.0,
        comp_init="complex_independent",
        c8_beta=2.0,
        c8_codebook="roots",
        c8_grad_mode="softmax",
        phase3p1_padding_mode="low_code",
        phase2p1_mode="c8",
        phase3p1_mode="fixed",
        lut_init_tau=1.0,
        neural_lut_hidden=8,
        neural_lut_residual_scale=1.0,
        lut_extra_bit="phase",
        magnitude_threshold_init=1.0,
        magnitude_bit_beta=2.0,
        magnitude_shadow_epsilon=0.05,
    ):
        super().__init__()
        if spectral_pool_scheme not in _SPECTRAL_SCHEMES:
            raise ValueError(f"Unknown spectral_pool_scheme: {spectral_pool_scheme}")
        
        self.num_blocks = num_blocks
        self.actual_blocks_per_stage = num_blocks * 2 # 将标准的 2 层块展开为 2 个单层 Bi-Real 块
        self.spectral_pool_scheme = spectral_pool_scheme
        self.spectral_pool_gamma = spectral_pool_gamma
        self.is_sar_input = is_sar_input
        self.is_binary = is_binary
        self.phase = phase
        self.lut_sets = lut_sets
        self.lut_allocation = lut_allocation
        self.lut_sets_per_channel = lut_sets_per_channel
        self.lut_inputs = lut_inputs
        self.lut_logit_init = lut_logit_init
        self.lut_init_mode = lut_init_mode
        self.lut_sign_flip_prob = lut_sign_flip_prob
        self.comp_init = comp_init
        self.c8_beta = c8_beta
        self.c8_codebook = c8_codebook
        self.c8_grad_mode = c8_grad_mode
        self.phase3p1_padding_mode = phase3p1_padding_mode
        self.phase2p1_mode = phase2p1_mode
        self.phase3p1_mode = phase3p1_mode
        self.lut_init_tau = lut_init_tau
        self.neural_lut_hidden = neural_lut_hidden
        self.neural_lut_residual_scale = neural_lut_residual_scale
        self.lut_extra_bit = lut_extra_bit
        self.magnitude_threshold_init = magnitude_threshold_init
        self.magnitude_bit_beta = magnitude_bit_beta
        self.magnitude_shadow_epsilon = magnitude_shadow_epsilon

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
                per_channel=per_channel, weight_grad_mode=weight_grad_mode, act_grad_mode=act_grad_mode,
                is_binary=self.is_binary, phase=self.phase, lut_sets=self.lut_sets,
                lut_allocation=self.lut_allocation, lut_sets_per_channel=self.lut_sets_per_channel,
                lut_inputs=self.lut_inputs, lut_logit_init=self.lut_logit_init,
                lut_init_mode=self.lut_init_mode,
                lut_sign_flip_prob=self.lut_sign_flip_prob,
                comp_init=self.comp_init,
                c8_beta=self.c8_beta,
                c8_codebook=self.c8_codebook,
                c8_grad_mode=self.c8_grad_mode,
                phase3p1_padding_mode=self.phase3p1_padding_mode,
                phase2p1_mode=self.phase2p1_mode,
                phase3p1_mode=self.phase3p1_mode,
                lut_init_tau=self.lut_init_tau,
                neural_lut_hidden=self.neural_lut_hidden,
                neural_lut_residual_scale=self.neural_lut_residual_scale,
                lut_extra_bit=self.lut_extra_bit,
                magnitude_threshold_init=self.magnitude_threshold_init,
                magnitude_bit_beta=self.magnitude_bit_beta,
                magnitude_shadow_epsilon=self.magnitude_shadow_epsilon,
            )
        )
        # Stage 的后续 Blocks 保持维度不变
        for _ in range(1, num_blocks):
            layers.append(
                BiRealComplexResidualBlock(
                    out_channels, out_channels, stride=1, projection=False,
                    spectral_pool_scheme=self.spectral_pool_scheme, spectral_pool_gamma=self.spectral_pool_gamma,
                    per_channel=per_channel, weight_grad_mode=weight_grad_mode, act_grad_mode=act_grad_mode,
                    is_binary=self.is_binary, phase=self.phase, lut_sets=self.lut_sets,
                    lut_allocation=self.lut_allocation, lut_sets_per_channel=self.lut_sets_per_channel,
                    lut_inputs=self.lut_inputs, lut_logit_init=self.lut_logit_init,
                    lut_init_mode=self.lut_init_mode,
                    lut_sign_flip_prob=self.lut_sign_flip_prob,
                    comp_init=self.comp_init,
                    c8_beta=self.c8_beta,
                    c8_codebook=self.c8_codebook,
                    c8_grad_mode=self.c8_grad_mode,
                    phase3p1_padding_mode=self.phase3p1_padding_mode,
                    phase2p1_mode=self.phase2p1_mode,
                    phase3p1_mode=self.phase3p1_mode,
                    lut_init_tau=self.lut_init_tau,
                    neural_lut_hidden=self.neural_lut_hidden,
                    neural_lut_residual_scale=self.neural_lut_residual_scale,
                    lut_extra_bit=self.lut_extra_bit,
                    magnitude_threshold_init=self.magnitude_threshold_init,
                    magnitude_bit_beta=self.magnitude_bit_beta,
                    magnitude_shadow_epsilon=self.magnitude_shadow_epsilon,
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

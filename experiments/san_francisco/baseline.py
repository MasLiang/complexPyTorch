"""Independent FP complex CNN used by the San Francisco baseline."""

import torch
from torch import nn

from complexPyTorch.complexLayers import (
    ComplexConv2d,
    ComplexReLU,
    NaiveComplexBatchNorm2d,
)


class ComplexConvBlock(nn.Module):
    """FP complex convolution, independent complex normalization, and ReLU."""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = ComplexConv2d(
            in_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            bias=False,
        )
        self.bn = NaiveComplexBatchNorm2d(out_channels)
        self.activation = ComplexReLU()

    def forward(self, inp):
        return self.activation(self.bn(self.conv(inp)))


class SanFranciscoFPComplexCNN(nn.Module):
    """Three-block independent FP complex CNN with a real/imag classifier."""

    def __init__(self, in_channels=6, num_classes=5):
        super().__init__()
        self.blocks = nn.Sequential(
            ComplexConvBlock(in_channels, 32),
            ComplexConvBlock(32, 64),
            ComplexConvBlock(64, 128),
        )
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(256, num_classes)

    def forward_features(self, inp):
        if not torch.is_complex(inp):
            raise TypeError("SanFranciscoFPComplexCNN expects torch.complex input")
        output = self.blocks(inp)
        pooled = torch.complex(
            self.pool(output.real),
            self.pool(output.imag),
        )
        return pooled.flatten(1)

    def forward(self, inp):
        features = self.forward_features(inp)
        return self.classifier(torch.cat((features.real, features.imag), dim=1))

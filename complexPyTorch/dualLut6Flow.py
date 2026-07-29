import math

import torch
import torch.nn.functional as F
from torch import nn

from .complexBinaryResNet import BinaryComplexResNet
from .complexFunctions import binary_sign, complex_binary_weight
from .complexLayers import BinaryComplexConv2d


DUAL_LUT6_PHASE = 2.7


def _inverse_softplus(value):
    value = float(value)
    if value <= 0.0:
        raise ValueError("Softplus target must be positive")
    return math.log(math.expm1(value))


def _positive_sign_ste(value, grad_mode="bireal"):
    hard = torch.where(value >= 0.0, torch.ones_like(value), -torch.ones_like(value))
    proxy = binary_sign(value, grad_mode=grad_mode)
    return hard.detach() + proxy - proxy.detach()


class TwoBitComplexActivation(nn.Module):
    """Per-component 2-bit activation with a hardware-exact ordered code.

    The final four levels are -high, -low, +low, +high. At rho=0, low and
    high are both one, so this activation reproduces the Phase2 binary
    activation except at exact zero. Increasing rho introduces the magnitude
    bit continuously while the forward values remain hardware quantized.
    """

    def __init__(
        self,
        channels,
        threshold_init=0.675,
        high_ratio=3.0,
        beta=4.0,
        grad_mode="bireal",
    ):
        super().__init__()
        if channels <= 0:
            raise ValueError("channels must be positive")
        if threshold_init <= 0.0:
            raise ValueError("threshold_init must be positive")
        if high_ratio <= 1.0:
            raise ValueError("high_ratio must be greater than one")
        if beta <= 0.0:
            raise ValueError("beta must be positive")

        shape = (1, int(channels), 1, 1)
        initial = torch.full(shape, _inverse_softplus(threshold_init))
        self.threshold_unconstrained = nn.Parameter(initial)
        self.grad_mode = grad_mode
        self.register_buffer("rho", torch.tensor(0.0))
        self.register_buffer("beta", torch.tensor(float(beta)))
        self.register_buffer("high_ratio", torch.tensor(float(high_ratio)))

    @property
    def threshold(self):
        return F.softplus(self.threshold_unconstrained)

    @torch.no_grad()
    def set_rho(self, rho):
        self.rho.fill_(min(max(float(rho), 0.0), 1.0))

    @torch.no_grad()
    def set_beta(self, beta):
        beta = float(beta)
        if beta <= 0.0:
            raise ValueError("beta must be positive")
        self.beta.fill_(beta)

    def levels(self, dtype=None, device=None):
        rho = self.rho
        ratio = 1.0 + rho * (self.high_ratio - 1.0)
        normalization = torch.sqrt((1.0 + ratio.square()) / 2.0)
        low = 1.0 / normalization
        high = ratio / normalization
        if dtype is not None or device is not None:
            low = low.to(dtype=dtype, device=device)
            high = high.to(dtype=dtype, device=device)
        return low, high

    def _component(self, value):
        sign = _positive_sign_ste(value, grad_mode=self.grad_mode)
        threshold = self.threshold.to(dtype=value.dtype, device=value.device)
        hard_magnitude = (value.abs() >= threshold).to(value.dtype)
        soft_magnitude = torch.sigmoid(
            self.beta.to(dtype=value.dtype) * (value.abs() - threshold)
        )
        magnitude = (
            hard_magnitude.detach()
            + soft_magnitude
            - soft_magnitude.detach()
        )
        low, high = self.levels(dtype=value.dtype, device=value.device)
        amplitude = low + magnitude * (high - low)
        return sign * amplitude

    def forward(self, inp):
        if not torch.is_complex(inp):
            raise TypeError("TwoBitComplexActivation expects a complex tensor")
        return torch.complex(
            self._component(inp.real),
            self._component(inp.imag),
        )

    def hard_code_bits(self, inp):
        """Return ordered-offset-binary bits for real and imaginary parts."""
        if not torch.is_complex(inp):
            raise TypeError("TwoBitComplexActivation expects a complex tensor")

        threshold = self.threshold.to(dtype=inp.real.dtype, device=inp.device)

        def component_bits(value):
            msb = (value >= 0.0).to(torch.int64)
            magnitude = (value.abs() >= threshold).to(torch.int64)
            lsb = (msb == magnitude).to(torch.int64)
            return torch.stack((msb, lsb), dim=-1)

        return component_bits(inp.real), component_bits(inp.imag)

    def decode_code_bits(self, bits, dtype=torch.float32):
        bits = bits.to(dtype=dtype)
        low, high = self.levels(dtype=dtype, device=bits.device)
        msb = bits[..., 0]
        lsb = bits[..., 1]
        return -high + (low + high) * msb + (high - low) * lsb


class DualLUT6ComplexConv2d(nn.Module):
    """Binary-weight complex convolution with an exact dual-LUT6_2 factorization."""

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=3,
        stride=1,
        padding=0,
        dilation=1,
        groups=1,
        bias=False,
        per_channel=True,
        weight_grad_mode="ste",
        implementation="standard",
    ):
        super().__init__()
        if implementation not in ("standard", "hardware"):
            raise ValueError("implementation must be standard or hardware")
        self.conv_r = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            dilation,
            groups,
            bias,
        )
        self.conv_i = nn.Conv2d(
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
        self.implementation = implementation

    @classmethod
    def from_binary(cls, source):
        if not isinstance(source, BinaryComplexConv2d):
            raise TypeError("source must be BinaryComplexConv2d")
        result = cls(
            source.conv_r.in_channels,
            source.conv_r.out_channels,
            kernel_size=source.conv_r.kernel_size,
            stride=source.stride,
            padding=source.padding,
            dilation=source.dilation,
            groups=source.groups,
            bias=source.conv_r.bias is not None,
            per_channel=source.per_channel,
            weight_grad_mode=source.weight_grad_mode,
        )
        result.conv_r = source.conv_r
        result.conv_i = source.conv_i
        return result

    def set_implementation(self, implementation):
        if implementation not in ("standard", "hardware"):
            raise ValueError("implementation must be standard or hardware")
        self.implementation = implementation

    def _binary_weight(self):
        return complex_binary_weight(
            torch.complex(self.conv_r.weight, self.conv_i.weight),
            per_channel=self.per_channel,
            grad_mode=self.weight_grad_mode,
        )

    def _conv(self, value, weight):
        return F.conv2d(
            value,
            weight,
            bias=None,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
            groups=self.groups,
        )

    def _forward_standard(self, inp, weight):
        real = self._conv(inp.real, weight.real) - self._conv(inp.imag, weight.imag)
        imag = self._conv(inp.imag, weight.real) + self._conv(inp.real, weight.imag)
        return torch.complex(real, imag)

    def _forward_hardware(self, inp, weight):
        # w = (1 + j) q. q is only a sign/swap route for binary complex weights.
        q_real = 0.5 * (weight.real + weight.imag)
        q_imag = 0.5 * (weight.imag - weight.real)
        u_real = self._conv(inp.real, q_real) - self._conv(inp.imag, q_imag)
        u_imag = self._conv(inp.real, q_imag) + self._conv(inp.imag, q_real)
        return torch.complex(u_real - u_imag, u_real + u_imag)

    def forward(self, inp):
        if self.conv_r.bias is not None or self.conv_i.bias is not None:
            raise RuntimeError("DualLUT6ComplexConv2d requires bias=False")
        weight = self._binary_weight()
        if self.implementation == "hardware":
            return self._forward_hardware(inp, weight)
        return self._forward_standard(inp, weight)


class DualLUT6ComplexResNet(BinaryComplexResNet):
    """Phase2-compatible ResNet using 2-bit activations and two LUT6_2 units."""

    def __init__(
        self,
        in_channels=3,
        num_blocks=3,
        start_filters=8,
        num_classes=10,
        spectral_pool_scheme="none",
        spectral_pool_gamma=0.0,
        per_channel=True,
        weight_grad_mode="ste",
        act_grad_mode="bireal",
        binary_stem=False,
        is_sar_input=False,
        threshold_init=0.675,
        high_ratio=3.0,
        magnitude_beta=4.0,
    ):
        super().__init__(
            in_channels=in_channels,
            num_blocks=num_blocks,
            start_filters=start_filters,
            num_classes=num_classes,
            spectral_pool_scheme=spectral_pool_scheme,
            spectral_pool_gamma=spectral_pool_gamma,
            per_channel=per_channel,
            weight_grad_mode=weight_grad_mode,
            act_grad_mode=act_grad_mode,
            binary_stem=binary_stem,
            is_sar_input=is_sar_input,
            is_binary=True,
            phase=2,
        )
        self.phase = DUAL_LUT6_PHASE
        self.two_bit_threshold_init = float(threshold_init)
        self.two_bit_high_ratio = float(high_ratio)
        self.two_bit_magnitude_beta = float(magnitude_beta)

        for stage in (self.stage2, self.stage3, self.stage4):
            for block in stage:
                channels = block.conv.conv_r.in_channels
                block.act = TwoBitComplexActivation(
                    channels,
                    threshold_init=threshold_init,
                    high_ratio=high_ratio,
                    beta=magnitude_beta,
                    grad_mode=act_grad_mode,
                )
                block.conv = DualLUT6ComplexConv2d.from_binary(block.conv)


def iter_two_bit_activations(model):
    for module in model.modules():
        if isinstance(module, TwoBitComplexActivation):
            yield module


def iter_dual_lut6_convs(model):
    for module in model.modules():
        if isinstance(module, DualLUT6ComplexConv2d):
            yield module


@torch.no_grad()
def set_two_bit_rho(model, rho):
    for module in iter_two_bit_activations(model):
        module.set_rho(rho)


def set_dual_lut6_implementation(model, implementation):
    for module in iter_dual_lut6_convs(model):
        module.set_implementation(implementation)


def two_bit_parameter_diagnostics(model):
    thresholds = [
        module.threshold.detach().reshape(-1).cpu()
        for module in iter_two_bit_activations(model)
    ]
    if not thresholds:
        raise RuntimeError("No TwoBitComplexActivation modules found")
    values = torch.cat(thresholds)
    first = next(iter_two_bit_activations(model))
    low, high = first.levels(dtype=torch.float32, device=torch.device("cpu"))
    return {
        "threshold_min": float(values.min().item()),
        "threshold_mean": float(values.mean().item()),
        "threshold_max": float(values.max().item()),
        "level_low": float(low.item()),
        "level_high": float(high.item()),
        "rho": float(first.rho.item()),
    }


class TwoBitOccupancyTracker:
    def __init__(self, model):
        self.counts = {
            "real": torch.zeros(4, dtype=torch.long),
            "imag": torch.zeros(4, dtype=torch.long),
        }
        self.handles = []
        self.entries = 0
        for module in iter_two_bit_activations(model):
            self.handles.append(module.register_forward_hook(self._hook))

    def _hook(self, module, inputs, output):
        del output
        real_bits, imag_bits = module.hard_code_bits(inputs[0])
        for name, bits in (("real", real_bits), ("imag", imag_bits)):
            indices = bits[..., 0] * 2 + bits[..., 1]
            self.counts[name] += torch.bincount(
                indices.detach().reshape(-1).cpu(), minlength=4
            )
            self.entries += indices.numel()

    def finish(self):
        for handle in self.handles:
            handle.remove()
        if self.entries == 0:
            raise RuntimeError("Two-bit occupancy tracker captured no entries")
        result = {}
        for name, counts in self.counts.items():
            total = max(int(counts.sum().item()), 1)
            result[name] = {
                "counts": counts.tolist(),
                "fractions": [float(value) / total for value in counts.tolist()],
            }
        return result


def dual_lut6_local_output_bits(xr_bit, xi_bit, wr_bit, wi_bit):
    bits = tuple(int(value) for value in (xr_bit, xi_bit, wr_bit, wi_bit))
    if any(value not in (0, 1) for value in bits):
        raise ValueError("LUT inputs must be binary")
    xr_bit, xi_bit, wr_bit, wi_bit = bits
    if wr_bit == 1 and wi_bit == 1:
        return xr_bit, xi_bit
    if wr_bit == 1 and wi_bit == 0:
        return xi_bit, 1 - xr_bit
    if wr_bit == 0 and wi_bit == 1:
        return 1 - xi_bit, xr_bit
    return 1 - xr_bit, 1 - xi_bit


def dual_lut6_init():
    """Return one LUT6_2 INIT shared by the high and low bit planes."""
    lower_o5 = 0
    upper_o6 = 0
    for address5 in range(32):
        xr_bit = (address5 >> 0) & 1
        xi_bit = (address5 >> 1) & 1
        wr_bit = (address5 >> 2) & 1
        wi_bit = (address5 >> 3) & 1
        ur_bit, ui_bit = dual_lut6_local_output_bits(
            xr_bit, xi_bit, wr_bit, wi_bit
        )
        lower_o5 |= ur_bit << address5
        upper_o6 |= ui_bit << address5
    return lower_o5 | (upper_o6 << 32)


def evaluate_dual_lut6_init(init_value, xr_bit, xi_bit, wr_bit, wi_bit):
    address = (
        int(xr_bit)
        | (int(xi_bit) << 1)
        | (int(wr_bit) << 2)
        | (int(wi_bit) << 3)
    )
    o5 = (int(init_value) >> address) & 1
    o6 = (int(init_value) >> (32 + address)) & 1
    return o5, o6


def dual_lut6_hardware_spec(high_ratio=3.0):
    normalization = math.sqrt((1.0 + float(high_ratio) ** 2) / 2.0)
    low = 1.0 / normalization
    high = float(high_ratio) / normalization
    init_value = dual_lut6_init()
    return {
        "primitive": "LUT6_2",
        "logical_inputs": 6,
        "local_outputs": 4,
        "lut6_2_units_per_complex_product": 2,
        "input_order": ["xr_plane_bit", "xi_plane_bit", "wr_sign", "wi_sign"],
        "unused_inputs": {"I4": 0, "I5": 1},
        "outputs": {"O5": "u_real_plane_bit", "O6": "u_imag_plane_bit"},
        "same_init_for_high_and_low_planes": True,
        "init_hex": "64'h{:016X}".format(init_value),
        "activation_codebook": {
            "00": -high,
            "01": -low,
            "10": low,
            "11": high,
        },
        "post_accumulation": {
            "real": "sum(u_real) - sum(u_imag)",
            "imag": "sum(u_real) + sum(u_imag)",
        },
        "local_sign_truncation": False,
        "padding": "Convolution padding is skipped; it is not encoded as an activation.",
    }

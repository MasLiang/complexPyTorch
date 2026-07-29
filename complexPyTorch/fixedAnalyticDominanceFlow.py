"""Fixed analytic dominance flow with comparator-aware gradients.

Training never evaluates a LUT. Three hard activation bits are decoded into
an octant-valued complex activation, multiplied by binary complex weights,
and locally compared before accumulation. The fixed Boolean operation is only
enumerated into two LUT5 truth tables at checkpoint/export time.
"""

import math

import torch
import torch.nn.functional as F
from torch import nn

from .complexBinaryResNet import (
    BiRealComplexResidualBlock,
    BinaryComplexResNet,
)
from .complexFunctions import binary_sign
from .complexLayers import (
    C8LUTAwareComplexBinaryConv2d,
    ComplexLUTConv2d,
)


FIXED_ANALYTIC_PHASE = 2.6
FIXED_ANALYTIC_BIT_NAMES = ("sign_r", "sign_i", "dominance")


class FixedDominanceActivation(nn.Module):
    """Emit hard semantic octant bits with differentiable proxies."""

    def __init__(self, beta=2.0, phase_normalized=False):
        super().__init__()
        if beta <= 0.0:
            raise ValueError("dominance beta must be positive")
        self.phase_normalized = bool(phase_normalized)
        self.register_buffer("dominance_beta", torch.tensor(float(beta)))

    def forward(self, inp):
        if not torch.is_complex(inp):
            raise TypeError("FixedDominanceActivation expects a complex tensor")

        proxy_r = binary_sign(inp.real, grad_mode="bireal")
        proxy_i = binary_sign(inp.imag, grad_mode="bireal")
        hard_r = torch.where(inp.real >= 0.0, 1.0, -1.0)
        hard_i = torch.where(inp.imag >= 0.0, 1.0, -1.0)
        signed_r = proxy_r + (hard_r - proxy_r).detach()
        signed_i = proxy_i + (hard_i - proxy_i).detach()

        dominance_score = inp.real.abs() - inp.imag.abs()
        if self.phase_normalized:
            epsilon = max(torch.finfo(inp.real.dtype).eps, 1e-6)
            magnitude = torch.sqrt(
                inp.real.square() + inp.imag.square() + epsilon * epsilon
            )
            dominance_score = dominance_score / magnitude
        beta = self.dominance_beta.to(dtype=inp.real.dtype)
        soft_dominance = torch.sigmoid(beta * dominance_score)
        hard_dominance = (inp.real.abs() >= inp.imag.abs()).to(inp.real.dtype)
        dominance = soft_dominance + (
            hard_dominance - soft_dominance
        ).detach()
        signed_dominance = dominance * 2.0 - 1.0
        return torch.cat((signed_r, signed_i, signed_dominance), dim=1)


class ComparatorAwareDominanceConv2d(C8LUTAwareComplexBinaryConv2d):
    """Fixed 5-to-2 analytic operator with a local-comparator surrogate."""

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=3,
        stride=1,
        padding=1,
        groups=1,
        per_channel=True,
        weight_grad_mode="ste",
        comparator_beta=1.0,
        comparator_scale=1.0,
    ):
        if comparator_beta <= 0.0:
            raise ValueError("comparator beta must be positive")
        if comparator_scale <= 0.0:
            raise ValueError("comparator scale must be positive")
        super().__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            groups=groups,
            bias=False,
            per_channel=per_channel,
            weight_grad_mode=weight_grad_mode,
            beta=1.0,
            c8_codebook_mode="octants",
            phase3p1_padding_mode="zero",
        )
        self.phase = FIXED_ANALYTIC_PHASE
        self.register_buffer(
            "c8_comparator_beta", torch.tensor(float(comparator_beta))
        )
        self.register_buffer(
            "c8_comparator_scale", torch.tensor(float(comparator_scale))
        )

        codebook = self.c8_codebook
        semantic_address = (
            (codebook[:, 0] >= 0).to(torch.long) * 4
            + (codebook[:, 1] >= 0).to(torch.long) * 2
            + (codebook[:, 0].abs() >= codebook[:, 1].abs()).to(torch.long)
        )
        semantic_to_phase = torch.empty(8, dtype=torch.long)
        semantic_to_phase[semantic_address] = torch.arange(8)
        self.register_buffer("c8_semantic_to_phase", semantic_to_phase)

    @property
    def comparator_beta(self):
        return self.c8_comparator_beta

    @property
    def comparator_scale(self):
        return self.c8_comparator_scale

    def _decode_activation_bits(self, activation_bits):
        if activation_bits.ndim != 4:
            raise ValueError("activation bits must be a 4D NCHW tensor")
        if activation_bits.shape[1] != 3 * self.in_channels:
            raise ValueError(
                "expected {} activation-bit channels, got {}".format(
                    3 * self.in_channels,
                    activation_bits.shape[1],
                )
            )
        signed_r, signed_i, signed_dominance = activation_bits.chunk(3, dim=1)
        dominance = (signed_dominance + 1.0) / 2.0
        codebook = self.c8_codebook.to(dtype=activation_bits.dtype)
        high = codebook.abs().amax()
        low = codebook.abs().amin()
        magnitude_r = low + (high - low) * dominance
        magnitude_i = high - (high - low) * dominance
        return signed_r * magnitude_r, signed_i * magnitude_i

    def _hard_phase_indices_from_semantic_bits(self, activation_bits):
        signed_r, signed_i, signed_dominance = activation_bits.detach().chunk(
            3, dim=1
        )
        address = (
            (signed_r >= 0.0).to(torch.long) * 4
            + (signed_i >= 0.0).to(torch.long) * 2
            + (signed_dominance >= 0.0).to(torch.long)
        )
        return self.c8_semantic_to_phase[address]

    def _comparator_features(self, decoded_r, decoded_i):
        difference = decoded_r - decoded_i
        total = decoded_r + decoded_i
        beta = self.comparator_beta.to(dtype=decoded_r.dtype)
        scale = self.comparator_scale.to(dtype=decoded_r.dtype)
        feature_a = torch.tanh(beta * difference / scale)
        feature_b = torch.tanh(-beta * difference / scale)
        feature_c = torch.tanh(beta * total / scale)
        feature_d = torch.tanh(-beta * total / scale)
        return self._group_major_features(
            (feature_a, feature_b, feature_c, feature_d)
        )

    def _differentiable_routes(self, weight_r_bin, weight_i_bin):
        positive_r = (weight_r_bin + 1.0) / 2.0
        positive_i = (weight_i_bin + 1.0) / 2.0
        state_pp = positive_r * positive_i
        state_nn = (1.0 - positive_r) * (1.0 - positive_i)
        state_pn = positive_r * (1.0 - positive_i)
        state_np = (1.0 - positive_r) * positive_i
        route_r = torch.stack(
            (state_pp, state_nn, state_pn, state_np), dim=1
        ).flatten(1, 2)
        route_i = torch.stack(
            (state_np, state_pn, state_pp, state_nn), dim=1
        ).flatten(1, 2)
        return route_r, route_i

    def _surrogate_output(
        self,
        decoded_r,
        decoded_i,
        weight_r_bin,
        weight_i_bin,
        alpha,
    ):
        feature_input = self._comparator_features(decoded_r, decoded_i)
        route_r, route_i = self._differentiable_routes(
            weight_r_bin, weight_i_bin
        )
        conv_args = {
            "stride": self.stride,
            "padding": self.padding,
            "dilation": self.dilation,
            "groups": self.groups,
        }
        signed_sum_r = F.conv2d(feature_input, route_r, **conv_args)
        signed_sum_i = F.conv2d(feature_input, route_i, **conv_args)
        return 2.0 * alpha * signed_sum_r, 2.0 * alpha * signed_sum_i

    def forward(self, activation_bits):
        decoded_r, decoded_i = self._decode_activation_bits(activation_bits)
        hard_index = self._hard_phase_indices_from_semantic_bits(
            activation_bits
        )
        weight_r_bin, weight_i_bin, alpha = self._binary_weight_state()
        hard_sum_r, hard_sum_i, local_count = (
            self._hard_local_sums_zero_padding_from_indices(hard_index)
        )
        hardware_r = (
            (hard_sum_r - local_count / 2.0) * 4.0 * alpha.detach()
        )
        hardware_i = (
            (hard_sum_i - local_count / 2.0) * 4.0 * alpha.detach()
        )
        if not torch.is_grad_enabled():
            return torch.complex(hardware_r, hardware_i)

        surrogate_r, surrogate_i = self._surrogate_output(
            decoded_r,
            decoded_i,
            weight_r_bin,
            weight_i_bin,
            alpha,
        )
        output_r = hardware_r.detach() - surrogate_r.detach() + surrogate_r
        output_i = hardware_i.detach() - surrogate_i.detach() + surrogate_i
        return torch.complex(output_r, output_i)


class FixedAnalyticDominanceComplexResNet(BinaryComplexResNet):
    """Phase1-initialized network using only fixed analytic local operators."""

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
        binary_stem=False,
        is_sar_input=False,
        dominance_beta=2.0,
        dominance_phase_normalized=False,
        comparator_beta=1.0,
        comparator_scale=1.0,
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
            binary_stem=binary_stem,
            is_sar_input=is_sar_input,
            is_binary=True,
            phase=3.1,
            c8_beta=dominance_beta,
            c8_codebook="octants",
            c8_grad_mode="semantic_ste",
            phase3p1_padding_mode="zero",
        )

        blocks = [
            module
            for module in self.modules()
            if isinstance(module, BiRealComplexResidualBlock)
        ]
        for block in blocks:
            old_conv = block.conv
            if not isinstance(old_conv, C8LUTAwareComplexBinaryConv2d):
                raise RuntimeError(
                    "fixed analytic flow expected a C8 local comparator"
                )
            block.act = FixedDominanceActivation(
                beta=dominance_beta,
                phase_normalized=dominance_phase_normalized,
            )
            block.conv = ComparatorAwareDominanceConv2d(
                in_channels=old_conv.in_channels,
                out_channels=old_conv.out_channels,
                kernel_size=old_conv.kernel_size,
                stride=old_conv.stride,
                padding=old_conv.padding,
                groups=old_conv.groups,
                per_channel=old_conv.per_channel,
                weight_grad_mode=old_conv.weight_grad_mode,
                comparator_beta=comparator_beta,
                comparator_scale=comparator_scale,
            )
            block.lut_inputs = 5

        self.phase = FIXED_ANALYTIC_PHASE
        self.lut_inputs = 5
        self.flow_name = "fixed analytic dominance comparator"
        self.dominance_beta = float(dominance_beta)
        self.dominance_phase_normalized = bool(dominance_phase_normalized)
        self.comparator_beta = float(comparator_beta)
        self.comparator_scale = float(comparator_scale)


def iter_fixed_analytic_convs(model):
    for module in model.modules():
        if isinstance(module, ComparatorAwareDominanceConv2d):
            yield module


@torch.no_grad()
def export_fixed_dominance_lut5():
    """Enumerate the fixed [sr, si, d, wr, wi] -> [br, bi] table."""

    states = torch.arange(32, dtype=torch.long)
    sign_r = ((states >> 4) & 1).to(torch.float32) * 2.0 - 1.0
    sign_i = ((states >> 3) & 1).to(torch.float32) * 2.0 - 1.0
    dominance = ((states >> 2) & 1).to(torch.float32)
    weight_r = ((states >> 1) & 1).to(torch.float32) * 2.0 - 1.0
    weight_i = (states & 1).to(torch.float32) * 2.0 - 1.0

    high = math.cos(math.pi / 8.0)
    low = math.sin(math.pi / 8.0)
    decoded_r = sign_r * (low + (high - low) * dominance)
    decoded_i = sign_i * (high - (high - low) * dominance)
    local_r = decoded_r * weight_r - decoded_i * weight_i
    local_i = decoded_r * weight_i + decoded_i * weight_r
    if bool((local_r == 0.0).any() or (local_i == 0.0).any()):
        raise RuntimeError("fixed dominance LUT unexpectedly contains a tie")
    return {
        "input_order": FIXED_ANALYTIC_BIT_NAMES + ("weight_r", "weight_i"),
        "real": (local_r >= 0.0).to(torch.uint8),
        "imag": (local_i >= 0.0).to(torch.uint8),
    }


def fixed_analytic_hardware_spec():
    return {
        "primitive": "two independent LUT5",
        "physical_inputs_per_lut": 5,
        "physical_outputs_per_complex_product": 2,
        "lut_units_per_complex_product": 2,
        "input_order": list(FIXED_ANALYTIC_BIT_NAMES)
        + ["weight_r", "weight_i"],
        "outputs": ["local_real_bit", "local_imag_bit"],
        "training_operator": "analytic complex multiply plus local comparator",
        "training_uses_lut": False,
        "truth_table_trainable": False,
        "post_accumulation": (
            "(sum(local_bits) - valid_fan_in / 2) * 4 * binary_weight_scale"
        ),
        "padding": "Padded locations perform no local hardware operation.",
    }


@torch.no_grad()
def fixed_analytic_diagnostics(model):
    tables = export_fixed_dominance_lut5()
    states0 = torch.arange(32, dtype=torch.long)
    states0 = states0[(states0 & 0b00100) == 0]
    states1 = states0 | 0b00100
    slice_diff = sum(
        int(
            (
                tables[name].index_select(0, states0)
                != tables[name].index_select(0, states1)
            ).sum().item()
        )
        for name in ("real", "imag")
    )
    slice_pairs = 2 * states0.numel()
    analytic_modules = list(iter_fixed_analytic_convs(model))
    lut_modules = sum(
        1 for module in model.modules() if isinstance(module, ComplexLUTConv2d)
    )
    return {
        "analytic_conv_modules": len(analytic_modules),
        "lut_modules": lut_modules,
        "training_uses_lut": False,
        "truth_table_trainable": False,
        "truth_table_entries_per_output": 32,
        "extra_bit_slice_hard_diff": slice_diff,
        "extra_bit_slice_pairs": slice_pairs,
        "extra_bit_slice_hard_diff_ratio": slice_diff / float(slice_pairs),
        "comparator_beta": (
            float(analytic_modules[0].comparator_beta.item())
            if analytic_modules
            else None
        ),
        "comparator_scale": (
            float(analytic_modules[0].comparator_scale.item())
            if analytic_modules
            else None
        ),
        "hardware_deployable": bool(analytic_modules and lut_modules == 0),
    }


class FixedAnalyticBitOccupancyTracker:
    """Count the eight hard semantic activation addresses."""

    def __init__(self, model):
        self.counts = None
        self.handles = [
            module.register_forward_pre_hook(self._hook, with_kwargs=True)
            for module in iter_fixed_analytic_convs(model)
        ]

    def _hook(self, module, args, kwargs):
        del module
        bits = kwargs.get("activation_bits")
        if bits is None and args:
            bits = args[0]
        if bits is None:
            return
        batch, channels3, height, width = bits.shape
        channels = channels3 // 3
        hard = (bits.detach() > 0.0).to(torch.long).reshape(
            batch, 3, channels, height, width
        )
        address = hard[:, 0] * 4 + hard[:, 1] * 2 + hard[:, 2]
        counts = torch.bincount(address.reshape(-1), minlength=8)
        self.counts = counts if self.counts is None else self.counts + counts

    def finish(self):
        for handle in self.handles:
            handle.remove()
        if self.counts is None:
            raise RuntimeError("fixed analytic occupancy captured no bits")
        counts = self.counts.detach().cpu()
        total = int(counts.sum().item())
        ratios = counts.to(torch.float64) / float(total)
        nonzero = ratios[ratios > 0.0]
        entropy = float(
            (-(nonzero * nonzero.log()).sum() / math.log(8.0)).item()
        )
        return {
            "counts": counts.tolist(),
            "ratios": ratios.tolist(),
            "total": total,
            "active_codes": int((counts > 0).sum().item()),
            "normalized_entropy": entropy,
            "extra_bit_one_ratio": int(counts[1::2].sum().item())
            / float(total),
        }


class FixedAnalyticBitGradientTracker:
    """Aggregate gradients entering the three bits used by the analytic op."""

    def __init__(self, model):
        self.handles = [
            module.register_forward_pre_hook(
                self._forward_hook, with_kwargs=True
            )
            for module in iter_fixed_analytic_convs(model)
        ]
        self.reset()

    def reset(self):
        self.elements = [0, 0, 0]
        self.nonzero = [None, None, None]
        self.absolute_sum = [None, None, None]
        self.maximum = [None, None, None]
        self.captured_tensors = 0

    def _forward_hook(self, module, args, kwargs):
        del module
        bits = kwargs.get("activation_bits")
        if bits is None and args:
            bits = args[0]
        if bits is None or not bits.requires_grad:
            return
        bits.register_hook(self._gradient_hook)

    def _gradient_hook(self, gradient):
        batch, channels3, height, width = gradient.shape
        channels = channels3 // 3
        values = gradient.detach().reshape(batch, 3, channels, height, width)
        self.captured_tensors += 1
        for bit in range(3):
            absolute = values[:, bit].abs()
            self.elements[bit] += absolute.numel()
            nonzero = (absolute > 0.0).sum()
            absolute_sum = absolute.sum()
            maximum = absolute.max()
            if self.nonzero[bit] is None:
                self.nonzero[bit] = nonzero
                self.absolute_sum[bit] = absolute_sum
                self.maximum[bit] = maximum
            else:
                self.nonzero[bit] = self.nonzero[bit] + nonzero
                self.absolute_sum[bit] = self.absolute_sum[bit] + absolute_sum
                self.maximum[bit] = torch.maximum(self.maximum[bit], maximum)

    def finish(self):
        result = {"captured_tensors": self.captured_tensors, "bits": {}}
        for index, name in enumerate(FIXED_ANALYTIC_BIT_NAMES):
            elements = self.elements[index]
            nonzero = (
                int(self.nonzero[index].item())
                if self.nonzero[index] is not None
                else 0
            )
            absolute_sum = (
                float(self.absolute_sum[index].item())
                if self.absolute_sum[index] is not None
                else 0.0
            )
            maximum = (
                float(self.maximum[index].item())
                if self.maximum[index] is not None
                else 0.0
            )
            result["bits"][name] = {
                "elements": elements,
                "nonzero_ratio": nonzero / float(elements) if elements else 0.0,
                "mean_abs": absolute_sum / float(elements) if elements else 0.0,
                "max_abs": maximum,
            }
        return result

    def close(self):
        for handle in self.handles:
            handle.remove()
        self.handles = []

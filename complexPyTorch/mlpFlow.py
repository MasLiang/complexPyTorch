#!/usr/bin/env python3
"""Isolated binary-MLP flow for Phases 1.2, 2.2, 3.2, and 4.2.

Phase 2.2 evaluates a binary MLP over ``[x_r, x_i, d, w_r, w_i]`` and keeps
its multi-level local scores. Phase 3.2 locally truncates the two scores, and
Phase 4.2 compiles the resulting binary function into two direct LUT5 tables.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .complexFunctions import (
    binary_scale_weight_complex,
    binary_sign,
    complex_avg_pool2d,
)
from .complexLayers import (
    BinaryComplexActivation,
    ComplexBatchNorm2d,
    ComplexConv2d,
    ComplexReLU,
    LUTFloatingConvFunction,
    SignWithSTE,
    binary_annealing,
)
from .complexResNet import LearnImagBlock, _SPECTRAL_SCHEMES, apply_spectral_pooling


MLP_FLOW_PHASES = (1.2, 2.2, 3.2, 4.2)
_D_BASIS_INDICES = (3, 5, 6, 7)


def _same_padding(kernel_size):
    if isinstance(kernel_size, tuple):
        return tuple(size // 2 for size in kernel_size)
    return kernel_size // 2


def _activation_basis(x_r, x_i, dominance):
    """Return [1, xr, xi, d, xr*xi, xr*d, xi*d, xr*xi*d]."""
    one = torch.ones_like(x_r)
    return torch.stack(
        (
            one,
            x_r,
            x_i,
            dominance,
            x_r * x_i,
            x_r * dominance,
            x_i * dominance,
            x_r * x_i * dominance,
        ),
        dim=2,
    )


def _activation_state_basis(states):
    x_r, x_i, dominance = states.unbind(dim=-1)
    one = torch.ones_like(x_r)
    return torch.stack(
        (
            one,
            x_r,
            x_i,
            dominance,
            x_r * x_i,
            x_r * dominance,
            x_i * dominance,
            x_r * x_i * dominance,
        ),
        dim=-1,
    )


def _weight_basis(w_r, w_i):
    return torch.stack(
        (torch.ones_like(w_r), w_r, w_i, w_r * w_i),
        dim=0,
    )


def _weight_state_basis(states):
    w_r, w_i = states.unbind(dim=-1)
    return torch.stack(
        (torch.ones_like(w_r), w_r, w_i, w_r * w_i),
        dim=-1,
    )


def _initial_operation_coefficients(num_sets):
    """Initialize every operation set to exact complex multiplication."""
    coefficients = torch.zeros(num_sets, 2, 8, 4)
    coefficients[:, 0, 1, 1] = 1.0
    coefficients[:, 0, 2, 2] = -1.0
    coefficients[:, 1, 1, 2] = 1.0
    coefficients[:, 1, 2, 1] = 1.0
    return coefficients
def _physical_states(device=None, dtype=torch.float32):
    state_index = torch.arange(32, device=device, dtype=torch.long)
    shifts = torch.arange(4, -1, -1, device=device, dtype=torch.long)
    states = ((state_index[:, None] >> shifts[None, :]) & 1).to(dtype)
    return states * 2.0 - 1.0


def _exact_complex_scores(states):
    x_r, x_i, _, w_r, w_i = states.unbind(dim=-1)
    return torch.stack(
        (
            x_r * w_r - x_i * w_i,
            x_r * w_i + x_i * w_r,
        ),
        dim=-1,
    )


def _initial_binary_mlp_parameters(num_sets, hidden_width, logit_init):
    """Build an exact binary detector MLP for the complex product."""
    if hidden_width < 64 or hidden_width % 64:
        raise ValueError("binary_mlp_hidden must be a positive multiple of 64")
    if logit_init <= 0.0 or logit_init > 1.0:
        raise ValueError("binary_mlp_logit_init must be in (0, 1]")

    states = _physical_states()
    copies_per_state = hidden_width // 32
    detector_addresses = states.repeat_interleave(copies_per_state, dim=0)
    hidden_weight = detector_addresses * float(logit_init)
    hidden_threshold = torch.full((hidden_width,), -4.0)

    targets = _exact_complex_scores(states)
    output_weight = torch.empty(2, hidden_width)
    for state_index in range(32):
        start = state_index * copies_per_state
        end = start + copies_per_state
        for output_index in range(2):
            target = float(targets[state_index, output_index].item())
            if target > 0.0:
                output_weight[output_index, start:end] = float(logit_init)
            elif target < 0.0:
                output_weight[output_index, start:end] = -float(logit_init)
            else:
                midpoint = start + copies_per_state // 2
                output_weight[output_index, start:midpoint] = float(logit_init)
                output_weight[output_index, midpoint:end] = -float(logit_init)

    return {
        "hidden_weight": hidden_weight.unsqueeze(0).repeat(num_sets, 1, 1),
        "hidden_threshold": hidden_threshold.unsqueeze(0).repeat(num_sets, 1),
        "output_weight": output_weight.unsqueeze(0).repeat(num_sets, 1, 1),
        "output_scale": 2.0 / float(copies_per_state),
    }



def _runtime_lut_conv_reference(
    x_probability,
    folded_table,
    runtime_inputs,
    kernel_size,
    stride,
    padding,
):
    """Small CPU reference path used by tests; CUDA uses the fused kernel."""
    batch, runtime_channels, height, width = x_probability.shape
    if runtime_channels % runtime_inputs != 0:
        raise ValueError("Runtime channels must be divisible by runtime inputs")
    input_channels = runtime_channels // runtime_inputs
    output_height = (height + 2 * padding - kernel_size) // stride + 1
    output_width = (width + 2 * padding - kernel_size) // stride + 1
    locations = output_height * output_width

    patches = F.unfold(
        x_probability,
        kernel_size=kernel_size,
        padding=padding,
        stride=stride,
    )
    patches = patches.reshape(
        batch,
        runtime_inputs,
        input_channels,
        kernel_size * kernel_size,
        locations,
    )
    patches = patches.permute(0, 2, 3, 1, 4).reshape(
        batch,
        input_channels * kernel_size * kernel_size,
        runtime_inputs,
        locations,
    )

    state = torch.arange(
        1 << runtime_inputs,
        device=x_probability.device,
        dtype=torch.long,
    )
    shifts = torch.arange(
        runtime_inputs - 1,
        -1,
        -1,
        device=x_probability.device,
        dtype=torch.long,
    )
    state_bits = ((state[:, None] >> shifts[None, :]) & 1).bool()
    probability = torch.where(
        state_bits.view(1, 1, -1, runtime_inputs, 1),
        patches.unsqueeze(2),
        1.0 - patches.unsqueeze(2),
    ).prod(dim=3)
    output = torch.einsum("bnsl,nso->bol", probability, folded_table)
    return output.reshape(batch, folded_table.shape[-1], output_height, output_width)


class MLPOperationComplexConv2d(nn.Module):
    """A learnable five-input, two-output local complex operation.

    Phase 1.2 retains the legacy floating polynomial. Phase 2.2 uses a binary
    detector MLP with multi-level local scores. Phase 3.2 locally truncates
    those scores, and Phase 4.2 compiles the hard function into trainable LUT5
    tables.
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=3,
        stride=1,
        padding=1,
        groups=1,
        phase=1.2,
        operation_sets=1,
        operation_allocation="layer",
        operation_sets_per_channel=1,
        dominance_beta=2.0,
        per_channel=True,
        weight_grad_mode="ste",
        binary_mlp_hidden=64,
        binary_mlp_logit_init=0.5,
        lut_logit_init=2.0,
        phase3p2_score_surrogate=False,
    ):
        super().__init__()
        phase = float(phase)
        if phase not in MLP_FLOW_PHASES:
            raise ValueError("MLP flow phase must be one of {}".format(MLP_FLOW_PHASES))
        if operation_allocation not in ("layer", "channel"):
            raise ValueError("operation_allocation must be 'layer' or 'channel'")
        if operation_sets < 1 or operation_sets_per_channel < 1:
            raise ValueError("Operation set counts must be positive")
        if dominance_beta <= 0.0:
            raise ValueError("dominance_beta must be positive")
        if binary_mlp_hidden < 64 or binary_mlp_hidden % 64:
            raise ValueError(
                "binary_mlp_hidden must be a positive multiple of 64"
            )
        if phase3p2_score_surrogate and phase != 3.2:
            raise ValueError(
                "phase3p2_score_surrogate is only valid for Phase 3.2"
            )

        if groups < 1 or in_channels % groups or out_channels % groups:
            raise ValueError("Invalid grouped-convolution channel configuration")

        self.in_channels = int(in_channels)
        self.out_channels = int(out_channels)
        self.kernel_size = int(kernel_size)
        self.stride = int(stride)
        self.padding = int(padding)
        self.groups = int(groups)
        self.phase = phase
        self.operation_allocation = operation_allocation
        self.layer_operation_sets = int(operation_sets)
        self.operation_sets_per_channel = int(operation_sets_per_channel)
        self.operation_sets = (
            int(operation_sets)
            if operation_allocation == "layer"
            else self.out_channels * int(operation_sets_per_channel)
        )
        self.per_channel = bool(per_channel)
        self.weight_grad_mode = weight_grad_mode
        self.lut_logit_init = float(lut_logit_init)
        self.binary_mlp_hidden = int(binary_mlp_hidden)
        self.binary_mlp_logit_init = float(binary_mlp_logit_init)
        self.phase3p2_score_surrogate = bool(phase3p2_score_surrogate)
        self.runtime_inputs = 3
        self.runtime_lut_size = 1 << self.runtime_inputs
        self.runtime_channels = self.runtime_inputs * self.in_channels
        self.in_channels_per_group = self.in_channels // self.groups
        self.lut_num = self.in_channels_per_group * self.kernel_size * self.kernel_size

        self.weight_r = nn.Parameter(
            torch.empty(
                self.out_channels,
                self.in_channels_per_group,
                self.kernel_size,
                self.kernel_size,
            )
        )
        self.weight_i = nn.Parameter(torch.empty_like(self.weight_r))
        nn.init.kaiming_uniform_(self.weight_r, a=math.sqrt(5.0))
        nn.init.kaiming_uniform_(self.weight_i, a=math.sqrt(5.0))

        self.register_buffer("dominance_beta", torch.tensor(float(dominance_beta)))
        states = _physical_states()
        exact_scores = _exact_complex_scores(states)
        self.register_buffer("binary_mlp_states", states, persistent=False)
        self.register_buffer(
            "initial_operation_scores",
            exact_scores.unsqueeze(0).repeat(self.operation_sets, 1, 1),
            persistent=False,
        )

        if self.phase == 1.2:
            initial_coefficients = _initial_operation_coefficients(
                self.operation_sets
            )
            self.operation_coefficients = nn.Parameter(
                initial_coefficients.clone()
            )
            self.register_buffer(
                "initial_operation_coefficients",
                initial_coefficients,
                persistent=False,
            )
            self.register_parameter("binary_mlp_hidden_weight", None)
            self.register_parameter("binary_mlp_output_weight", None)
            self.register_buffer(
                "binary_mlp_hidden_threshold", None, persistent=False
            )
            self.register_buffer(
                "binary_mlp_output_scale", None, persistent=False
            )
            self.register_buffer(
                "initial_binary_mlp_hidden_weight", None, persistent=False
            )
            self.register_buffer(
                "initial_binary_mlp_output_weight", None, persistent=False
            )
        else:
            self.register_parameter("operation_coefficients", None)
            self.register_buffer(
                "initial_operation_coefficients", None, persistent=False
            )
            mlp_initial = _initial_binary_mlp_parameters(
                self.operation_sets,
                self.binary_mlp_hidden,
                self.binary_mlp_logit_init,
            )
            self.binary_mlp_hidden_weight = nn.Parameter(
                mlp_initial["hidden_weight"].clone()
            )
            self.binary_mlp_output_weight = nn.Parameter(
                mlp_initial["output_weight"].clone()
            )
            self.register_buffer(
                "binary_mlp_hidden_threshold",
                mlp_initial["hidden_threshold"],
            )
            self.register_buffer(
                "binary_mlp_output_scale",
                torch.tensor(mlp_initial["output_scale"]),
            )
            self.register_buffer(
                "initial_binary_mlp_hidden_weight",
                mlp_initial["hidden_weight"],
                persistent=False,
            )
            self.register_buffer(
                "initial_binary_mlp_output_weight",
                mlp_initial["output_weight"],
                persistent=False,
            )
        if operation_allocation == "layer":
            output_set_ids = torch.arange(self.out_channels) % int(operation_sets)
            set_ids = output_set_ids.unsqueeze(0).expand(self.lut_num, -1).clone()
        else:
            channel_base = (
                torch.arange(self.out_channels).unsqueeze(0)
                * int(operation_sets_per_channel)
            )
            local_ids = (
                torch.arange(self.lut_num).unsqueeze(1)
                % int(operation_sets_per_channel)
            )
            set_ids = channel_base + local_ids
        self.register_buffer("operation_set_ids", set_ids.to(torch.long))

        connection_channel = torch.arange(self.in_channels).repeat_interleave(
            self.kernel_size * self.kernel_size
        )
        connection_y = (
            torch.arange(self.kernel_size * self.kernel_size) // self.kernel_size
        ).repeat(self.in_channels)
        connection_x = (
            torch.arange(self.kernel_size * self.kernel_size) % self.kernel_size
        ).repeat(self.in_channels)
        flat_channel = torch.empty(
            self.in_channels * self.kernel_size * self.kernel_size * self.runtime_inputs,
            dtype=torch.int32,
        )
        flat_y = torch.empty_like(flat_channel)
        flat_x = torch.empty_like(flat_channel)
        for input_index in range(self.runtime_inputs):
            flat_channel[input_index::self.runtime_inputs] = (
                connection_channel + input_index * self.in_channels
            )
            flat_y[input_index::self.runtime_inputs] = connection_y
            flat_x[input_index::self.runtime_inputs] = connection_x
        self.register_buffer("flat_channel", flat_channel)
        self.register_buffer("flat_y", flat_y)
        self.register_buffer("flat_x", flat_x)

        if self.phase == 4.2:
            initial_r, initial_i = self.enumerated_operation_scores()
            self.lut_r = nn.Parameter(self._scores_to_initial_logits(initial_r))
            self.lut_i = nn.Parameter(self._scores_to_initial_logits(initial_i))
            self.register_buffer(
                "compiled_lut_reference_r",
                self.lut_r.detach().clone(),
                persistent=False,
            )
            self.register_buffer(
                "compiled_lut_reference_i",
                self.lut_i.detach().clone(),
                persistent=False,
            )
        else:
            self.register_parameter("lut_r", None)
            self.register_parameter("lut_i", None)
            self.register_buffer(
                "compiled_lut_reference_r",
                None,
                persistent=False,
            )
            self.register_buffer(
                "compiled_lut_reference_i",
                None,
                persistent=False,
            )
        self.register_buffer("tau", torch.tensor(1.0))
        self.register_buffer("hard", torch.tensor(0.0))

    def _scores_to_initial_logits(self, scores):
        magnitude = torch.full_like(scores, self.lut_logit_init)
        return torch.where(scores >= 0.0, magnitude, -magnitude)

    def dominance_signal(self, source, binary):
        epsilon = max(torch.finfo(source.real.dtype).eps, 1e-6)
        magnitude = torch.sqrt(
            source.real.square() + source.imag.square() + epsilon * epsilon
        )
        margin = (source.real.abs() - source.imag.abs()) / magnitude
        beta = self.dominance_beta.to(dtype=margin.dtype)
        soft = torch.tanh(beta * margin)
        if not binary:
            return soft
        hard = torch.where(
            margin >= 0.0,
            torch.ones_like(margin),
            -torch.ones_like(margin),
        )
        return soft + (hard - soft).detach()

    def _selected_coefficients(self):
        selected = self.operation_coefficients[self.operation_set_ids]
        selected = selected.permute(1, 0, 2, 3, 4)
        return selected.reshape(
            self.out_channels,
            self.in_channels_per_group,
            self.kernel_size,
            self.kernel_size,
            2,
            8,
            4,
        )

    def _operation_convolution(self, inp, phase_source, binary):
        dominance = self.dominance_signal(phase_source, binary=binary)
        activation_basis = _activation_basis(inp.real, inp.imag, dominance)
        activation_features = activation_basis.reshape(
            inp.shape[0],
            self.in_channels * 8,
            inp.shape[2],
            inp.shape[3],
        )

        if binary:
            w_r = binary_sign(self.weight_r, grad_mode=self.weight_grad_mode)
            w_i = binary_sign(self.weight_i, grad_mode=self.weight_grad_mode)
        else:
            w_r = self.weight_r
            w_i = self.weight_i
        weight_basis = _weight_basis(w_r, w_i)
        selected = self._selected_coefficients()
        effective_kernel = torch.einsum(
            "oihwupq,qoihw->uoiphw",
            selected,
            weight_basis,
        ).reshape(
            2,
            self.out_channels,
            self.in_channels_per_group * 8,
            self.kernel_size,
            self.kernel_size,
        )

        output_r = F.conv2d(
            activation_features,
            effective_kernel[0],
            stride=self.stride,
            padding=self.padding,
            groups=self.groups,
        )
        output_i = F.conv2d(
            activation_features,
            effective_kernel[1],
            stride=self.stride,
            padding=self.padding,
            groups=self.groups,
        )
        if binary:
            alpha = binary_scale_weight_complex(
                torch.complex(self.weight_r, self.weight_i),
                per_channel=self.per_channel,
            ).reshape(1, self.out_channels, 1, 1)
            output_r = output_r * alpha
            output_i = output_i * alpha
        return torch.complex(output_r, output_i)

    def binary_mlp_scores(self):
        if self.binary_mlp_hidden_weight is None:
            raise RuntimeError("Binary MLP scores are unavailable in Phase 1.2")
        hidden_weight = binary_sign(
            self.binary_mlp_hidden_weight,
            grad_mode=self.weight_grad_mode,
        )
        hidden_pre = (
            torch.einsum(
                "shd,nd->snh",
                hidden_weight,
                self.binary_mlp_states,
            )
            + self.binary_mlp_hidden_threshold.unsqueeze(1)
        )
        hidden_bits = SignWithSTE.apply(hidden_pre)
        output_weight = binary_sign(
            self.binary_mlp_output_weight,
            grad_mode=self.weight_grad_mode,
        )
        scores = self.binary_mlp_output_scale * torch.einsum(
            "soh,snh->sno",
            output_weight,
            hidden_bits,
        )
        return scores[..., 0], scores[..., 1]

    def enumerated_operation_scores(self):
        if self.binary_mlp_hidden_weight is not None:
            return self.binary_mlp_scores()
        state_index = torch.arange(
            32,
            device=self.operation_coefficients.device,
            dtype=torch.long,
        )
        shifts = torch.arange(
            4,
            -1,
            -1,
            device=state_index.device,
            dtype=torch.long,
        )
        state = ((state_index[:, None] >> shifts[None, :]) & 1).to(
            dtype=self.operation_coefficients.dtype
        )
        state = state * 2.0 - 1.0
        activation_basis = _activation_state_basis(state[:, :3])
        weight_basis = _weight_state_basis(state[:, 3:])
        scores = torch.einsum(
            "sopq,np,nq->sno",
            self.operation_coefficients,
            activation_basis,
            weight_basis,
        )
        return scores[..., 0], scores[..., 1]

    @torch.no_grad()
    def initialize_lut_from_operation(self):
        if self.lut_r is None or self.lut_i is None:
            raise RuntimeError("Direct LUT parameters only exist in Phase 4.2")
        score_r, score_i = self.enumerated_operation_scores()
        self.lut_r.copy_(self._scores_to_initial_logits(score_r))
        self.lut_i.copy_(self._scores_to_initial_logits(score_i))
        self.compiled_lut_reference_r.copy_(self.lut_r)
        self.compiled_lut_reference_i.copy_(self.lut_i)

    @torch.no_grad()
    def set_lut_state(self, tau, hard_ratio):
        self.tau.fill_(float(tau))
        self.hard.fill_(float(hard_ratio))

    def hard_lut_tables(self):
        if self.phase == 4.2:
            score_r, score_i = self.lut_r, self.lut_i
        else:
            score_r, score_i = self.enumerated_operation_scores()
        return (score_r >= 0.0).to(torch.uint8), (score_i >= 0.0).to(torch.uint8)

    def _fold_table_with_weights(self, table, p00, p01, p10, p11):
        selected = table[self.operation_set_ids]
        grouped = selected.reshape(
            self.lut_num,
            self.out_channels,
            self.runtime_lut_size,
            4,
        )
        probabilities = torch.stack((p00, p01, p10, p11), dim=-1)
        return (grouped * probabilities.unsqueeze(2)).sum(dim=-1).permute(0, 2, 1)

    def _runtime_lut_convolution(self, x_probability, folded_table, offsets):
        if x_probability.is_cuda:
            return LUTFloatingConvFunction.apply(
                x_probability,
                folded_table.contiguous(),
                offsets,
                self.groups,
                self.runtime_inputs,
                self.kernel_size,
                self.stride,
                self.padding,
            )
        return _runtime_lut_conv_reference(
            x_probability,
            folded_table,
            self.runtime_inputs,
            self.kernel_size,
            self.stride,
            self.padding,
        )

    def _phase3_score_surrogate_outputs(
        self,
        x_probability,
        offsets,
        score_r,
        score_i,
        p00,
        p01,
        p10,
        p11,
    ):
        hard_r = (score_r >= 0.0).to(dtype=score_r.dtype)
        hard_i = (score_i >= 0.0).to(dtype=score_i.dtype)
        detached_probabilities = tuple(
            probability.detach() for probability in (p00, p01, p10, p11)
        )
        folded_hard_r = self._fold_table_with_weights(
            hard_r.detach(), *detached_probabilities
        )
        folded_hard_i = self._fold_table_with_weights(
            hard_i.detach(), *detached_probabilities
        )

        if not (self.training and torch.is_grad_enabled()):
            return (
                self._runtime_lut_convolution(
                    x_probability, folded_hard_r, offsets
                ),
                self._runtime_lut_convolution(
                    x_probability, folded_hard_i, offsets
                ),
                None,
                None,
            )

        folded_score_r = self._fold_table_with_weights(
            score_r, p00, p01, p10, p11
        )
        folded_score_i = self._fold_table_with_weights(
            score_i, p00, p01, p10, p11
        )
        if self.groups == 1:
            combined_r = torch.cat((folded_hard_r, folded_score_r), dim=-1)
            combined_i = torch.cat((folded_hard_i, folded_score_i), dim=-1)
            combined_output_r = self._runtime_lut_convolution(
                x_probability, combined_r, offsets
            )
            combined_output_i = self._runtime_lut_convolution(
                x_probability, combined_i, offsets
            )
            output_hard_r, output_score_r = combined_output_r.split(
                self.out_channels, dim=1
            )
            output_hard_i, output_score_i = combined_output_i.split(
                self.out_channels, dim=1
            )
        else:
            output_hard_r = self._runtime_lut_convolution(
                x_probability, folded_hard_r, offsets
            )
            output_hard_i = self._runtime_lut_convolution(
                x_probability, folded_hard_i, offsets
            )
            output_score_r = self._runtime_lut_convolution(
                x_probability, folded_score_r, offsets
            )
            output_score_i = self._runtime_lut_convolution(
                x_probability, folded_score_i, offsets
            )
        return output_hard_r, output_hard_i, output_score_r, output_score_i

    def _local_binary_convolution(self, inp, phase_source):
        dominance = self.dominance_signal(phase_source, binary=True)
        x_cat = torch.cat((inp.real, inp.imag, dominance), dim=1)
        x_probability = (x_cat + 1.0) / 2.0

        weight_r = self.weight_r.reshape(self.out_channels, self.lut_num).t()
        weight_i = self.weight_i.reshape(self.out_channels, self.lut_num).t()
        probability_r = SignWithSTE.apply(weight_r)
        probability_i = SignWithSTE.apply(weight_i)
        p00 = (1.0 - probability_r) * (1.0 - probability_i)
        p01 = (1.0 - probability_r) * probability_i
        p10 = probability_r * (1.0 - probability_i)
        p11 = probability_r * probability_i

        use_score_surrogate = (
            self.phase == 3.2 and self.phase3p2_score_surrogate
        )
        table_is_binary = self.phase != 2.2
        if self.phase == 2.2:
            table_r, table_i = self.enumerated_operation_scores()
        elif self.phase == 3.2:
            score_r, score_i = self.enumerated_operation_scores()
            if not use_score_surrogate:
                table_r = SignWithSTE.apply(score_r)
                table_i = SignWithSTE.apply(score_i)
        else:
            table_r = binary_annealing(self.lut_r, tau=self.tau, hard=self.hard)
            table_i = binary_annealing(self.lut_i, tau=self.tau, hard=self.hard)

        padded_width = inp.shape[3] + 2 * self.padding
        offsets = (
            (self.flat_y * padded_width + self.flat_x) * self.runtime_channels
            + self.flat_channel
        )
        if use_score_surrogate:
            output_r, output_i, score_output_r, score_output_i = (
                self._phase3_score_surrogate_outputs(
                    x_probability,
                    offsets,
                    score_r,
                    score_i,
                    p00,
                    p01,
                    p10,
                    p11,
                )
            )
        else:
            folded_r = self._fold_table_with_weights(
                table_r, p00, p01, p10, p11
            )
            folded_i = self._fold_table_with_weights(
                table_i, p00, p01, p10, p11
            )
            output_r = self._runtime_lut_convolution(
                x_probability, folded_r, offsets
            )
            output_i = self._runtime_lut_convolution(
                x_probability, folded_i, offsets
            )

        alpha = binary_scale_weight_complex(
            torch.complex(self.weight_r, self.weight_i),
            per_channel=self.per_channel,
        ).reshape(1, self.out_channels, 1, 1)
        if table_is_binary:
            output_r = (output_r - self.lut_num / 2.0) * 4.0
            output_i = (output_i - self.lut_num / 2.0) * 4.0
        output_r = output_r * alpha
        output_i = output_i * alpha
        if use_score_surrogate and score_output_r is not None:
            score_output_r = score_output_r * alpha
            score_output_i = score_output_i * alpha
            output_r = score_output_r + (output_r - score_output_r).detach()
            output_i = score_output_i + (output_i - score_output_i).detach()
        return torch.complex(output_r, output_i)

    def forward(self, inp, phase_source=None):
        if not torch.is_complex(inp):
            raise TypeError("MLPOperationComplexConv2d expects complex input")
        if phase_source is None:
            phase_source = inp
        if self.phase == 1.2:
            return self._operation_convolution(inp, phase_source, binary=False)
        return self._local_binary_convolution(inp, phase_source)


class MLPFlowResidualBlock(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        phase,
        kernel_size=3,
        stride=1,
        projection=False,
        spectral_pool_scheme="none",
        spectral_pool_gamma=0.0,
        operation_sets=1,
        operation_allocation="layer",
        operation_sets_per_channel=1,
        dominance_beta=2.0,
        per_channel=True,
        weight_grad_mode="ste",
        act_grad_mode="bireal",
        binary_mlp_hidden=64,
        binary_mlp_logit_init=0.5,
        lut_logit_init=2.0,
        phase3p2_score_surrogate=False,
    ):
        super().__init__()
        self.phase = float(phase)
        self.projection = bool(projection)
        self.spectral_pool_scheme = spectral_pool_scheme
        self.spectral_pool_gamma = float(spectral_pool_gamma)
        self.bn_pre = ComplexBatchNorm2d(in_channels, eps=1e-4)
        self.act = (
            ComplexReLU()
            if self.phase == 1.2
            else BinaryComplexActivation(grad_mode=act_grad_mode)
        )
        self.conv = MLPOperationComplexConv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=_same_padding(kernel_size),
            phase=self.phase,
            operation_sets=operation_sets,
            operation_allocation=operation_allocation,
            operation_sets_per_channel=operation_sets_per_channel,
            dominance_beta=dominance_beta,
            per_channel=per_channel,
            weight_grad_mode=weight_grad_mode,
            binary_mlp_hidden=binary_mlp_hidden,
            binary_mlp_logit_init=binary_mlp_logit_init,
            lut_logit_init=lut_logit_init,
            phase3p2_score_surrogate=phase3p2_score_surrogate,
        )
        self.bn_post = ComplexBatchNorm2d(out_channels, eps=1e-4)

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
                ComplexBatchNorm2d(out_channels, eps=1e-4),
            )
        else:
            self.proj = None

    def forward(self, inp):
        identity = inp
        phase_source = self.bn_pre(inp)
        output = self.act(phase_source)
        if self.projection and self.spectral_pool_scheme == "proj":
            output = apply_spectral_pooling(output, self.spectral_pool_gamma)
            phase_source = apply_spectral_pooling(
                phase_source,
                self.spectral_pool_gamma,
            )
        output = self.conv(output, phase_source=phase_source)
        output = self.bn_post(output)

        if self.proj is not None:
            if self.spectral_pool_scheme == "proj":
                identity = apply_spectral_pooling(
                    identity,
                    self.spectral_pool_gamma,
                )
            identity = self.proj(identity)
        return output + identity


class MLPFlowComplexResNet(nn.Module):
    def __init__(
        self,
        in_channels=3,
        num_blocks=3,
        start_filters=11,
        num_classes=10,
        phase=1.2,
        spectral_pool_scheme="none",
        spectral_pool_gamma=0.0,
        is_sar_input=False,
        binary_stem=False,
        operation_sets=1,
        operation_allocation="layer",
        operation_sets_per_channel=1,
        dominance_beta=2.0,
        per_channel=True,
        weight_grad_mode="ste",
        act_grad_mode="bireal",
        binary_mlp_hidden=64,
        binary_mlp_logit_init=0.5,
        lut_logit_init=2.0,
        phase3p2_score_surrogate=False,
    ):
        super().__init__()
        phase = float(phase)
        if phase not in MLP_FLOW_PHASES:
            raise ValueError("MLP flow phase must be one of {}".format(MLP_FLOW_PHASES))
        if spectral_pool_scheme not in _SPECTRAL_SCHEMES:
            raise ValueError("Unknown spectral_pool_scheme: {}".format(spectral_pool_scheme))

        self.phase = phase
        self.num_blocks = int(num_blocks)
        self.actual_blocks_per_stage = self.num_blocks * 2
        self.spectral_pool_scheme = spectral_pool_scheme
        self.spectral_pool_gamma = float(spectral_pool_gamma)
        self.is_sar_input = bool(is_sar_input)
        self.operation_allocation = operation_allocation
        self.operation_sets = int(operation_sets)
        self.operation_sets_per_channel = int(operation_sets_per_channel)
        self.binary_mlp_hidden = int(binary_mlp_hidden)
        self.binary_mlp_logit_init = float(binary_mlp_logit_init)
        self.phase3p2_score_surrogate = bool(phase3p2_score_surrogate)

        if not self.is_sar_input:
            self.learn_imag = LearnImagBlock(
                in_channels,
                in_channels,
                in_channels,
                kernel_size=1,
            )
        if binary_stem:
            raise ValueError("The isolated MLP flow currently keeps the stem full precision")
        self.conv1 = ComplexConv2d(
            in_channels,
            start_filters,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
        )
        self.bn1 = ComplexBatchNorm2d(start_filters, eps=1e-4)

        block_kwargs = {
            "phase": phase,
            "spectral_pool_scheme": spectral_pool_scheme,
            "spectral_pool_gamma": spectral_pool_gamma,
            "operation_sets": operation_sets,
            "operation_allocation": operation_allocation,
            "operation_sets_per_channel": operation_sets_per_channel,
            "dominance_beta": dominance_beta,
            "per_channel": per_channel,
            "weight_grad_mode": weight_grad_mode,
            "act_grad_mode": act_grad_mode,
            "binary_mlp_hidden": binary_mlp_hidden,
            "binary_mlp_logit_init": binary_mlp_logit_init,
            "lut_logit_init": lut_logit_init,
            "phase3p2_score_surrogate": phase3p2_score_surrogate,
        }
        channels = start_filters
        self.stage2 = self._make_stage(
            channels,
            channels,
            self.actual_blocks_per_stage,
            stride=1,
            block_kwargs=block_kwargs,
        )
        stride3 = 1 if spectral_pool_scheme == "nodownsample" else 2
        self.stage3 = self._make_stage(
            channels,
            channels * 2,
            self.actual_blocks_per_stage,
            stride=stride3,
            block_kwargs=block_kwargs,
        )
        channels *= 2
        stride4 = 1 if spectral_pool_scheme == "nodownsample" else 2
        self.stage4 = self._make_stage(
            channels,
            channels * 2,
            self.actual_blocks_per_stage,
            stride=stride4,
            block_kwargs=block_kwargs,
        )
        channels *= 2
        self.final_channels = channels
        self.fc = nn.Linear(self.final_channels * 2, num_classes)

    @staticmethod
    def _make_stage(in_channels, out_channels, num_blocks, stride, block_kwargs):
        layers = [
            MLPFlowResidualBlock(
                in_channels,
                out_channels,
                stride=stride,
                projection=True,
                **block_kwargs,
            )
        ]
        for _ in range(1, num_blocks):
            layers.append(
                MLPFlowResidualBlock(
                    out_channels,
                    out_channels,
                    stride=1,
                    projection=False,
                    **block_kwargs,
                )
            )
        return nn.ModuleList(layers)

    def _maybe_stage_pool(self, inp, block_index):
        if (
            self.spectral_pool_scheme == "stagemiddle"
            and block_index == self.actual_blocks_per_stage // 2
        ):
            return apply_spectral_pooling(inp, self.spectral_pool_gamma)
        return inp

    def forward(self, inp):
        if not self.is_sar_input and not torch.is_complex(inp):
            inp = torch.complex(inp, self.learn_imag(inp))
        output = self.bn1(self.conv1(inp))

        for index, block in enumerate(self.stage2):
            output = self._maybe_stage_pool(block(output), index)
        if self.spectral_pool_scheme == "nodownsample":
            output = apply_spectral_pooling(output, self.spectral_pool_gamma)
        for index, block in enumerate(self.stage3):
            output = self._maybe_stage_pool(block(output), index)
        if self.spectral_pool_scheme == "nodownsample":
            output = apply_spectral_pooling(output, self.spectral_pool_gamma)
        for index, block in enumerate(self.stage4):
            output = self._maybe_stage_pool(block(output), index)

        if self.spectral_pool_scheme == "nodownsample":
            output = apply_spectral_pooling(output, self.spectral_pool_gamma)
            output = complex_avg_pool2d(output, kernel_size=32)
        else:
            output = complex_avg_pool2d(output, kernel_size=8)
        output = torch.cat((output.real, output.imag), dim=1)
        return self.fc(output.reshape(output.shape[0], -1))


def iter_mlp_operation_layers(model):
    for module in model.modules():
        if isinstance(module, MLPOperationComplexConv2d):
            yield module


@torch.no_grad()
def initialize_phase4_luts(model):
    count = 0
    for module in iter_mlp_operation_layers(model):
        module.initialize_lut_from_operation()
        count += 1
    if count == 0:
        raise RuntimeError("No MLP-operation layers found for LUT initialization")
    return count


@torch.no_grad()
def set_phase4_lut_state(model, tau, hard_ratio):
    count = 0
    for module in iter_mlp_operation_layers(model):
        if module.phase == 4.2:
            module.set_lut_state(tau, hard_ratio)
            count += 1
    if count == 0:
        raise RuntimeError("No Phase 4.2 LUT layers found")
    return count


@torch.no_grad()
def clamp_binary_mlp_latents(model, limit=1.0):
    if limit <= 0.0:
        raise ValueError("Binary MLP latent clamp limit must be positive")
    count = 0
    for module in iter_mlp_operation_layers(model):
        for parameter in (
            module.binary_mlp_hidden_weight,
            module.binary_mlp_output_weight,
        ):
            if parameter is not None:
                parameter.clamp_(-float(limit), float(limit))
                count += parameter.numel()
    return count


@torch.no_grad()
def operation_diagnostics(model):
    layers = []
    total_hard_changed = 0
    total_hard_entries = 0
    total_d_sensitive = 0
    total_d_pairs = 0
    total_parameters = 0
    parameter_delta_sum = 0.0
    total_d_score_entries = 0
    d_score_abs_sum = 0.0
    total_score_entries = 0
    score_abs_sum = 0.0
    score_abs_min = math.inf
    zero_scores = 0

    for layer_index, module in enumerate(iter_mlp_operation_layers(model)):
        if module.operation_coefficients is not None:
            parameter_pairs = (
                (
                    module.operation_coefficients,
                    module.initial_operation_coefficients,
                ),
            )
        else:
            parameter_pairs = (
                (
                    module.binary_mlp_hidden_weight,
                    module.initial_binary_mlp_hidden_weight,
                ),
                (
                    module.binary_mlp_output_weight,
                    module.initial_binary_mlp_output_weight,
                ),
            )
        parameter_count = sum(current.numel() for current, _ in parameter_pairs)
        parameter_delta_sq = sum(
            (current - initial).square().sum().item()
            for current, initial in parameter_pairs
        )

        operation_score_r, operation_score_i = (
            module.enumerated_operation_scores()
        )
        if module.phase == 4.2:
            score_r, score_i = module.lut_r, module.lut_i
            initial_r = module.compiled_lut_reference_r
            initial_i = module.compiled_lut_reference_i
        else:
            score_r, score_i = operation_score_r, operation_score_i
            initial_r = module.initial_operation_scores[..., 0]
            initial_i = module.initial_operation_scores[..., 1]

        scores = torch.stack((score_r, score_i), dim=-1)
        initial_scores = torch.stack((initial_r, initial_i), dim=-1)
        hard = scores >= 0.0
        initial_hard = initial_scores >= 0.0
        changed = int((hard != initial_hard).sum().item())
        hard_entries = int(hard.numel())

        d0_indices = torch.arange(32, device=scores.device)
        d0_indices = d0_indices[((d0_indices >> 2) & 1) == 0]
        d1_indices = d0_indices | 0b00100
        d_score_delta = scores[:, d1_indices] - scores[:, d0_indices]
        d_sensitive = int(
            (hard[:, d0_indices] != hard[:, d1_indices]).sum().item()
        )
        d_pairs = int(hard[:, d0_indices].numel())

        score_abs = scores.abs()
        layer_score_abs_min = float(score_abs.min().item())
        layer_score_abs_mean = float(score_abs.mean().item())
        layer_zero_scores = int((score_abs < 1e-8).sum().item())
        layer_d_abs_mean = float(d_score_delta.abs().mean().item())
        layer_d_abs_max = float(d_score_delta.abs().max().item())
        layer_entry = {
            "layer_index": layer_index,
            "allocation": module.operation_allocation,
            "sets": module.operation_sets,
            "coefficient_delta_l2": math.sqrt(parameter_delta_sq),
            "d_coefficient_abs_mean": layer_d_abs_mean,
            "d_coefficient_abs_max": layer_d_abs_max,
            "score_abs_min": layer_score_abs_min,
            "score_abs_mean": layer_score_abs_mean,
            "zero_score_ratio": layer_zero_scores / float(score_abs.numel()),
            "hard_changed": changed,
            "hard_entries": hard_entries,
            "hard_changed_ratio": changed / float(hard_entries),
            "d_sensitive_pairs": d_sensitive,
            "d_pairs": d_pairs,
            "d_sensitive_ratio": d_sensitive / float(d_pairs),
        }
        layers.append(layer_entry)
        total_hard_changed += changed
        total_hard_entries += hard_entries
        total_d_sensitive += d_sensitive
        total_d_pairs += d_pairs
        total_parameters += parameter_count
        parameter_delta_sum += parameter_delta_sq
        total_d_score_entries += d_score_delta.numel()
        d_score_abs_sum += d_score_delta.abs().sum().item()
        total_score_entries += score_abs.numel()
        score_abs_sum += score_abs.sum().item()
        score_abs_min = min(score_abs_min, layer_score_abs_min)
        zero_scores += layer_zero_scores

    if not layers:
        raise RuntimeError("No MLP-operation layers found for diagnostics")
    return {
        "layers": layers,
        "layer_count": len(layers),
        "coefficient_count": total_parameters,
        "coefficient_delta_l2": math.sqrt(parameter_delta_sum),
        "d_coefficient_abs_mean": d_score_abs_sum / float(total_d_score_entries),
        "score_abs_min": score_abs_min,
        "score_abs_mean": score_abs_sum / float(total_score_entries),
        "zero_score_ratio": zero_scores / float(total_score_entries),
        "hard_changed": total_hard_changed,
        "hard_entries": total_hard_entries,
        "hard_changed_ratio": total_hard_changed / float(total_hard_entries),
        "d_sensitive_pairs": total_d_sensitive,
        "d_pairs": total_d_pairs,
        "d_sensitive_ratio": total_d_sensitive / float(total_d_pairs),
    }


@torch.no_grad()
def export_hard_lut_tables(model):
    tables = {}
    for name, module in model.named_modules():
        if isinstance(module, MLPOperationComplexConv2d):
            table_r, table_i = module.hard_lut_tables()
            tables[name] = {
                "real": table_r.cpu(),
                "imag": table_i.cpu(),
                "allocation": module.operation_allocation,
                "operation_sets": module.operation_sets,
            }
    if not tables:
        raise RuntimeError("No MLP-operation layers found for LUT export")
    return tables

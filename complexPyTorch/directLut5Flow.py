"""Isolated hard-forward Direct LUT5 flow with trainable activation bits.

The flow starts from a Phase1 checkpoint, but every residual-block local
operator is hardware-shaped from the first forward pass: three activation
bits plus two binary weight bits address two independent LUT5 tables, one for
the real output bit and one for the imaginary output bit.
"""

import math

import torch

from .complexBinaryResNet import BinaryComplexResNet
from .complexLayers import ComplexLUTConv2d


DIRECT_LUT5_PHASE = 2.5
DIRECT_LUT5_ENCODERS = ("phase", "dominance")


def direct_lut5_bit_names(encoder_mode):
    if encoder_mode == "phase":
        return ("phase_gray_2", "phase_gray_1", "phase_gray_0")
    if encoder_mode == "dominance":
        return ("sign_r", "sign_i", "dominance")
    raise ValueError("Unknown Direct LUT5 encoder: {}".format(encoder_mode))


class DirectLUT5ComplexResNet(BinaryComplexResNet):
    """Hard 5-to-2 LUT network initialized from an analytic C8 product."""

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
        encoder_mode="phase",
        c8_beta=2.0,
        lut_sets=1,
        lut_allocation="layer",
        lut_sets_per_channel=1,
        initial_logit_margin=0.25,
    ):
        if encoder_mode not in DIRECT_LUT5_ENCODERS:
            raise ValueError(
                "encoder_mode must be one of {}".format(
                    DIRECT_LUT5_ENCODERS
                )
            )
        if initial_logit_margin <= 0.0:
            raise ValueError("initial_logit_margin must be positive")

        semantic = encoder_mode == "dominance"
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
            phase=3.1,
            lut_sets=lut_sets,
            lut_allocation=lut_allocation,
            lut_sets_per_channel=lut_sets_per_channel,
            lut_inputs=5,
            lut_init_mode="raw",
            c8_beta=c8_beta,
            c8_codebook="octants" if semantic else "roots",
            c8_grad_mode="semantic_ste" if semantic else "softmax",
            phase3p1_mode="semantic_lut5" if semantic else "learned_lut5",
            lut_init_tau=1.0,
        )
        self.phase = DIRECT_LUT5_PHASE
        self.direct_lut5_encoder = encoder_mode
        self.direct_lut5_initial_logit_margin = float(initial_logit_margin)
        self._configure_hard_luts()

    @torch.no_grad()
    def _configure_hard_luts(self):
        for module in iter_direct_lut5_convs(self):
            for name in ("lut_r", "lut_i"):
                logits = getattr(module, name)
                signs = torch.where(
                    logits >= 0.0,
                    torch.ones_like(logits),
                    -torch.ones_like(logits),
                )
                logits.copy_(
                    signs
                    * logits.abs().clamp_min(
                        self.direct_lut5_initial_logit_margin
                    )
                )
                module.register_buffer(
                    "direct_initial_sign_{}".format(name[-1]),
                    (logits >= 0.0).detach().clone(),
                    persistent=False,
                )
            module.tau.fill_(1.0)
            module.hard.fill_(1.0)


def iter_direct_lut5_convs(model):
    for module in model.modules():
        if isinstance(module, ComplexLUTConv2d) and module.lut_inputs == 5:
            yield module


def set_direct_lut5_trainable(model, trainable):
    value = bool(trainable)
    for module in iter_direct_lut5_convs(model):
        if module.lut_parameterization != "direct":
            raise RuntimeError("Direct LUT5 flow requires direct LUT parameters")
        module.lut_r.requires_grad_(value)
        module.lut_i.requires_grad_(value)


def direct_lut5_margin_loss(model, margin):
    losses = []
    for module in iter_direct_lut5_convs(model):
        for logits in module.effective_lut_logits():
            losses.append(torch.relu(float(margin) - logits.abs()).mean())
    if not losses:
        raise RuntimeError("Direct LUT5 model contains no LUT modules")
    return torch.stack(losses).mean()


@torch.no_grad()
def direct_lut5_diagnostics(model, near_zero_margin=0.1):
    logits_all = []
    sign_differences = 0
    entries = 0
    slice_hard_differences = 0
    slice_soft_difference_sum = 0.0
    slice_pairs = 0
    fully_hard = True
    module_count = 0

    state = torch.arange(32, dtype=torch.long)
    slice0 = state[(state & 0b00100) == 0]
    slice1 = slice0 | 0b00100

    for module in iter_direct_lut5_convs(model):
        module_count += 1
        fully_hard = fully_hard and bool(module.hard.item() >= 0.5)
        for suffix, logits in zip(("r", "i"), module.effective_lut_logits()):
            detached = logits.detach()
            initial = getattr(module, "direct_initial_sign_{}".format(suffix))
            current = detached >= 0.0
            entries += current.numel()
            sign_differences += int((current != initial).sum().item())
            left = detached.index_select(1, slice0.to(detached.device))
            right = detached.index_select(1, slice1.to(detached.device))
            slice_hard_differences += int(
                ((left >= 0.0) != (right >= 0.0)).sum().item()
            )
            slice_soft_difference_sum += float(
                (left - right).abs().sum().item()
            )
            slice_pairs += left.numel()
            logits_all.append(detached.reshape(-1).cpu())

    if not logits_all:
        raise RuntimeError("Direct LUT5 diagnostics found no LUT entries")
    values = torch.cat(logits_all)
    near_zero = int((values.abs() < float(near_zero_margin)).sum().item())
    return {
        "lut_modules": module_count,
        "entries": entries,
        "sign_diff": sign_differences,
        "sign_diff_ratio": sign_differences / float(entries),
        "near_zero": near_zero,
        "near_zero_ratio": near_zero / float(entries),
        "near_zero_margin": float(near_zero_margin),
        "abs_logit_min": float(values.abs().min().item()),
        "abs_logit_mean": float(values.abs().mean().item()),
        "abs_logit_max": float(values.abs().max().item()),
        "extra_bit_slice_hard_diff": slice_hard_differences,
        "extra_bit_slice_pairs": slice_pairs,
        "extra_bit_slice_hard_diff_ratio": (
            slice_hard_differences / float(slice_pairs)
        ),
        "extra_bit_slice_soft_abs_diff_mean": (
            slice_soft_difference_sum / float(slice_pairs)
        ),
        "fully_hard": fully_hard,
    }


@torch.no_grad()
def capture_hard_lut_tables(model):
    tables = {}
    for name, module in model.named_modules():
        if not isinstance(module, ComplexLUTConv2d) or module.lut_inputs != 5:
            continue
        lut_r, lut_i = module.export_hard_lut_tables()
        tables[name] = {
            "real": lut_r.detach().cpu(),
            "imag": lut_i.detach().cpu(),
        }
    if not tables:
        raise RuntimeError("Direct LUT5 export found no LUT tables")
    return tables


def direct_lut5_hardware_spec(encoder_mode):
    bit_names = direct_lut5_bit_names(encoder_mode)
    return {
        "primitive": "two independent LUT5",
        "physical_inputs_per_lut": 5,
        "physical_outputs_per_complex_product": 2,
        "lut_units_per_complex_product": 2,
        "activation_encoder": encoder_mode,
        "input_order": list(bit_names) + ["weight_r", "weight_i"],
        "outputs": ["local_real_bit", "local_imag_bit"],
        "local_outputs_are_binary": True,
        "post_accumulation": (
            "(sum(local_bits) - fan_in / 2) * 4 * binary_weight_scale"
        ),
        "padding": "Convolution padding is skipped, not encoded as data.",
    }


class DirectBitOccupancyTracker:
    """Measure the hard three-bit activation addresses seen by LUT layers."""

    def __init__(self, model):
        self.counts = None
        self.handles = []
        for module in iter_direct_lut5_convs(model):
            self.handles.append(
                module.register_forward_pre_hook(
                    self._hook,
                    with_kwargs=True,
                )
            )

    def _hook(self, module, args, kwargs):
        del module, args
        activation_bits = kwargs.get("activation_bits")
        if activation_bits is None:
            return
        batch, channels3, height, width = activation_bits.shape
        if channels3 % 3 != 0:
            raise RuntimeError("Direct LUT5 activation channel count is invalid")
        channels = channels3 // 3
        bits = (activation_bits.detach() > 0.0).to(torch.long).reshape(
            batch, 3, channels, height, width
        )
        addresses = bits[:, 0] * 4 + bits[:, 1] * 2 + bits[:, 2]
        counts = torch.bincount(addresses.reshape(-1), minlength=8)
        if self.counts is None:
            self.counts = counts
        else:
            self.counts = self.counts + counts

    def finish(self):
        for handle in self.handles:
            handle.remove()
        if self.counts is None:
            raise RuntimeError("Direct LUT5 occupancy tracker captured no entries")
        counts = self.counts.detach().cpu()
        total = int(counts.sum().item())
        if total == 0:
            raise RuntimeError("Direct LUT5 occupancy tracker captured no entries")
        ratios = counts.to(torch.float64) / float(total)
        nonzero = ratios[ratios > 0.0]
        entropy = float((-(nonzero * nonzero.log()).sum() / math.log(8.0)).item())
        extra_one = int(counts[1::2].sum().item())
        return {
            "counts": counts.tolist(),
            "ratios": ratios.tolist(),
            "total": total,
            "active_codes": int((counts > 0).sum().item()),
            "normalized_entropy": entropy,
            "extra_bit_one_ratio": extra_one / float(total),
        }


class DirectBitGradientTracker:
    """Aggregate gradients entering each of the three explicit address bits."""

    def __init__(self, model):
        self.handles = []
        for module in iter_direct_lut5_convs(model):
            self.handles.append(
                module.register_forward_pre_hook(
                    self._forward_hook,
                    with_kwargs=True,
                )
            )
        self.reset()

    def reset(self):
        self.elements = [0, 0, 0]
        self.nonzero = [None, None, None]
        self.absolute_sum = [None, None, None]
        self.maximum = [None, None, None]
        self.captured_tensors = 0

    def _forward_hook(self, module, args, kwargs):
        del module, args
        activation_bits = kwargs.get("activation_bits")
        if activation_bits is None or not activation_bits.requires_grad:
            return
        activation_bits.register_hook(self._gradient_hook)

    def _gradient_hook(self, gradient):
        batch, channels3, height, width = gradient.shape
        if channels3 % 3 != 0:
            raise RuntimeError("Direct LUT5 gradient channel count is invalid")
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
                self.absolute_sum[bit] = (
                    self.absolute_sum[bit] + absolute_sum
                )
                self.maximum[bit] = torch.maximum(
                    self.maximum[bit], maximum
                )

    def finish(self, bit_names):
        result = {"captured_tensors": self.captured_tensors, "bits": {}}
        for index, name in enumerate(bit_names):
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
                "nonzero_ratio": (
                    nonzero / float(elements) if elements else 0.0
                ),
                "mean_abs": (
                    absolute_sum / float(elements)
                    if elements
                    else 0.0
                ),
                "max_abs": maximum,
            }
        return result

    def close(self):
        for handle in self.handles:
            handle.remove()
        self.handles = []

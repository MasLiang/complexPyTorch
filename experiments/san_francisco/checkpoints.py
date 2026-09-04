"""Checkpoint transitions for the San Francisco BiReal three-stage flow."""

from pathlib import Path

import torch

from complexPyTorch.complexLayers import PairLUT4ComplexConv2d


def checkpoint_state(checkpoint):
    if isinstance(checkpoint, dict) and isinstance(checkpoint.get("model"), dict):
        return checkpoint["model"]
    if isinstance(checkpoint, dict) and checkpoint and all(
        torch.is_tensor(value) for value in checkpoint.values()
    ):
        return checkpoint
    raise ValueError("Checkpoint does not contain a model state dictionary")


def load_checkpoint(path, device="cpu"):
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError("Checkpoint not found: {}".format(path))
    return torch.load(path, map_location=device)


def load_matching_stage(model, path, expected_stage, device="cpu"):
    """Load an architecture-compatible checkpoint and require full coverage."""

    checkpoint = load_checkpoint(path, device=device)
    stored_stage = checkpoint.get("stage") if isinstance(checkpoint, dict) else None
    if stored_stage is not None and stored_stage != expected_stage:
        raise ValueError(
            "Checkpoint stage is {!r}, expected {!r}".format(
                stored_stage, expected_stage
            )
        )
    model.load_state_dict(checkpoint_state(checkpoint), strict=True)
    return checkpoint



def load_lut6_residual_from_lut4(model, path, device="cpu"):
    """Expand categorical LUT4 bases and initialize zero dominance residuals."""

    checkpoint = load_checkpoint(path, device=device)
    stored_stage = checkpoint.get("stage") if isinstance(checkpoint, dict) else None
    if stored_stage is not None and stored_stage != "bireal_lut":
        raise ValueError(
            "LUT6 residual initialization requires bireal_lut, got {!r}".format(
                stored_stage
            )
        )
    source = checkpoint_state(checkpoint)
    target = model.state_dict()
    target_luts = [
        (name, module)
        for name, module in model.named_modules()
        if isinstance(module, PairLUT4ComplexConv2d)
    ]
    lut_prefixes = [name + "." for name, _ in target_luts]

    def belongs_to_lut(key):
        return any(key.startswith(prefix) for prefix in lut_prefixes)

    shared = {
        key: value
        for key, value in source.items()
        if key in target and not belongs_to_lut(key) and target[key].shape == value.shape
    }
    missing_shared = [
        key for key in target if not belongs_to_lut(key) and key not in shared
    ]
    if missing_shared:
        raise RuntimeError(
            "LUT4 checkpoint does not cover shared LUT6 state: {}".format(
                missing_shared
            )
        )
    model.load_state_dict(shared, strict=False)

    expanded = 0
    for name, module in target_luts:
        source_key = name + ".weight"
        source_weight = source.get(source_key)
        expected_shape = (
            module.out_channels,
            module.group_num,
            16,
            4,
        )
        if source_weight is None or tuple(source_weight.shape) != expected_shape:
            raise RuntimeError(
                "{} must have categorical LUT4 shape {}, got {}".format(
                    source_key,
                    expected_shape,
                    None if source_weight is None else tuple(source_weight.shape),
                )
            )
        with torch.no_grad():
            module.dominance_base.copy_(
                source_weight.to(
                    device=module.dominance_base.device,
                    dtype=module.dominance_base.dtype,
                )
            )
            module.weight.zero_()
            module.dominance_base_correction.zero_()
            module.dominance_residual_alpha.fill_(0.0)
            module.dominance_base_alpha.fill_(0.0)
        expanded += 1
    if expanded == 0:
        raise RuntimeError("No categorical LUT4 operators were expanded")
    return checkpoint, expanded


def load_lut6_lut4_residual(model, path, device="cpu"):
    """Load a trained LUT6 residual model before releasing LUT4 correction."""

    return load_matching_stage(model, path, "lut6_residual", device=device)



def load_bireal_from_fp(model, path, device="cpu"):
    """Initialize binary operators from the corresponding FP operator weights."""

    return load_matching_stage(model, path, "bireal_fp", device=device)


def load_lut_from_bireal(model, path, device="cpu"):
    """Copy shared state and compile every binary complex conv into PairLUT4."""

    checkpoint = load_checkpoint(path, device=device)
    stored_stage = checkpoint.get("stage") if isinstance(checkpoint, dict) else None
    if stored_stage is not None and stored_stage != "bireal":
        raise ValueError(
            "LUT initialization requires a bireal checkpoint, got {!r}".format(
                stored_stage
            )
        )
    source = checkpoint_state(checkpoint)
    target = model.state_dict()
    lut_prefixes = [
        name + "."
        for name, module in model.named_modules()
        if isinstance(module, PairLUT4ComplexConv2d)
    ]

    def belongs_to_lut(key):
        return any(key.startswith(prefix) for prefix in lut_prefixes)

    shared = {
        key: value
        for key, value in source.items()
        if key in target and not belongs_to_lut(key) and target[key].shape == value.shape
    }
    missing_shared = [
        key
        for key in target
        if not belongs_to_lut(key) and key not in shared
    ]
    if missing_shared:
        raise RuntimeError(
            "BiReal checkpoint does not cover shared LUT-stage state: {}".format(
                missing_shared
            )
        )
    model.load_state_dict(shared, strict=False)

    compiled = 0
    for name, module in model.named_modules():
        if not isinstance(module, PairLUT4ComplexConv2d):
            continue
        real_key = name + ".conv_r.weight"
        imag_key = name + ".conv_i.weight"
        if real_key not in source or imag_key not in source:
            raise RuntimeError(
                "Cannot compile {}: missing {} or {}".format(
                    name, real_key, imag_key
                )
            )
        module.initialize_from_binary_complex_weights(
            source[real_key], source[imag_key]
        )
        compiled += 1
    if compiled == 0:
        raise RuntimeError("No PairLUT4 operators were compiled")
    return checkpoint, compiled

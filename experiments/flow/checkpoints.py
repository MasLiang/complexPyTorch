"""Shared FP -> BiReal -> LUT4 -> LUT6 checkpoint transitions."""

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


def _require_stage(checkpoint, expected):
    stage = checkpoint.get("stage") if isinstance(checkpoint, dict) else None
    if stage is not None and stage != expected:
        raise ValueError("Checkpoint stage is {!r}, expected {!r}".format(stage, expected))


def load_matching_stage(model, path, expected_stage, device="cpu"):
    checkpoint = load_checkpoint(path, device=device)
    _require_stage(checkpoint, expected_stage)
    model.load_state_dict(checkpoint_state(checkpoint), strict=True)
    return checkpoint


def _lut_modules(model):
    modules = [
        (name, module)
        for name, module in model.named_modules()
        if isinstance(module, PairLUT4ComplexConv2d)
    ]
    if not modules:
        raise RuntimeError("Model contains no PairLUT4ComplexConv2d operators")
    return modules


def _load_shared_non_lut_state(model, source):
    target = model.state_dict()
    prefixes = [name + "." for name, _ in _lut_modules(model)]

    def is_lut_key(key):
        return any(key.startswith(prefix) for prefix in prefixes)

    shared = {
        key: value
        for key, value in source.items()
        if key in target and not is_lut_key(key) and target[key].shape == value.shape
    }
    missing = [key for key in target if not is_lut_key(key) and key not in shared]
    if missing:
        raise RuntimeError("Checkpoint does not cover shared state: {}".format(missing))
    model.load_state_dict(shared, strict=False)


def compile_lut4_from_bireal(model, path, device="cpu"):
    checkpoint = load_checkpoint(path, device=device)
    _require_stage(checkpoint, "bireal")
    source = checkpoint_state(checkpoint)
    _load_shared_non_lut_state(model, source)
    compiled = 0
    for name, module in _lut_modules(model):
        real_key = name + ".conv_r.weight"
        imag_key = name + ".conv_i.weight"
        if real_key not in source or imag_key not in source:
            raise RuntimeError("Cannot compile {} from binary weights".format(name))
        module.initialize_from_binary_complex_weights(source[real_key], source[imag_key])
        compiled += 1
    return checkpoint, compiled


def expand_lut6_from_lut4(model, path, device="cpu"):
    checkpoint = load_checkpoint(path, device=device)
    _require_stage(checkpoint, "lut4")
    source = checkpoint_state(checkpoint)
    _load_shared_non_lut_state(model, source)
    expanded = 0
    for name, module in _lut_modules(model):
        if module.parameterization != "categorical_residual" or module.lut_inputs != 6:
            raise ValueError("{} is not a categorical dominance LUT6".format(name))
        source_key = name + ".weight"
        source_weight = source.get(source_key)
        expected_shape = (module.out_channels, module.group_num, 16, 4)
        if source_weight is None or tuple(source_weight.shape) != expected_shape:
            raise RuntimeError("{} must have categorical LUT4 shape {}".format(source_key, expected_shape))
        with torch.no_grad():
            module.dominance_base.copy_(source_weight.to(module.dominance_base))
            module.weight.zero_()
            module.dominance_base_correction.zero_()
            module.dominance_residual_alpha.fill_(0.0)
            module.dominance_base_alpha.fill_(0.0)
        expanded += 1
    return checkpoint, expanded

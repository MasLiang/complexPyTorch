"""Structural tests for the San Francisco BiReal three-stage flow."""

from pathlib import Path

import torch

from complexPyTorch.complexLayers import (
    BinaryComplexBitActivation,
    BinaryComplexConv2d,
    PairLUT4ComplexConv2d,
)
from experiments.san_francisco.checkpoints import (
    load_bireal_from_fp,
    load_lut_from_bireal,
    load_lut6_lut4_residual,
    load_lut6_residual_from_lut4,
)
from experiments.san_francisco.models import build_san_francisco_bireal_model


def small_model(stage):
    return build_san_francisco_bireal_model(
        stage,
        start_filters=4,
        num_blocks=1,
        pair_lut_parameterization="categorical",
    )


def save_stage(model, stage, path):
    torch.save({"model": model.state_dict(), "stage": stage}, path)


def test_all_stages_forward_and_backward():
    inputs = torch.randn(2, 6, 11, 11, dtype=torch.complex64)
    for stage in (
        "bireal_fp",
        "bireal",
        "bireal_lut",
        "lut6_residual",
        "lut6_lut4_residual",
    ):
        model = small_model(stage)
        logits = model(inputs)
        assert logits.shape == (2, 5)
        logits.square().mean().backward()
        assert all(
            parameter.grad is None or torch.isfinite(parameter.grad).all()
            for parameter in model.parameters()
        )


def test_fp_to_bireal_checkpoint_transition(tmp_path):
    fp_model = small_model("bireal_fp")
    checkpoint = Path(tmp_path) / "fp.pt"
    save_stage(fp_model, "bireal_fp", checkpoint)
    bireal_model = small_model("bireal")
    load_bireal_from_fp(bireal_model, checkpoint)
    assert any(
        isinstance(module, BinaryComplexConv2d)
        for module in bireal_model.modules()
    )
    assert torch.equal(bireal_model.fc.weight, fp_model.fc.weight)



def test_lut4_to_residual_stages_are_initially_equivalent(tmp_path):
    inputs = torch.randn(2, 6, 11, 11, dtype=torch.complex64)
    lut4_model = small_model("bireal_lut").eval()
    lut4_checkpoint = Path(tmp_path) / "lut4.pt"
    save_stage(lut4_model, "bireal_lut", lut4_checkpoint)

    lut6_model = small_model("lut6_residual").eval()
    _, expanded = load_lut6_residual_from_lut4(lut6_model, lut4_checkpoint)
    with torch.no_grad():
        lut4_logits = lut4_model(inputs)
        lut6_logits = lut6_model(inputs)
    assert expanded > 0
    assert torch.equal(lut4_logits, lut6_logits)
    for module in lut6_model.modules():
        if not isinstance(module, PairLUT4ComplexConv2d):
            continue
        assert module.parameterization == "categorical_residual"
        assert module.lut_inputs == 6
        assert torch.count_nonzero(module.weight) == 0
        assert torch.count_nonzero(module.dominance_base_correction) == 0

    residual_checkpoint = Path(tmp_path) / "lut6_residual.pt"
    save_stage(lut6_model, "lut6_residual", residual_checkpoint)
    final_model = small_model("lut6_lut4_residual").eval()
    load_lut6_lut4_residual(final_model, residual_checkpoint)
    with torch.no_grad():
        final_logits = final_model(inputs)
    assert torch.equal(lut6_logits, final_logits)

def test_bireal_to_lut_checkpoint_transition(tmp_path):
    bireal_model = small_model("bireal")
    checkpoint = Path(tmp_path) / "bireal.pt"
    save_stage(bireal_model, "bireal", checkpoint)
    lut_model = small_model("bireal_lut")
    _, compiled = load_lut_from_bireal(lut_model, checkpoint)
    lut_modules = [
        module
        for module in lut_model.modules()
        if isinstance(module, PairLUT4ComplexConv2d)
    ]
    assert compiled == len(lut_modules) > 0
    assert all(module.lut_inputs == 4 for module in lut_modules)
    assert all(module.parameterization == "categorical" for module in lut_modules)
    assert sum(
        isinstance(module, BinaryComplexBitActivation)
        for module in lut_model.modules()
    ) == compiled
    assert torch.equal(lut_model.fc.weight, bireal_model.fc.weight)



def test_residual_checkpoint_selection_waits_for_full_alpha():
    from types import SimpleNamespace

    from experiments.san_francisco.train_bireal_flow import (
        residual_selection_is_eligible,
    )

    args = SimpleNamespace(residual_alpha_end=1.0, base_alpha_end=0.0)
    assert not residual_selection_is_eligible(0.9, 0.0, args)
    assert residual_selection_is_eligible(1.0, 0.0, args)
    args.base_alpha_end = 1.0
    assert not residual_selection_is_eligible(1.0, 0.95, args)
    assert residual_selection_is_eligible(1.0, 1.0, args)


def test_option_a_shortcut_removes_projection_parameters():
    model = build_san_francisco_bireal_model(
        "lut6_residual",
        start_filters=4,
        num_blocks=1,
        shortcut_mode="option_a",
    )
    inputs = torch.randn(2, 6, 11, 11, dtype=torch.complex64)
    assert model(inputs).shape == (2, 5)
    projected_blocks = [model.stage3[0], model.stage4[0]]
    assert all(
        sum(parameter.numel() for parameter in block.proj.parameters()) == 0
        for block in projected_blocks
    )
    projection_input = inputs.new_zeros(2, 4, 12, 12)
    assert model.stage3[0].proj(projection_input).shape == (2, 8, 6, 6)

#!/usr/bin/env python3
"""Audit the clean Phase 1/2/3 complex Bi-Real route."""

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import training
from complexPyTorch.complexBinaryResNet import BinaryComplexResNet
from complexPyTorch.complexLayers import (
    BinaryComplexActivation,
    BinaryComplexBitActivation,
    BinaryComplexConv2d,
    ComplexConv2d,
    PairLUT4ComplexConv2d,
    TripleLUT6ComplexConv2d,
    TwoLUTComplexConv2d,
)


def audit(root):
    root = Path(root)
    failures = []
    if set(training.PHASE_DESCRIPTIONS) != {1, 2, 3}:
        failures.append("training.PHASE_DESCRIPTIONS is not exactly {1, 2, 3}")

    scripts = sorted(path.name for path in root.glob("run_phase*.sh"))
    if scripts != ["run_phase1.sh", "run_phase2.sh", "run_phase3.sh"]:
        failures.append("active phase scripts are {}".format(scripts))

    models = {
        phase: BinaryComplexResNet(
            in_channels=3,
            num_blocks=1,
            start_filters=16,
            num_classes=10,
            is_sar_input=False,
            phase=phase,
        )
        for phase in (1, 2, 3)
    }
    expected = {
        1: ComplexConv2d,
        2: BinaryComplexConv2d,
        3: PairLUT4ComplexConv2d,
    }
    for phase, model in models.items():
        if not isinstance(model.stage2[0].conv, expected[phase]):
            failures.append("Phase {} uses {}".format(phase, type(model.stage2[0].conv).__name__))
        if not hasattr(model, "learn_imag"):
            failures.append("Phase {} CIFAR path has no LearnImagBlock".format(phase))
        if hasattr(model.stage2[0], "bn_pre"):
            failures.append("Phase {} still exposes pre-BN".format(phase))

    phase3 = models[3]
    if not isinstance(models[2].stage2[0].act, BinaryComplexActivation):
        failures.append("Phase 2 does not use signed binary activation")
    if isinstance(models[2].stage2[0].act, BinaryComplexBitActivation):
        failures.append("Phase 2 unexpectedly uses 0/1 LUT bit activation")
    if not isinstance(phase3.stage2[0].act, BinaryComplexBitActivation):
        failures.append("Phase 3 does not use exact 0/1 LUT bit activation")
    operators = [
        block.conv
        for stage in (phase3.stage2, phase3.stage3, phase3.stage4)
        for block in stage
    ]
    if not all(isinstance(op, PairLUT4ComplexConv2d) for op in operators):
        failures.append("Phase 3 contains a non-grouped complex LUT operator")
    for operator in operators:
        expected_groups = (
            operator.logical_positions
            + operator.complex_inputs_per_lut
            - 1
        ) // operator.complex_inputs_per_lut
        expected_shape = (
            2 * operator.out_channels,
            expected_groups,
            1 << operator.lut_inputs,
        )
        if tuple(operator.weight.shape) != expected_shape:
            failures.append(
                "Phase 3 independent LUT shape {} != {}".format(
                    tuple(operator.weight.shape), expected_shape
                )
            )

    parsed = training.parse_args([])
    if parsed.pair_lut_parameterization != "independent":
        failures.append("PairLUT parameterization default is not independent")
    if parsed.pair_lut_inputs != 4:
        failures.append("PairLUT input default is not 4")

    forbidden = ("phase3_mode", "pre_bn", "bireal_topology")
    for name in vars(parsed):
        if any(token in name.lower() for token in forbidden):
            failures.append("archived CLI option remains: " + name)
    return failures


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=str(ROOT))
    args = parser.parse_args(argv)
    failures = audit(args.root)
    if failures:
        for failure in failures:
            print("FAIL: " + failure)
        raise SystemExit(1)
    print("PASS: active scripts expose one launcher per Phase 1/2/3")
    print("PASS: Phase 2 is complex Bi-Real")
    print("PASS: Phase 2 uses +/-1 while Phase 3 LUT inputs use exact 0/1")
    print("PASS: Phase 3 uses configurable channel-major complex groups and two LUT outputs")
    print("PASS: each grouped complex LUT has 2^k entries and hard binary forward")


if __name__ == "__main__":
    main()

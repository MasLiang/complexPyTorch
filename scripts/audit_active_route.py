#!/usr/bin/env python3
"""Verify the active Phase 1/2/3 route and retained LUT building blocks."""

import argparse
import sys
from pathlib import Path

import torch.nn as nn


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import training
from complexPyTorch.complexBinaryResNet import BinaryComplexResNet
from complexPyTorch.complexLayers import (
    ComplexLUTConv2d,
    LUTAwareComplexBinaryConv2d,
    PairLUTNeuronConv2d,
)


def audit(root):
    root = Path(root)
    failures = []
    if set(training.PHASE_DESCRIPTIONS) != {1, 2, 3}:
        failures.append("training.PHASE_DESCRIPTIONS is not exactly {1, 2, 3}")

    run_phase_scripts = sorted(
        path.name for path in root.glob("run_phase*.sh")
    )
    if run_phase_scripts != [
        "run_phase1.sh",
        "run_phase2.sh",
        "run_phase3.sh",
        "run_phase3_real_compatible.sh",
    ]:
        failures.append(
            "active phase scripts are {}".format(run_phase_scripts)
        )

    archived_module_names = [
        "directLut5Flow.py",
        "dualLut6Flow.py",
        "fixedAnalyticDominanceFlow.py",
        "mlpFlow.py",
    ]
    for name in archived_module_names:
        if (root / "complexPyTorch" / name).exists():
            failures.append("experimental module remains active: " + name)

    for phase in (1, 2):
        model = BinaryComplexResNet(
            in_channels=3,
            num_blocks=1,
            start_filters=2,
            num_classes=10,
            is_sar_input=False,
            phase=phase,
        )
        lut_modules = [
            module.__class__.__name__
            for module in model.modules()
            if "LUT" in module.__class__.__name__
        ]
        if lut_modules:
            failures.append(
                "Phase {} instantiates LUT modules: {}".format(
                    phase,
                    lut_modules,
                )
            )
    phase3 = BinaryComplexResNet(
        in_channels=3,
        num_blocks=1,
        start_filters=2,
        num_classes=10,
        is_sar_input=False,
        phase=3,
    )
    pair_layers = [
        module
        for module in phase3.modules()
        if isinstance(module, PairLUTNeuronConv2d)
    ]
    if not pair_layers:
        failures.append("Phase 3 instantiates no PairLUTNeuronConv2d layers")
    legacy_active = [
        module.__class__.__name__
        for module in phase3.modules()
        if isinstance(module, (ComplexLUTConv2d, LUTAwareComplexBinaryConv2d))
    ]
    if legacy_active:
        failures.append("Phase 3 instantiates legacy LUT layers: {}".format(legacy_active))
    if any(".conv.conv_" in name for name, _ in phase3.named_parameters()):
        failures.append("Phase 3 pair-LUT path retains spatial convolution weights")
    if not issubclass(ComplexLUTConv2d, nn.Module):
        failures.append("ComplexLUTConv2d is unavailable")
    if not issubclass(LUTAwareComplexBinaryConv2d, nn.Module):
        failures.append("LUTAwareComplexBinaryConv2d is unavailable")
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
    print("PASS: active route exposes only Phase 1/2/3")
    print("PASS: Phase 1/2 models instantiate no LUT modules")
    print("PASS: Phase 3 replaces spatial weights with pair-LUT neurons")
    print("PASS: retained LUT complex layers remain importable")


if __name__ == "__main__":
    main()

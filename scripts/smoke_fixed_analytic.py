#!/usr/bin/env python3
"""Run a compact forward/backward audit of the fixed analytic flow."""

import argparse
import json
from pathlib import Path
import sys

import torch
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from complexPyTorch.fixedAnalyticDominanceFlow import (
    FixedAnalyticBitGradientTracker,
    FixedAnalyticDominanceComplexResNet,
    export_fixed_dominance_lut5,
    fixed_analytic_diagnostics,
)
from training import load_phase_checkpoint


def parse_args(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--checkpoint")
    parser.add_argument("--batch-size", default=2, type=int)
    parser.add_argument("--start-filter", default=4, type=int)
    parser.add_argument("--num-blocks", default=1, type=int)
    parser.add_argument("--dominance-beta", default=2.0, type=float)
    parser.add_argument("--comparator-beta", default=1.0, type=float)
    parser.add_argument("--comparator-scale", default=1.0, type=float)
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    device = torch.device(args.device)
    model = FixedAnalyticDominanceComplexResNet(
        num_blocks=args.num_blocks,
        start_filters=args.start_filter,
        num_classes=10,
        dominance_beta=args.dominance_beta,
        comparator_beta=args.comparator_beta,
        comparator_scale=args.comparator_scale,
    )
    if args.checkpoint:
        load_phase_checkpoint(model, args.checkpoint, expected_phase=1)
    model = model.to(device)
    model.train()
    tracker = FixedAnalyticBitGradientTracker(model)
    data = torch.randn(args.batch_size, 3, 32, 32, device=device)
    target = torch.arange(args.batch_size, device=device) % 10
    output = model(data)
    loss = F.cross_entropy(output, target)
    loss.backward()
    gradients = tracker.finish()
    tracker.close()
    diagnostics = fixed_analytic_diagnostics(model)
    tables = export_fixed_dominance_lut5()

    if not bool(torch.isfinite(output).all() and torch.isfinite(loss)):
        raise RuntimeError("fixed analytic smoke produced non-finite values")
    if diagnostics["lut_modules"] != 0:
        raise RuntimeError("fixed analytic smoke found a LUT module")
    if any(
        values["mean_abs"] <= 0.0
        for values in gradients["bits"].values()
    ):
        raise RuntimeError("one or more semantic bits received zero gradient")
    report = {
        "device": str(device),
        "checkpoint": args.checkpoint,
        "output_shape": list(output.shape),
        "loss": float(loss.item()),
        "finite": True,
        "diagnostics": diagnostics,
        "bit_gradients": gradients,
        "truth_table_real": tables["real"].tolist(),
        "truth_table_imag": tables["imag"].tolist(),
    }
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Run one full-model CUDA forward/backward for a Direct LUT5 encoder."""

import argparse
import json
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch
import torch.nn.functional as F

from complexPyTorch.directLut5Flow import (
    DirectBitGradientTracker,
    DirectLUT5ComplexResNet,
    direct_lut5_bit_names,
    direct_lut5_diagnostics,
    iter_direct_lut5_convs,
)
from training import load_phase_checkpoint


def parse_args(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--encoder-mode", choices=["phase", "dominance"], required=True)
    parser.add_argument("--batch-size", default=2, type=int)
    parser.add_argument("--start-filter", default=11, type=int)
    parser.add_argument("--num-blocks", default=3, type=int)
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the LUT backend smoke test")
    model = DirectLUT5ComplexResNet(
        num_blocks=args.num_blocks,
        start_filters=args.start_filter,
        num_classes=10,
        is_sar_input=False,
        encoder_mode=args.encoder_mode,
    )
    load_phase_checkpoint(model, args.checkpoint, expected_phase=1)
    model = model.cuda().train()
    tracker = DirectBitGradientTracker(model)
    data = torch.randn(args.batch_size, 3, 32, 32, device="cuda")
    target = torch.arange(args.batch_size, device="cuda") % 10
    output = model(data)
    loss = F.cross_entropy(output, target)
    loss.backward()
    torch.cuda.synchronize()
    gradients = tracker.finish(direct_lut5_bit_names(args.encoder_mode))
    tracker.close()

    lut_grad_entries = 0
    lut_grad_nonzero = 0
    for module in iter_direct_lut5_convs(model):
        for parameter in (module.lut_r, module.lut_i):
            if parameter.grad is None:
                continue
            lut_grad_entries += parameter.grad.numel()
            lut_grad_nonzero += int((parameter.grad != 0.0).sum().item())
    result = {
        "encoder_mode": args.encoder_mode,
        "loss": float(loss.item()),
        "output_shape": list(output.shape),
        "finite_output": bool(torch.isfinite(output).all().item()),
        "bit_gradients": gradients,
        "lut_gradient_entries": lut_grad_entries,
        "lut_gradient_nonzero": lut_grad_nonzero,
        "lut_diagnostics": direct_lut5_diagnostics(model),
    }
    print(json.dumps(result, indent=2, sort_keys=True))
    if not result["finite_output"]:
        raise RuntimeError("Direct LUT5 produced non-finite output")
    if lut_grad_nonzero == 0:
        raise RuntimeError("Direct LUT5 tables received no gradient")
    for name, values in gradients["bits"].items():
        if values["mean_abs"] == 0.0:
            raise RuntimeError("Activation bit {} received no gradient".format(name))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

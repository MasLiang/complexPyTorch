#!/usr/bin/env python3
"""Export the fixed dual-LUT6_2 truth table and learned 2-bit thresholds."""

import argparse
import json
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from complexPyTorch.dualLut6Flow import (
    DUAL_LUT6_PHASE,
    dual_lut6_hardware_spec,
)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Export a Phase2.7 checkpoint for dual LUT6_2 hardware"
    )
    parser.add_argument("checkpoint")
    parser.add_argument("-o", "--output")
    return parser.parse_args(argv)


def checkpoint_phase(checkpoint):
    for container in (
        checkpoint,
        checkpoint.get("metrics", {}),
        checkpoint.get("args", {}),
    ):
        if isinstance(container, dict) and container.get("phase") is not None:
            return float(container["phase"])
    return None


def strip_wrappers(key):
    changed = True
    while changed:
        changed = False
        for prefix in ("_orig_mod.", "module."):
            if key.startswith(prefix):
                key = key[len(prefix):]
                changed = True
    return key


def main(argv=None):
    args = parse_args(argv)
    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.is_file():
        raise FileNotFoundError("Checkpoint not found: {}".format(checkpoint_path))
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    phase = checkpoint_phase(checkpoint)
    if phase != DUAL_LUT6_PHASE:
        raise ValueError(
            "{} is Phase {}, not Phase {}".format(
                checkpoint_path,
                phase,
                DUAL_LUT6_PHASE,
            )
        )
    if not checkpoint.get("hardware_deployable", False):
        raise ValueError(
            "{} was saved before rho reached one".format(checkpoint_path)
        )

    checkpoint_args = checkpoint.get("args", {})
    high_ratio = float(checkpoint_args.get("high_ratio", 3.0))
    state = checkpoint.get("model", checkpoint.get("state_dict", checkpoint))
    thresholds = {}
    for key, value in state.items():
        clean_key = strip_wrappers(key)
        if clean_key.endswith("threshold_unconstrained"):
            layer_name = clean_key.rsplit(".", 1)[0]
            thresholds[layer_name] = (
                F.softplus(value.detach()).reshape(-1).tolist()
            )
    if not thresholds:
        raise RuntimeError("Checkpoint contains no learned 2-bit thresholds")

    payload = {
        "phase": DUAL_LUT6_PHASE,
        "source_checkpoint": str(checkpoint_path),
        "epoch": checkpoint.get("epoch"),
        "metrics": checkpoint.get("metrics", {}),
        "two_bit_diagnostics": checkpoint.get("two_bit_diagnostics", {}),
        "two_bit_occupancy": checkpoint.get("two_bit_occupancy"),
        "hardware_spec": dual_lut6_hardware_spec(high_ratio),
        "activation_thresholds": thresholds,
    }
    output = (
        Path(args.output)
        if args.output
        else checkpoint_path.with_suffix(".hardware.json")
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
    print("Exported {}".format(output))
    print(
        "Shared LUT6_2 INIT: {}".format(
            payload["hardware_spec"]["init_hex"]
        )
    )
    print("Activation threshold layers: {}".format(len(thresholds)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

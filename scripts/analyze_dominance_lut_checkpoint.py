#!/usr/bin/env python3
"""Measure how categorical LUT6 checkpoints use the two dominance bits."""

import argparse
import json
from pathlib import Path

import torch


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--summary-only", action="store_true")
    return parser.parse_args()


def checkpoint_state(payload):
    if isinstance(payload, dict):
        for key in ("model", "state_dict"):
            if isinstance(payload.get(key), dict):
                return payload[key]
    return payload


def effective_logits(name, residual, state):
    base_name = name.rsplit(".", 1)[0] + ".dominance_base"
    base = state.get(base_name)
    if not torch.is_tensor(base):
        return residual, None
    alpha_name = name.rsplit(".", 1)[0] + ".dominance_residual_alpha"
    alpha_value = state.get(alpha_name, torch.tensor(1.0))
    alpha = float(alpha_value.detach().cpu())

    state_ids = torch.arange(64)
    old_addresses = (
        ((state_ids >> 5) & 1) * 8
        + ((state_ids >> 4) & 1) * 4
        + ((state_ids >> 2) & 1) * 2
        + ((state_ids >> 1) & 1)
    )
    expanded_base = base.index_select(2, old_addresses)
    assignment = torch.nn.functional.one_hot(
        old_addresses, num_classes=16
    ).to(residual.dtype)
    residual_mean = torch.einsum(
        "ogsc,sa->ogac", residual, assignment
    ) / 4.0
    centered = residual - residual_mean.index_select(2, old_addresses)
    logits = expanded_base + alpha * centered
    metadata = {
        "parameterization": "categorical_residual",
        "alpha": alpha,
        "residual_rms": float(residual.detach().float().square().mean().sqrt()),
        "centered_residual_rms": float(
            centered.detach().float().square().mean().sqrt()
        ),
    }
    return logits, metadata
def bit_sensitivity(categories, address_bit):
    address_mask = 1 << (5 - address_bit)
    low = torch.arange(64)
    low = low[(low & address_mask) == 0]
    high = low | address_mask
    first = categories[..., low]
    second = categories[..., high]
    first_real = torch.div(first, 2, rounding_mode="floor")
    second_real = torch.div(second, 2, rounding_mode="floor")
    return {
        "category_change_fraction": float((first != second).float().mean()),
        "real_bit_change_fraction": float(
            (first_real != second_real).float().mean()
        ),
        "imag_bit_change_fraction": float(
            ((first % 2) != (second % 2)).float().mean()
        ),
    }


def tensor_report(name, logits):
    logits = logits.detach().float().cpu()
    categories = logits.argmax(dim=-1)
    sorted_logits = logits.sort(dim=-1, descending=True).values
    margins = sorted_logits[..., 0] - sorted_logits[..., 1]
    occupancy = torch.bincount(categories.reshape(-1), minlength=4).float()
    occupancy /= occupancy.sum()
    return {
        "name": name,
        "shape": list(logits.shape),
        "class_fractions": [float(value) for value in occupancy],
        "top1_top2_margin": {
            "mean": float(margins.mean()),
            "median": float(margins.median()),
            "min": float(margins.min()),
            "max": float(margins.max()),
        },
        "p0": bit_sensitivity(categories, address_bit=2),
        "p1": bit_sensitivity(categories, address_bit=5),
    }


def aggregate(reports):
    weights = torch.tensor(
        [
            report["shape"][0] * report["shape"][1] * 32
            for report in reports
        ],
        dtype=torch.float64,
    )
    weights /= weights.sum()

    def weighted(path):
        values = []
        for report in reports:
            value = report
            for key in path:
                value = value[key]
            values.append(value)
        return float(torch.dot(weights, torch.tensor(values, dtype=torch.float64)))

    return {
        "layers": len(reports),
        "p0_category_change_fraction": weighted(
            ("p0", "category_change_fraction")
        ),
        "p1_category_change_fraction": weighted(
            ("p1", "category_change_fraction")
        ),
        "p0_real_bit_change_fraction": weighted(
            ("p0", "real_bit_change_fraction")
        ),
        "p0_imag_bit_change_fraction": weighted(
            ("p0", "imag_bit_change_fraction")
        ),
        "p1_real_bit_change_fraction": weighted(
            ("p1", "real_bit_change_fraction")
        ),
        "p1_imag_bit_change_fraction": weighted(
            ("p1", "imag_bit_change_fraction")
        ),
        "random_four_class_reference": {
            "category_change_fraction": 0.75,
            "single_output_bit_change_fraction": 0.5,
        },
    }


def main():
    args = parse_args()
    payload = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    state = checkpoint_state(payload)
    reports = []
    for name, value in state.items():
        if not name.endswith(".conv.weight") or not torch.is_tensor(value):
            continue
        is_slice_residual = (
            value.ndim == 4 and tuple(value.shape[-2:]) == (64, 4)
        )
        if not is_slice_residual:
            continue
        logits, metadata = effective_logits(name, value, state)
        report = tensor_report(name, logits)
        if metadata is not None:
            report.update(metadata)
        reports.append(report)
    if not reports:
        raise RuntimeError("No categorical residual LUT6 tensors found")

    result = {
        "checkpoint": str(args.checkpoint),
        "metrics": payload.get("metrics", {}) if isinstance(payload, dict) else {},
        "aggregate": aggregate(reports),
        "layers": [] if args.summary_only else reports,
    }
    rendered = json.dumps(result, indent=2, sort_keys=True)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)


if __name__ == "__main__":
    main()

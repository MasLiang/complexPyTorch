#!/usr/bin/env python3
"""Summarize hard-table learning and fifth-bit use in a LUT5 checkpoint."""

import argparse
import json
import math
from pathlib import Path

import torch


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument(
        "--json-output",
        type=Path,
        help="Optional path for the same machine-readable JSON report",
    )
    parser.add_argument(
        "--fail-if-no-d-dependence",
        action="store_true",
        help="Exit with status 2 when every hard LUT output ignores d",
    )
    return parser.parse_args()


def unwrap_checkpoint(checkpoint):
    if not isinstance(checkpoint, dict):
        return checkpoint, {}, {}
    state = checkpoint.get(
        "model",
        checkpoint.get("state_dict", checkpoint),
    )
    args = checkpoint.get("args")
    metrics = checkpoint.get("metrics")
    return (
        state,
        args if isinstance(args, dict) else {},
        metrics if isinstance(metrics, dict) else {},
    )


def initial_lut4_table(output_name, init_mode, logit_abs):
    state = torch.arange(16)
    xr = (((state >> 3) & 1).float() * 2.0) - 1.0
    xi = (((state >> 2) & 1).float() * 2.0) - 1.0
    wr = (((state >> 1) & 1).float() * 2.0) - 1.0
    wi = ((state & 1).float() * 2.0) - 1.0
    raw = xr * wr - xi * wi
    if output_name == "imag":
        raw = xr * wi + xi * wr

    if init_mode == "raw":
        table4 = raw * (logit_abs / 2.0)
    else:
        table4 = torch.where(
            raw >= 0,
            torch.full_like(raw, logit_abs),
            torch.full_like(raw, -logit_abs),
        )

    state5 = torch.arange(32)
    state4 = ((state5 & 0b11000) >> 1) | (state5 & 0b00011)
    return table4[state4]


def c8_unit_codebook(mode):
    if mode == "roots":
        inv_sqrt2 = 1.0 / math.sqrt(2.0)
        return torch.tensor(
            [
                [-inv_sqrt2, -inv_sqrt2],
                [-1.0, 0.0],
                [-inv_sqrt2, inv_sqrt2],
                [0.0, 1.0],
                [inv_sqrt2, inv_sqrt2],
                [1.0, 0.0],
                [inv_sqrt2, -inv_sqrt2],
                [0.0, -1.0],
            ]
        )
    if mode == "octants":
        high = math.cos(math.pi / 8.0)
        low = math.sin(math.pi / 8.0)
        return torch.tensor(
            [
                [-high, -low],
                [-high, low],
                [-low, high],
                [low, high],
                [high, low],
                [high, -low],
                [low, -high],
                [-low, -high],
            ]
        )
    raise ValueError("Unknown C8 codebook mode: {}".format(mode))


def initial_c8_product_table(
    output_name,
    codebook_mode,
    tau,
    semantic_address=False,
):
    state = torch.arange(32)
    activation_address = state >> 2
    wr = (((state >> 1) & 1).float() * 2.0) - 1.0
    wi = ((state & 1).float() * 2.0) - 1.0

    if semantic_address:
        if codebook_mode != "octants":
            raise ValueError("semantic C8 product requires octants")
        sign_r = (((activation_address >> 2) & 1).float() * 2.0) - 1.0
        sign_i = (((activation_address >> 1) & 1).float() * 2.0) - 1.0
        dominance = (activation_address & 1).float()
        codebook = c8_unit_codebook("octants")
        high = codebook.abs().amax()
        low = codebook.abs().amin()
        magnitude_r = low + (high - low) * dominance
        magnitude_i = high - (high - low) * dominance
        decoded = torch.stack(
            (sign_r * magnitude_r, sign_i * magnitude_i),
            dim=1,
        )
    else:
        phase_index = torch.arange(8, dtype=torch.long)
        gray_address = phase_index ^ (phase_index >> 1)
        address_to_phase = torch.empty(8, dtype=torch.long)
        address_to_phase[gray_address] = phase_index
        decoded = c8_unit_codebook(codebook_mode)[
            address_to_phase[activation_address]
        ]
    raw = decoded[:, 0] * wr - decoded[:, 1] * wi
    if output_name == "imag":
        raw = decoded[:, 0] * wi + decoded[:, 1] * wr
    epsilon = 1e-6
    return torch.atanh(
        (raw / 2.0).clamp(-1.0 + epsilon, 1.0 - epsilon)
    ) / tau


def initial_table(
    output_name,
    strategy,
    init_mode,
    logit_abs,
    c8_codebook_mode,
    init_tau,
):
    if strategy in ("c8_product", "semantic_c8_product"):
        return initial_c8_product_table(
            output_name,
            c8_codebook_mode,
            init_tau,
            semantic_address=(strategy == "semantic_c8_product"),
        )
    return initial_lut4_table(output_name, init_mode, logit_abs)


def summarize_table(table, initial):
    table = table.detach().float().cpu()
    if table.ndim == 1:
        table = table.unsqueeze(0)
    if table.ndim != 2 or table.shape[1] != 32:
        raise ValueError(
            "Expected a [sets, 32] LUT5 table, got {}".format(
                tuple(table.shape)
            )
        )

    state = torch.arange(32)
    slice0 = state[((state >> 2) & 1) == 0]
    slice1 = slice0 | 0b00100
    left = table.index_select(1, slice0)
    right = table.index_select(1, slice1)
    slice_delta = (right - left).abs()
    hard_slice_diff = (right >= 0) != (left >= 0)

    initial = initial.unsqueeze(0).expand_as(table)
    init_delta = (table - initial).abs()
    init_sign_diff = (table >= 0) != (initial >= 0)

    return {
        "sets": int(table.shape[0]),
        "entries": int(table.numel()),
        "positive_entries": int((table >= 0).sum().item()),
        "near_zero_entries": int((table.abs() < 0.05).sum().item()),
        "init_sign_diff": int(init_sign_diff.sum().item()),
        "init_sign_diff_ratio": float(init_sign_diff.float().mean().item()),
        "init_abs_delta_mean": float(init_delta.mean().item()),
        "init_abs_delta_max": float(init_delta.max().item()),
        "d_slice_pairs": int(slice_delta.numel()),
        "d_slice_hard_diff": int(hard_slice_diff.sum().item()),
        "d_slice_hard_diff_ratio": float(
            hard_slice_diff.float().mean().item()
        ),
        "d_slice_soft_abs_diff_mean": float(slice_delta.mean().item()),
        "d_slice_soft_abs_diff_max": float(slice_delta.max().item()),
    }


def aggregate(reports):
    keys = (
        "entries",
        "positive_entries",
        "near_zero_entries",
        "init_sign_diff",
        "d_slice_pairs",
        "d_slice_hard_diff",
    )
    result = {key: sum(item[key] for item in reports) for key in keys}
    result["tables"] = len(reports)
    result["init_sign_diff_ratio"] = (
        result["init_sign_diff"] / float(result["entries"])
    )
    result["d_slice_hard_diff_ratio"] = (
        result["d_slice_hard_diff"] / float(result["d_slice_pairs"])
    )
    result["d_slice_soft_abs_diff_mean"] = sum(
        item["d_slice_soft_abs_diff_mean"] * item["d_slice_pairs"]
        for item in reports
    ) / float(result["d_slice_pairs"])
    result["d_slice_soft_abs_diff_max"] = max(
        item["d_slice_soft_abs_diff_max"] for item in reports
    )
    return result


def neural_lut_tables(state, residual_scale, pure_mlp=False):
    addresses = torch.arange(32).unsqueeze(1)
    shifts = torch.arange(4, -1, -1).unsqueeze(0)
    addresses = (
        ((addresses >> shifts) & 1).float().mul_(2.0).sub_(1.0)
    )
    tables = []
    suffix = "lut_generator_hidden_weight"
    for key, hidden_weight in state.items():
        if not key.endswith(suffix):
            continue
        prefix = key[: -len(suffix)]
        required = {
            "hidden_bias": prefix + "lut_generator_hidden_bias",
            "output_weight": prefix + "lut_generator_output_weight",
            "output_bias": prefix + "lut_generator_output_bias",
        }
        if not pure_mlp:
            required.update(
                {
                    "lut_r": prefix + "lut_r",
                    "lut_i": prefix + "lut_i",
                }
            )
        missing = [
            name for name, state_key in required.items()
            if state_key not in state
        ]
        if missing:
            raise RuntimeError(
                "Neural LUT module {} is missing {}".format(
                    prefix.rstrip("."),
                    missing,
                )
            )
        hidden = torch.tanh(
            torch.einsum(
                "shd,nd->snh",
                hidden_weight.float(),
                addresses,
            )
            + state[required["hidden_bias"]].float().unsqueeze(1)
        )
        generated = (
            torch.einsum(
                "soh,snh->sno",
                state[required["output_weight"]].float(),
                hidden,
            )
            + state[required["output_bias"]].float().unsqueeze(1)
        )
        if pure_mlp:
            effective_r = generated[:, :, 0]
            effective_i = generated[:, :, 1]
        else:
            residual = generated * residual_scale
            effective_r = state[required["lut_r"]].float() + residual[:, :, 0]
            effective_i = state[required["lut_i"]].float() + residual[:, :, 1]
        tables.extend(
            [
                (
                    prefix + "effective_lut_r",
                    "real",
                    effective_r,
                ),
                (
                    prefix + "effective_lut_i",
                    "imag",
                    effective_i,
                ),
            ]
        )
    return tables


def build_report(checkpoint_path):
    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=False,
    )
    state, saved_args, metrics = unwrap_checkpoint(checkpoint)
    init_mode = saved_args.get("lut_init_mode", "raw")
    logit_abs = abs(float(saved_args.get("lut_logit_init", 2.0)))
    phase3p1_mode = saved_args.get("phase3p1_mode", "fixed")
    init_strategy = (
        "semantic_c8_product"
        if phase3p1_mode == "semantic_lut5"
        else (
            "c8_product"
            if phase3p1_mode in (
                "learned_lut5",
                "neural_lut5",
                "pure_mlp",
            )
            else "duplicate_lut4"
        )
    )
    c8_codebook_mode = saved_args.get("c8_codebook", "octants")
    init_tau = float(saved_args.get("lut_tau_min", 1.0))

    parameterization = (
        "pure_mlp"
        if phase3p1_mode == "pure_mlp"
        else (
            "neural_residual"
            if phase3p1_mode == "neural_lut5"
            else "direct_entries"
        )
    )
    if phase3p1_mode in ("neural_lut5", "pure_mlp"):
        table_values = neural_lut_tables(
            state,
            float(saved_args.get("neural_lut_residual_scale", 1.0)),
            pure_mlp=(phase3p1_mode == "pure_mlp"),
        )
    else:
        table_values = []
        for key, value in state.items():
            if not torch.is_tensor(value):
                continue
            if key.endswith(".lut_r") or key == "lut_r":
                output_name = "real"
            elif key.endswith(".lut_i") or key == "lut_i":
                output_name = "imag"
            else:
                continue
            if value.ndim not in (1, 2) or value.shape[-1] != 32:
                continue
            table_values.append((key, output_name, value))

    tables = []
    for key, output_name, value in table_values:
        baseline = initial_table(
            output_name,
            init_strategy,
            init_mode,
            logit_abs,
            c8_codebook_mode,
            init_tau,
        )
        table_report = summarize_table(
            value,
            baseline,
        )
        table_report["name"] = key
        table_report["output"] = output_name
        tables.append(table_report)

    if not tables:
        raise RuntimeError(
            "Checkpoint contains no reconstructable 32-entry LUT5 tables"
        )

    hard_values = [
        float(value.item())
        for key, value in state.items()
        if key.endswith(".hard")
        and torch.is_tensor(value)
        and value.numel() == 1
    ]
    summary = aggregate(tables)
    summary["hard_d_dependence"] = summary["d_slice_hard_diff"] > 0
    summary["hard_activation_lsb_dependence"] = (
        summary["d_slice_hard_diff"] > 0
    )
    slice_bit = (
        "dominance"
        if init_strategy == "semantic_c8_product"
        else "c8_gray_bit_0"
        if init_strategy == "c8_product"
        else "dominance"
    )
    return {
        "checkpoint": str(checkpoint_path),
        "phase": checkpoint.get("phase") if isinstance(checkpoint, dict) else None,
        "epoch": checkpoint.get("epoch") if isinstance(checkpoint, dict) else None,
        "phase2p1_mode": saved_args.get("phase2p1_mode"),
        "phase3p1_mode": phase3p1_mode,
        "parameterization": parameterization,
        "neural_lut_hidden": saved_args.get("neural_lut_hidden"),
        "neural_lut_residual_scale": saved_args.get(
            "neural_lut_residual_scale"
        ),
        "lut5_init_strategy": init_strategy,
        "slice_bit": slice_bit,
        "lut_init_mode": init_mode,
        "lut_logit_init": logit_abs,
        "lut_init_tau": init_tau,
        "c8_codebook": c8_codebook_mode,
        "hard_ratio_min": min(hard_values) if hard_values else None,
        "hard_ratio_max": max(hard_values) if hard_values else None,
        "checkpoint_hard_mode": metrics.get("hard_mode"),
        "explicit_hard_table_modules": (
            len(checkpoint.get("hard_lut_tables", {}))
            if isinstance(checkpoint, dict)
            and isinstance(checkpoint.get("hard_lut_tables"), dict)
            else 0
        ),
        "summary": summary,
        "tables": tables,
    }


def main():
    args = parse_args()
    report = build_report(args.checkpoint)
    rendered = json.dumps(report, indent=2, sort_keys=True)
    print(rendered)
    if args.json_output is not None:
        args.json_output.parent.mkdir(parents=True, exist_ok=True)
        args.json_output.write_text(rendered + "\n", encoding="utf-8")
    if (
        args.fail_if_no_d_dependence
        and not report["summary"]["hard_d_dependence"]
    ):
        raise SystemExit(2)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Analyze learned Boolean structure in Phase 3 pair-LUT checkpoints."""

import argparse
import json
from pathlib import Path

import torch


PARTS = ("r", "i")


def _quantiles(values):
    probabilities = torch.tensor(
        [0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0],
        dtype=torch.float64,
    )
    result = torch.quantile(values.to(torch.float64), probabilities)
    return {
        str(float(probability)): float(value)
        for probability, value in zip(probabilities, result)
    }


def _table_metrics(logits, initial_sign=None):
    if logits.ndim != 3 or logits.shape[1] != 16:
        raise ValueError(
            "Expected pair-LUT logits shaped [pairs, 16, outputs], got {}".format(
                tuple(logits.shape)
            )
        )
    hard = logits >= 0.0
    functions = hard.permute(0, 2, 1).reshape(-1, 16)
    constant = (functions == functions[:, :1]).all(dim=1)
    powers = (1 << torch.arange(16, dtype=torch.int64)).view(1, 16)
    function_ids = (functions.to(torch.int64) * powers).sum(dim=1)

    sensitivity = []
    state_ids = torch.arange(16, dtype=torch.int64)
    for shift in (3, 2, 1, 0):
        low = state_ids[((state_ids >> shift) & 1) == 0]
        high = low | (1 << shift)
        changed = functions[:, low] != functions[:, high]
        sensitivity.append(float(changed.to(torch.float64).mean()))

    result = {
        "entries": int(hard.numel()),
        "functions": int(functions.shape[0]),
        "unique_functions": int(torch.unique(function_ids).numel()),
        "ones_fraction": float(hard.to(torch.float64).mean()),
        "constant_function_fraction": float(constant.to(torch.float64).mean()),
        "mean_boolean_sensitivity": float(sum(sensitivity) / len(sensitivity)),
        "sensitivity_by_input_bit": sensitivity,
        "absolute_logit_quantiles": _quantiles(logits.abs().reshape(-1)),
    }
    if initial_sign is not None:
        if initial_sign.shape != hard.shape:
            raise ValueError(
                "Initial sign shape {} does not match logits {}".format(
                    tuple(initial_sign.shape), tuple(hard.shape)
                )
            )
        sign_diff = hard != initial_sign.to(torch.bool)
        result["sign_diff"] = int(sign_diff.sum())
        result["sign_diff_fraction"] = float(
            sign_diff.to(torch.float64).mean()
        )
    return result, hard


def _state_dict(payload):
    if isinstance(payload, dict) and isinstance(payload.get("model"), dict):
        return payload["model"]
    if isinstance(payload, dict) and isinstance(payload.get("state_dict"), dict):
        return payload["state_dict"]
    if isinstance(payload, dict):
        return payload
    raise TypeError("Checkpoint does not contain a state dictionary")


def summarize(checkpoint_path):
    checkpoint_path = Path(checkpoint_path)
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state = _state_dict(payload)
    layer_prefixes = sorted(
        key[: -len(".lut_r")]
        for key in state
        if key.endswith(".lut_r")
        and not key.endswith("initial_lut_r_sign")
    )
    if not layer_prefixes:
        raise RuntimeError("No pair-LUT logits found in {}".format(checkpoint_path))

    layers = []
    totals = {
        "entries": 0,
        "ones": 0,
        "sign_diff": 0,
        "functions": 0,
        "constant_functions": 0,
        "sensitivity_numerator": [0.0, 0.0, 0.0, 0.0],
        "real_imag_differences": 0,
        "real_imag_entries": 0,
    }
    all_abs_logits = []

    for prefix in layer_prefixes:
        layer = {"name": prefix, "parts": {}}
        hard_parts = {}
        for part in PARTS:
            logits_key = "{}.lut_{}".format(prefix, part)
            initial_key = "{}.initial_lut_{}_sign".format(prefix, part)
            logits = state[logits_key].detach().cpu()
            initial = state.get(initial_key)
            if initial is not None:
                initial = initial.detach().cpu()
            metrics, hard = _table_metrics(logits, initial)
            layer["parts"][part] = metrics
            hard_parts[part] = hard
            all_abs_logits.append(logits.abs().reshape(-1))

            totals["entries"] += metrics["entries"]
            totals["ones"] += int(hard.sum())
            totals["sign_diff"] += metrics.get("sign_diff", 0)
            totals["functions"] += metrics["functions"]
            totals["constant_functions"] += round(
                metrics["constant_function_fraction"] * metrics["functions"]
            )
            for index, value in enumerate(metrics["sensitivity_by_input_bit"]):
                totals["sensitivity_numerator"][index] += (
                    value * metrics["functions"]
                )

        pair_difference = hard_parts["r"] != hard_parts["i"]
        layer["real_imag_hamming_fraction"] = float(
            pair_difference.to(torch.float64).mean()
        )
        totals["real_imag_differences"] += int(pair_difference.sum())
        totals["real_imag_entries"] += pair_difference.numel()
        layers.append(layer)

    sensitivity = [
        value / totals["functions"]
        for value in totals["sensitivity_numerator"]
    ]
    aggregate = {
        "layers": len(layers),
        "entries": totals["entries"],
        "functions": totals["functions"],
        "ones_fraction": totals["ones"] / totals["entries"],
        "constant_function_fraction": (
            totals["constant_functions"] / totals["functions"]
        ),
        "mean_boolean_sensitivity": sum(sensitivity) / len(sensitivity),
        "sensitivity_by_input_bit": sensitivity,
        "real_imag_hamming_fraction": (
            totals["real_imag_differences"] / totals["real_imag_entries"]
        ),
        "sign_diff": totals["sign_diff"],
        "sign_diff_fraction": totals["sign_diff"] / totals["entries"],
        "absolute_logit_quantiles": _quantiles(torch.cat(all_abs_logits)),
    }
    return {
        "checkpoint": str(checkpoint_path),
        "aggregate": aggregate,
        "layers": layers,
    }


def to_markdown(result, include_layers=False):
    aggregate = result["aggregate"]
    lines = [
        "# Pair-LUT Checkpoint Analysis",
        "",
        "- Checkpoint: `{}`".format(result["checkpoint"]),
        "- Layers: `{}`; scalar Boolean functions: `{}`".format(
            aggregate["layers"], aggregate["functions"]
        ),
        "- Hard ones: `{:.4f}`".format(aggregate["ones_fraction"]),
        "- Constant functions: `{:.4f}`".format(
            aggregate["constant_function_fraction"]
        ),
        "- Mean Boolean sensitivity: `{:.4f}`; bits: `{}`".format(
            aggregate["mean_boolean_sensitivity"],
            ", ".join(
                "{:.4f}".format(value)
                for value in aggregate["sensitivity_by_input_bit"]
            ),
        ),
        "- Real/imag table Hamming distance: `{:.4f}`".format(
            aggregate["real_imag_hamming_fraction"]
        ),
        "- Sign changes from initialization: `{}/{} ({:.4f})`".format(
            aggregate["sign_diff"],
            aggregate["entries"],
            aggregate["sign_diff_fraction"],
        ),
        "- Absolute-logit quantiles: `{}`".format(
            json.dumps(aggregate["absolute_logit_quantiles"], sort_keys=True)
        ),
    ]
    if include_layers:
        lines.extend(["", "## Layers"])
        for layer in result["layers"]:
            lines.append(
                "- `{}`: sensitivity r/i `{:.4f}/{:.4f}`, constants r/i "
                "`{:.4f}/{:.4f}`, r-i Hamming `{:.4f}`".format(
                    layer["name"],
                    layer["parts"]["r"]["mean_boolean_sensitivity"],
                    layer["parts"]["i"]["mean_boolean_sensitivity"],
                    layer["parts"]["r"]["constant_function_fraction"],
                    layer["parts"]["i"]["constant_function_fraction"],
                    layer["real_imag_hamming_fraction"],
                )
            )
    return "\n".join(lines) + "\n"


def parse_args(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoints", nargs="+")
    parser.add_argument("--format", choices=("markdown", "json"), default="markdown")
    parser.add_argument("--include-layers", action="store_true")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    results = [summarize(path) for path in args.checkpoints]
    if args.format == "json":
        print(json.dumps(results, indent=2, sort_keys=True))
        return
    for index, result in enumerate(results):
        if index:
            print()
        print(to_markdown(result, include_layers=args.include_layers), end="")


if __name__ == "__main__":
    main()

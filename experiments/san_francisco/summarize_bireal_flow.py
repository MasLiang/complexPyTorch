#!/usr/bin/env python3
"""Summarize and compare completed San Francisco BiReal/LUT flows."""

import argparse
import json
from pathlib import Path


STAGES = (
    "bireal_fp",
    "bireal",
    "bireal_lut",
    "lut6_residual",
    "lut6_lut4_residual",
)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--flow-root", required=True)
    parser.add_argument("--reference-root")
    parser.add_argument("--output")
    return parser.parse_args(argv)


def load_flow(root):
    root = Path(root)
    stages = {}
    expected_support = None
    for stage in STAGES:
        stage_root = root / stage
        metrics_path = stage_root / "test_metrics.json"
        history_path = stage_root / "history.json"
        if not metrics_path.is_file() or not history_path.is_file():
            raise FileNotFoundError("Incomplete stage: {}".format(stage_root))
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        history = json.loads(history_path.read_text(encoding="utf-8"))
        eligible = [
            record for record in history
            if record.get("selection_eligible", True)
        ]
        if not eligible:
            raise ValueError("No checkpoint-eligible epochs in {}".format(stage_root))
        best = max(eligible, key=lambda record: record["val"]["oa"])
        support = [sum(row) for row in metrics["confusion_matrix"]]
        if expected_support is None:
            expected_support = support
        elif support != expected_support:
            raise ValueError("Test class support changed at {}".format(stage_root))
        stages[stage] = {
            "test_oa": metrics["oa"],
            "test_aa": metrics["aa"],
            "per_class_accuracy": metrics["per_class_accuracy"],
            "best_val_oa": best["val"]["oa"],
            "best_val_aa": best["val"]["aa"],
            "best_val_epoch": best["epoch"],
        }
    return {"stages": stages, "class_support": expected_support}


def transition_delta(flow, start, end):
    before = flow["stages"][start]
    after = flow["stages"][end]
    return {
        "oa": after["test_oa"] - before["test_oa"],
        "aa": after["test_aa"] - before["test_aa"],
        "per_class": [
            new - old
            for old, new in zip(
                before["per_class_accuracy"], after["per_class_accuracy"]
            )
        ],
    }


def percentage_points(value):
    return "{:+.2f}".format(100.0 * value)


def main(argv=None):
    args = parse_args(argv)
    root = Path(args.flow_root)
    flow = load_flow(root)
    reference = load_flow(args.reference_root) if args.reference_root else None
    lut6_gain = transition_delta(flow, "bireal_lut", "lut6_residual")
    payload = {
        "flow_root": str(root),
        **flow,
        "lut6_over_lut4": lut6_gain,
    }
    if reference is not None:
        payload["reference_root"] = str(Path(args.reference_root))
        payload["reference"] = reference
        payload["reference_lut6_over_lut4"] = transition_delta(
            reference, "bireal_lut", "lut6_residual"
        )

    lines = [
        "# San Francisco BiReal/LUT flow summary",
        "",
        "| Stage | Test OA | Test AA | Best eligible validation OA | Epoch |",
        "|---|---:|---:|---:|---:|",
    ]
    for stage in STAGES:
        item = flow["stages"][stage]
        lines.append(
            "| {} | {:.2%} | {:.2%} | {:.2%} | {} |".format(
                stage,
                item["test_oa"],
                item["test_aa"],
                item["best_val_oa"],
                item["best_val_epoch"],
            )
        )

    total = sum(flow["class_support"])
    lines.extend([
        "",
        "## Test class support",
        "",
        "| Class | Samples | Share |",
        "|---|---:|---:|",
    ])
    for index, support in enumerate(flow["class_support"], start=1):
        lines.append("| {} | {} | {:.2%} |".format(index, support, support / total))

    lines.extend([
        "",
        "## Per-class test accuracy",
        "",
        "| Stage | Class 1 | Class 2 | Class 3 | Class 4 | Class 5 |",
        "|---|---:|---:|---:|---:|---:|",
    ])
    for stage in STAGES:
        values = flow["stages"][stage]["per_class_accuracy"]
        lines.append(
            "| {} | {} |".format(
                stage, " | ".join("{:.2%}".format(value) for value in values)
            )
        )

    lines.extend([
        "",
        "## LUT4 to LUT6 residual delta",
        "",
        "| Flow | OA pp | AA pp | Class 1 pp | Class 2 pp | Class 3 pp | Class 4 pp | Class 5 pp |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ])
    deltas = [("current", lut6_gain)]
    if reference is not None:
        deltas.append(("reference", payload["reference_lut6_over_lut4"]))
    for name, delta in deltas:
        values = [delta["oa"], delta["aa"], *delta["per_class"]]
        lines.append(
            "| {} | {} |".format(
                name, " | ".join(percentage_points(value) for value in values)
            )
        )

    if reference is not None:
        lines.extend([
            "",
            "## Current minus reference flow",
            "",
            "| Stage | Test OA pp | Test AA pp |",
            "|---|---:|---:|",
        ])
        for stage in STAGES:
            current = flow["stages"][stage]
            old = reference["stages"][stage]
            lines.append(
                "| {} | {} | {} |".format(
                    stage,
                    percentage_points(current["test_oa"] - old["test_oa"]),
                    percentage_points(current["test_aa"] - old["test_aa"]),
                )
            )

    output = Path(args.output) if args.output else root / "flow_summary.md"
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    output.with_suffix(".json").write_text(
        json.dumps(payload, indent=2) + "\n", encoding="utf-8"
    )
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

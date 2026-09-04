#!/usr/bin/env python3
"""Summarize repeated San Francisco FP/BiReal/LUT core flows."""

import argparse
import json
from pathlib import Path
import statistics


STAGES = (
    ("fp_baseline", "fp_baseline"),
    ("bireal_fp", "bireal_flow/bireal_fp"),
    ("bireal", "bireal_flow/bireal"),
    ("lut4", "bireal_flow/bireal_lut"),
    ("lut6", "bireal_flow/lut6_residual"),
)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--flow-roots", nargs="+", required=True)
    parser.add_argument("--labels", nargs="+")
    parser.add_argument("--output", required=True)
    return parser.parse_args(argv)


def load_flow(root):
    result = {}
    for stage, relative in STAGES:
        path = root / relative / "test_metrics.json"
        if not path.is_file():
            raise FileNotFoundError("Incomplete flow; missing {}".format(path))
        metrics = json.loads(path.read_text(encoding="utf-8"))
        result[stage] = {
            "oa": metrics["oa"],
            "aa": metrics["aa"],
            "per_class_accuracy": metrics["per_class_accuracy"],
        }
    return result


def mean_std(values):
    mean = statistics.fmean(values)
    std = statistics.stdev(values) if len(values) > 1 else 0.0
    return mean, std


def main(argv=None):
    args = parse_args(argv)
    roots = [Path(value) for value in args.flow_roots]
    labels = args.labels or [root.name for root in roots]
    if len(labels) != len(roots):
        raise ValueError("--labels must match --flow-roots")
    flows = {label: load_flow(root) for label, root in zip(labels, roots)}

    lines = [
        "# San Francisco Multi-Seed Core-Flow Results",
        "",
        "All rows report final test OA/AA from the checkpoint selected by "
        "validation OA. The intended reproducibility protocol fixes the "
        "spatial split and varies only the training seed.",
        "",
        "## Per-seed test results",
        "",
        "| Seed | Independent FP OA/AA | Shared FP OA/AA | BiReal OA/AA | LUT4 OA/AA | LUT6 residual OA/AA | LUT6-LUT4 OA pp |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for label, flow in flows.items():
        lines.append(
            "| {} | {:.2%}/{:.2%} | {:.2%}/{:.2%} | {:.2%}/{:.2%} | "
            "{:.2%}/{:.2%} | {:+.2f} |".format(
                label,
                flow["fp_baseline"]["oa"],
                flow["fp_baseline"]["aa"],
                flow["bireal_fp"]["oa"],
                flow["bireal_fp"]["aa"],
                flow["bireal"]["oa"],
                flow["bireal"]["aa"],
                flow["lut4"]["oa"],
                flow["lut4"]["aa"],
                flow["lut6"]["oa"],
                flow["lut6"]["aa"],
                100.0 * (flow["lut6"]["oa"] - flow["lut4"]["oa"]),
            )
        )

    aggregate = {}
    lines.extend([
        "",
        "## Mean and sample standard deviation",
        "",
        "| Stage | OA | AA |",
        "|---|---:|---:|",
    ])
    for stage, _ in STAGES:
        oa_mean, oa_std = mean_std([flow[stage]["oa"] for flow in flows.values()])
        aa_mean, aa_std = mean_std([flow[stage]["aa"] for flow in flows.values()])
        aggregate[stage] = {
            "oa_mean": oa_mean,
            "oa_std": oa_std,
            "aa_mean": aa_mean,
            "aa_std": aa_std,
        }
        lines.append(
            "| {} | {:.2%} +/- {:.2%} | {:.2%} +/- {:.2%} |".format(
                stage, oa_mean, oa_std, aa_mean, aa_std
            )
        )

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    output.with_suffix(".json").write_text(
        json.dumps(
            {
                "roots": [str(root) for root in roots],
                "flows": flows,
                "aggregate": aggregate,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

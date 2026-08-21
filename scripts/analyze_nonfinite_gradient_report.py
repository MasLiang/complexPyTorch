#!/usr/bin/env python3
"""Summarize a non-finite gradient report emitted by training.py."""

import argparse
import json
from pathlib import Path


def _gradient_names(report):
    return [item["name"] for item in report.get("nonfinite_gradients", [])]


def _boundary_rows(report):
    hooks = report.get("backward_hooks", {})
    rows = []
    for record in hooks.get("records", []):
        outputs = record.get("grad_output", [])
        if not outputs:
            continue
        stats = outputs[0]
        rows.append(
            {
                "sequence": record.get("sequence"),
                "module": record.get("module"),
                "finite": stats.get("finite", 0),
                "nan": stats.get("nan", 0),
                "inf": stats.get("inf", 0),
                "total": stats.get("total", 0),
                "finite_max_abs": stats.get("finite_max_abs"),
            }
        )
    return sorted(rows, key=lambda row: row["sequence"])


def summarize(report):
    return {
        "epoch": report.get("epoch"),
        "batch_index": report.get("batch_index"),
        "loss": report.get("loss"),
        "output_max_abs": report.get("output_max_abs"),
        "diagnosis": report.get("diagnosis"),
        "nonfinite_gradient_tensor_count": report.get(
            "nonfinite_gradient_tensor_count", 0
        ),
        "nonfinite_gradient_names": _gradient_names(report),
        "block_boundaries": _boundary_rows(report),
    }


def print_text(summary):
    print(
        "epoch={epoch} batch={batch_index} diagnosis={diagnosis} "
        "loss={loss:.6g} output_max_abs={output_max_abs:.6g}".format(
            **summary
        )
    )
    print(
        "nonfinite parameter gradients: {}".format(
            summary["nonfinite_gradient_tensor_count"]
        )
    )
    for name in summary["nonfinite_gradient_names"]:
        print("  {}".format(name))
    print("block output-gradient boundaries (backward order):")
    for row in summary["block_boundaries"]:
        print(
            "  {sequence:>2} {module:<12} finite={finite}/{total} "
            "nan={nan} inf={inf} max_abs={finite_max_abs}".format(**row)
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("report", type=Path)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    report = json.loads(args.report.read_text(encoding="utf-8"))
    summary = summarize(report)
    if args.json:
        print(json.dumps(summary, indent=2, sort_keys=True))
    else:
        print_text(summary)


if __name__ == "__main__":
    main()

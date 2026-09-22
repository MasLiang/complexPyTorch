#!/usr/bin/env python3
"""Generate the paper-facing Pol-InSAR/OpenSARShip result table from unified runs."""

import argparse
import json
from pathlib import Path


STAGES = ("fp", "bireal", "lut4", "lut6_residual")


def load_stage(root, stage):
    path = root / stage / "metrics.json"
    return None if not path.is_file() else json.loads(path.read_text(encoding="utf-8"))


def value(metrics, key):
    return "-" if metrics is None else "{:.2f}".format(100.0 * metrics[key])


def protocol(root):
    path = root / "protocol.json"
    return {} if not path.is_file() else json.loads(path.read_text(encoding="utf-8"))


def section(label, root):
    payload = protocol(root)
    rows = {stage: load_stage(root, stage) for stage in STAGES}
    lines = ["## {}".format(label), "", "Protocol: `{}`.".format(payload.get("dataset_protocol", "pending")), "", "| Model | Val OA | Val AA | Test OA | Test AA | Macro F1 |", "|---|---:|---:|---:|---:|---:|"]
    for stage, title in (("fp", "FP"), ("bireal", "BiReal"), ("lut4", "LUT4"), ("lut6_residual", "LUT6 residual")):
        metrics = rows[stage]
        selection = None
        history = root / stage / "history.json"
        if history.is_file():
            values = json.loads(history.read_text(encoding="utf-8"))
            selection = max((item.get("selection") for item in values if item.get("selection")), key=lambda item: item["oa"], default=None)
        lines.append("| {} | {} | {} | {} | {} | {} |".format(title, value(selection, "oa"), value(selection, "aa"), value(metrics, "oa"), value(metrics, "aa"), value(metrics, "macro_f1")))
    bireal, lut4, lut6 = rows["bireal"], rows["lut4"], rows["lut6_residual"]
    lines.extend(["", "Relative test-OA/AA changes (percentage points):", "", "| Change | OA | AA |", "|---|---:|---:|"])
    for title, left, right in (("BiReal -> LUT4", bireal, lut4), ("LUT4 -> LUT6", lut4, lut6), ("FP -> LUT6", rows["fp"], lut6)):
        oa = "-" if left is None or right is None else "{:+.2f}".format(100.0 * (right["oa"] - left["oa"]))
        aa = "-" if left is None or right is None else "{:+.2f}".format(100.0 * (right["aa"] - left["aa"]))
        lines.append("| {} | {} | {} |".format(title, oa, aa))
    lines.append("")
    return "\n".join(lines)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pol-root", default="runs/pol_insar_island/reduced_t3/seed0")
    parser.add_argument("--opensarship-root", default="runs/opensarship_slc/full_flow/seed0")
    parser.add_argument("--output", default="POL_INSAR_OPENSARSHIP_RESULTS.md")
    args = parser.parse_args(argv)
    content = "# Pol-InSAR-Island and OpenSARShip Complex LUT Results\n\n" + section("Pol-InSAR-Island FP1/L T3", Path(args.pol_root)) + section("OpenSARShip SLC", Path(args.opensarship_root))
    Path(args.output).write_text(content, encoding="utf-8")
    print(args.output)


if __name__ == "__main__":
    main()

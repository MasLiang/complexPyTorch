#!/usr/bin/env python3
"""Summarize a fixed analytic dominance training workdir."""

import argparse
import json
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.analyze_training_run import analyze_workdir


def read_json(path):
    if not path.is_file():
        raise FileNotFoundError("Missing metrics file: {}".format(path))
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def analyze_fixed_analytic(workdir):
    workdir = Path(workdir)
    metrics = read_json(workdir / "fixed_analytic_metrics.json")
    generic = analyze_workdir(workdir)
    best = metrics.get("best_metrics", {})
    last = metrics.get("last_metrics", {})
    args = metrics.get("args", {})
    diagnostics = metrics.get("analytic_diagnostics", {})
    gradients = metrics.get("bit_gradients") or {}
    occupancy = metrics.get("bit_occupancy") or {}
    configured_epochs = int(args.get("num_epochs", 0) or 0)
    parsed_epochs = int(generic.get("epoch_count", 0) or 0)
    status = (
        "complete"
        if configured_epochs and parsed_epochs >= configured_epochs
        else "running_or_incomplete"
    )

    warnings = []
    if metrics.get("training_uses_lut") is not False:
        warnings.append("metrics do not certify training_uses_lut=false")
    if diagnostics.get("lut_modules") != 0:
        warnings.append("model diagnostics found LUT modules")
    if not diagnostics.get("hardware_deployable"):
        warnings.append("checkpoint is not marked hardware deployable")
    if occupancy and occupancy.get("active_codes", 0) < 8:
        warnings.append("not all eight semantic addresses are active")
    for name, values in gradients.get("bits", {}).items():
        if values.get("mean_abs", 0.0) <= 0.0:
            warnings.append("{} has zero recorded gradient".format(name))
    if (
        last.get("train_acc") is not None
        and last.get("val_acc") is not None
        and last["train_acc"] - last["val_acc"] > 0.15
    ):
        warnings.append("final train/validation gap exceeds 15 percentage points")

    return {
        "workdir": str(workdir),
        "status": status,
        "configured_epochs": configured_epochs,
        "parsed_epoch_count": parsed_epochs,
        "best_checkpoint": metrics.get("best_checkpoint"),
        "best_metrics": best,
        "last_metrics": last,
        "analytic_diagnostics": diagnostics,
        "bit_gradients": gradients,
        "bit_occupancy": occupancy,
        "hardware_spec": metrics.get("hardware_spec", {}),
        "training_uses_lut": metrics.get("training_uses_lut"),
        "truth_table_trainable": metrics.get("truth_table_trainable"),
        "warnings": warnings,
    }


def _fmt(value, digits=4):
    if value is None:
        return "n/a"
    if isinstance(value, float):
        return ("{:.%df}" % digits).format(value)
    return str(value)


def render_markdown(report):
    best = report["best_metrics"]
    last = report["last_metrics"]
    diagnostics = report["analytic_diagnostics"]
    occupancy = report["bit_occupancy"]
    lines = [
        "# Fixed Analytic Dominance Run",
        "",
        "- Workdir: `{}`".format(report["workdir"]),
        "- Status: `{}` ({}/{})".format(
            report["status"],
            report["parsed_epoch_count"],
            report["configured_epochs"],
        ),
        "- Training uses LUT: `{}`".format(report["training_uses_lut"]),
        "- Truth table trainable: `{}`".format(
            report["truth_table_trainable"]
        ),
        "",
        "| checkpoint | epoch | train | val | test |",
        "|---|---:|---:|---:|---:|",
        "| best | {} | {} | {} | {} |".format(
            best.get("epoch", "n/a"),
            _fmt(best.get("train_acc")),
            _fmt(best.get("val_acc")),
            _fmt(best.get("test_acc")),
        ),
        "| last | {} | {} | {} | {} |".format(
            last.get("epoch", "n/a"),
            _fmt(last.get("train_acc")),
            _fmt(last.get("val_acc")),
            _fmt(last.get("test_acc")),
        ),
        "",
        "## Operator Audit",
        "",
        "- Analytic conv modules: `{}`".format(
            diagnostics.get("analytic_conv_modules", "n/a")
        ),
        "- LUT modules: `{}`".format(diagnostics.get("lut_modules", "n/a")),
        "- Dominance slice differences: `{}/{}` ({})".format(
            diagnostics.get("extra_bit_slice_hard_diff", "n/a"),
            diagnostics.get("extra_bit_slice_pairs", "n/a"),
            _fmt(diagnostics.get("extra_bit_slice_hard_diff_ratio")),
        ),
        "- Comparator beta/scale: `{}/{}`".format(
            diagnostics.get("comparator_beta", "n/a"),
            diagnostics.get("comparator_scale", "n/a"),
        ),
    ]
    if occupancy:
        lines.extend(
            [
                "- Active codes: `{}/8`".format(
                    occupancy.get("active_codes", "n/a")
                ),
                "- Dominance-one ratio: `{}`".format(
                    _fmt(occupancy.get("extra_bit_one_ratio"))
                ),
                "- Normalized entropy: `{}`".format(
                    _fmt(occupancy.get("normalized_entropy"))
                ),
            ]
        )
    gradients = report["bit_gradients"].get("bits", {})
    if gradients:
        lines.extend(["", "## Bit Gradients", ""])
        for name, values in gradients.items():
            lines.append(
                "- `{}`: nonzero={}, mean_abs={}, max_abs={}".format(
                    name,
                    _fmt(values.get("nonzero_ratio")),
                    _fmt(values.get("mean_abs"), digits=6),
                    _fmt(values.get("max_abs"), digits=6),
                )
            )
    if report["warnings"]:
        lines.extend(["", "## Warnings", ""])
        lines.extend("- {}".format(item) for item in report["warnings"])
    return "\n".join(lines) + "\n"


def parse_args(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("workdir", type=Path)
    parser.add_argument("--format", choices=["markdown", "json"], default="markdown")
    parser.add_argument("--output", type=Path)
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    report = analyze_fixed_analytic(args.workdir)
    text = (
        json.dumps(report, indent=2, sort_keys=True) + "\n"
        if args.format == "json"
        else render_markdown(report)
    )
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding="utf-8")
    else:
        print(text, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Create a reusable status report for one hard Direct LUT5 run."""

import argparse
import json
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.analyze_training_run import (
    analyze_workdir,
    parse_invocation_options,
)


def read_json(path):
    if not path.is_file():
        return {}
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def analyze_direct_lut5(workdir):
    workdir = Path(workdir)
    generic = analyze_workdir(workdir, requested_phase=2.5)
    metrics = read_json(workdir / "direct_lut5_metrics.json")
    options = parse_invocation_options(generic.get("invocation"))
    configured_epochs = None
    try:
        configured_epochs = int(options.get("--num-epochs"))
    except (TypeError, ValueError):
        pass

    best = metrics.get("best_metrics", {})
    last = metrics.get("last_metrics", {})
    last_epoch = last.get("epoch")
    if configured_epochs is None or last_epoch is None:
        status = "unknown"
    elif int(last_epoch) >= configured_epochs:
        status = "complete"
    else:
        status = "running_or_stopped"

    gradients = metrics.get("bit_gradients") or {}
    gradient_bits = gradients.get("bits", {})
    zero_gradient_bits = [
        name
        for name, values in gradient_bits.items()
        if float(values.get("mean_abs", 0.0)) == 0.0
    ]
    diagnostics = metrics.get("lut_diagnostics", {})
    occupancy = metrics.get("bit_occupancy", {})

    warnings = []
    if gradients and int(gradients.get("captured_tensors", 0)) == 0:
        warnings.append("No explicit activation-bit gradients were captured.")
    if zero_gradient_bits:
        warnings.append(
            "Zero mean gradient for bit(s): {}.".format(
                ", ".join(zero_gradient_bits)
            )
        )
    if occupancy and int(occupancy.get("active_codes", 0)) < 8:
        warnings.append(
            "Only {} of 8 activation codes are occupied.".format(
                occupancy.get("active_codes")
            )
        )
    if (
        diagnostics
        and last.get("lut_trainable") is True
        and int(diagnostics.get("sign_diff", 0)) == 0
    ):
        warnings.append("No LUT truth-table sign has changed from initialization.")

    return {
        "workdir": str(workdir),
        "status": status,
        "configured_epochs": configured_epochs,
        "parsed_epoch_count": generic.get("epoch_count", 0),
        "encoder_mode": metrics.get("encoder_mode"),
        "best_checkpoint": metrics.get("best_checkpoint"),
        "best_metrics": best,
        "last_metrics": last,
        "lut_diagnostics": diagnostics,
        "bit_gradients": gradients,
        "bit_occupancy": occupancy,
        "warnings": warnings,
        "invocation": generic.get("invocation"),
    }


def metric_line(name, values):
    return (
        "- {}: epoch={}, train_acc={}, val_acc={}, test_acc={}".format(
            name,
            values.get("epoch", "n/a"),
            _format_number(values.get("train_acc")),
            _format_number(values.get("val_acc")),
            _format_number(values.get("test_acc")),
        )
    )


def _format_number(value):
    if value is None:
        return "n/a"
    if isinstance(value, (int, float)):
        return "{:.4f}".format(value)
    return str(value)


def render_markdown(report):
    diagnostics = report["lut_diagnostics"]
    occupancy = report["bit_occupancy"]
    gradients = report["bit_gradients"]
    lines = [
        "# Direct LUT5 Run Report",
        "",
        "- Workdir: `{}`".format(report["workdir"]),
        "- Encoder: `{}`".format(report["encoder_mode"] or "unknown"),
        "- Status: `{}` ({}/{} epochs)".format(
            report["status"],
            report["last_metrics"].get("epoch", report["parsed_epoch_count"]),
            report["configured_epochs"] or "?",
        ),
        metric_line("Best", report["best_metrics"]),
        metric_line("Last", report["last_metrics"]),
        "",
        "## LUT State",
        "",
        "- Sign changes: {}/{} ({:.4f}%)".format(
            diagnostics.get("sign_diff", 0),
            diagnostics.get("entries", 0),
            100.0 * float(diagnostics.get("sign_diff_ratio", 0.0)),
        ),
        "- Extra-bit slice differences: {}/{} ({:.4f}%)".format(
            diagnostics.get("extra_bit_slice_hard_diff", 0),
            diagnostics.get("extra_bit_slice_pairs", 0),
            100.0
            * float(diagnostics.get("extra_bit_slice_hard_diff_ratio", 0.0)),
        ),
        "- Mean absolute logit: {}".format(
            _format_number(diagnostics.get("abs_logit_mean"))
        ),
        "- Hardware-hard checkpoint: `{}`".format(
            diagnostics.get("fully_hard", False)
        ),
        "",
        "## Activation Bits",
        "",
        "- Occupied codes: {}/8; entropy={}; extra-bit-one ratio={}".format(
            occupancy.get("active_codes", 0),
            _format_number(occupancy.get("normalized_entropy")),
            _format_number(occupancy.get("extra_bit_one_ratio")),
        ),
    ]
    for name, values in gradients.get("bits", {}).items():
        lines.append(
            "- `{}` gradient: nonzero={:.4f}, mean_abs={:.3e}, max_abs={:.3e}".format(
                name,
                float(values.get("nonzero_ratio", 0.0)),
                float(values.get("mean_abs", 0.0)),
                float(values.get("max_abs", 0.0)),
            )
        )
    if report["warnings"]:
        lines.extend(("", "## Warnings", ""))
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
    report = analyze_direct_lut5(args.workdir)
    if args.format == "json":
        rendered = json.dumps(report, indent=2, sort_keys=True) + "\n"
    else:
        rendered = render_markdown(report)
    if args.output is None:
        print(rendered, end="")
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
        print("Wrote {}".format(args.output))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

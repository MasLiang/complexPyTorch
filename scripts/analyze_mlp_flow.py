#!/usr/bin/env python3
"""Summarize one or more isolated MLP-flow training directories."""

import argparse
import json
import re
from pathlib import Path


PHASE_RE = re.compile(r"MLP flow Phase (1\.2|2\.2|3\.2|4\.2)")
EPOCH_RE = re.compile(
    r"Epoch (?P<epoch>\d+) .*?train_acc=(?P<train_acc>[0-9.]+) "
    r"val_loss=(?P<val_loss>[0-9.]+) val_acc=(?P<val_acc>[0-9.]+) "
    r"test_loss=(?P<test_loss>[0-9.]+) test_acc=(?P<test_acc>[0-9.]+) "
    r".*?hard_changed=(?P<hard_changed>[0-9.]+) "
    r"d_sensitive=(?P<d_sensitive>[0-9.]+) "
    r"d_(?:coeff|score)=(?P<d_coeff>[0-9.]+)"
    r"(?: zero_score=(?P<zero_score>[0-9.]+))?"
)
INITIAL_RE = re.compile(
    r"Initial .*?_acc=(?P<selection_acc>[0-9.]+) "
    r"test_acc=(?P<test_acc>[0-9.]+) "
    r"hard_changed=(?P<hard_changed>[0-9.]+) "
    r"d_sensitive=(?P<d_sensitive>[0-9.]+) "
    r"d_(?:coeff|score)=(?P<d_coeff>[0-9.]+)"
    r"(?: zero_score=(?P<zero_score>[0-9.]+))?"
)


def parse_training_log(path):
    phases = {}
    current_phase = None
    if not path.is_file():
        return phases
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        phase_match = PHASE_RE.search(line)
        if phase_match:
            current_phase = phase_match.group(1)
            phases.setdefault(current_phase, {"epochs": []})
            continue
        if current_phase is None:
            continue
        initial_match = INITIAL_RE.search(line)
        if initial_match:
            phases[current_phase]["initial"] = {
                key: float(value)
                for key, value in initial_match.groupdict().items()
                if value is not None
            }
            continue
        epoch_match = EPOCH_RE.search(line)
        if epoch_match:
            record = {
                key: (int(value) if key == "epoch" else float(value))
                for key, value in epoch_match.groupdict().items()
                if value is not None
            }
            phases[current_phase]["epochs"].append(record)
    for values in phases.values():
        epochs = values["epochs"]
        if epochs:
            values["last"] = epochs[-1]
            values["best_logged_val"] = max(
                epochs,
                key=lambda item: item["val_acc"],
            )
    return phases


def load_metrics(path):
    if not path.is_file():
        return {}
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def summarize_workdir(workdir):
    workdir = Path(workdir).resolve()
    metrics = load_metrics(workdir / "mlp_flow_metrics.json")
    log_phases = parse_training_log(workdir / "logs" / "train.txt")
    summary = {
        "workdir": str(workdir),
        "metrics": metrics,
        "log_phases": log_phases,
    }
    allocations = []
    for metric in metrics.values():
        diagnostics = metric.get("operation_diagnostics", {})
        layers = diagnostics.get("layers", [])
        if layers:
            allocations.append(layers[0].get("allocation"))
    summary["allocation"] = allocations[-1] if allocations else None
    return summary


def markdown_report(summaries):
    lines = ["# MLP Operation Flow Report", ""]
    for summary in summaries:
        lines.extend(
            [
                "## {}".format(Path(summary["workdir"]).name),
                "",
                "- Workdir: `{}`".format(summary["workdir"]),
                "- Allocation: `{}`".format(summary.get("allocation") or "unknown"),
                "",
                "| Phase | Best epoch | Val acc | Test acc | Hard changed | d-sensitive | Zero score |",
                "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
            ]
        )
        for phase in ("1p2", "2p2", "3p2", "4p2"):
            metric = summary["metrics"].get("phase{}".format(phase))
            if metric is None:
                continue
            diagnostics = metric.get("operation_diagnostics", {})
            lines.append(
                "| {} | {} | {:.4f} | {:.4f} | {:.4f} | {:.4f} | {:.4f} |".format(
                    phase.replace("p", "."),
                    metric.get("epoch", "-"),
                    metric.get("val_acc", 0.0),
                    metric.get("test_acc", 0.0),
                    diagnostics.get("hard_changed_ratio", 0.0),
                    diagnostics.get("d_sensitive_ratio", 0.0),
                    diagnostics.get("zero_score_ratio", 0.0),
                )
            )
        lines.append("")
        for phase, values in sorted(summary["log_phases"].items()):
            last = values.get("last")
            if last is None:
                continue
            lines.append(
                "- Phase {} currently/last logged epoch {}: val {:.4f}, test {:.4f}, d-sensitive {:.4f}.".format(
                    phase,
                    last["epoch"],
                    last["val_acc"],
                    last["test_acc"],
                    last["d_sensitive"],
                )
            )
        lines.append("")
    if len(summaries) == 2:
        lines.extend(["## Direct Comparison", ""])
        left, right = summaries
        for phase in ("1p2", "2p2", "3p2", "4p2"):
            left_metric = left["metrics"].get("phase{}".format(phase))
            right_metric = right["metrics"].get("phase{}".format(phase))
            if left_metric is None or right_metric is None:
                continue
            delta = right_metric.get("test_acc", 0.0) - left_metric.get("test_acc", 0.0)
            lines.append(
                "- Phase {} test-accuracy delta (second - first): {:+.4f}.".format(
                    phase.replace("p", "."),
                    delta,
                )
            )
        lines.append("")
    return "\n".join(lines)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("workdirs", nargs="+", type=Path)
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args(argv)

    summaries = [summarize_workdir(path) for path in args.workdirs]
    output_dir = (
        args.output_dir.resolve()
        if args.output_dir is not None
        else args.workdirs[0].resolve()
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "mlp_flow_report.json"
    markdown_path = output_dir / "mlp_flow_report.md"
    with json_path.open("w", encoding="utf-8") as handle:
        json.dump({"runs": summaries}, handle, indent=2, sort_keys=True)
    markdown_path.write_text(markdown_report(summaries), encoding="utf-8")
    print("Wrote {}".format(json_path))
    print("Wrote {}".format(markdown_path))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


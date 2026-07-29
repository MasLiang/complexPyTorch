#!/usr/bin/env python3
"""Summarize a Phase2.7 dual-LUT6_2 training directory."""

import argparse
import json
import math
import re
import statistics
from pathlib import Path


NUMBER = r"[-+0-9.eE]+"
INITIAL_RE = re.compile(
    r"Initial Phase2 anchor: val_acc=(?P<val_acc>" + NUMBER + r") "
    r"test_acc=(?P<test_acc>" + NUMBER + r") "
    r"rho=(?P<rho>" + NUMBER + r") "
    r"levels=\((?P<level_low>" + NUMBER + r"), "
    r"(?P<level_high>" + NUMBER + r")\) "
    r"hardware_max_diff=(?P<hardware_max_diff>" + NUMBER + r") "
    r"pred_mismatch=(?P<prediction_mismatches>\d+)/(?P<samples>\d+)"
)
EPOCH_RE = re.compile(
    r"Epoch (?P<epoch>\d+) "
    r"train_loss=(?P<train_loss>" + NUMBER + r") "
    r"train_acc=(?P<train_acc>" + NUMBER + r") "
    r"val_loss=(?P<val_loss>" + NUMBER + r") "
    r"val_acc=(?P<val_acc>" + NUMBER + r") "
    r"test_loss=(?P<test_loss>" + NUMBER + r") "
    r"test_acc=(?P<test_acc>" + NUMBER + r") "
    r"rho=(?P<rho>" + NUMBER + r") "
    r"levels=\((?P<level_low>" + NUMBER + r"),"
    r"(?P<level_high>" + NUMBER + r")\) "
    r"threshold=\((?P<threshold_min>" + NUMBER + r"),"
    r"(?P<threshold_mean>" + NUMBER + r"),"
    r"(?P<threshold_max>" + NUMBER + r")\) "
    r".*?deployable=(?P<deployable>True|False) "
    r"best_(?P<selection_name>val|test)=(?P<best_selection>" + NUMBER + r") "
    r"time=(?P<time_seconds>" + NUMBER + r")s"
)
HARDWARE_RE = re.compile(
    r"First full 2-bit hardware check: "
    r"max_diff=(?P<max_abs_logit_difference>" + NUMBER + r") "
    r"mean_diff=(?P<mean_abs_logit_difference>" + NUMBER + r") "
    r"pred_mismatch=(?P<prediction_mismatches>\d+)/(?P<samples>\d+)"
)
COMPLETED_RE = re.compile(
    r"Completed Phase2\.7: best_(?P<selection_name>val|test)_acc="
    r"(?P<best_acc>" + NUMBER + r") at epoch=(?P<epoch>\d+)"
)


def load_json(path):
    if not path.is_file():
        return {}
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def parse_training_log(path):
    result = {
        "initial": None,
        "epochs": [],
        "first_full_hardware_check": None,
        "completed": None,
    }
    if not path.is_file():
        raise FileNotFoundError("Training log not found: {}".format(path))
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        match = INITIAL_RE.search(line)
        if match:
            result["initial"] = {
                key: (
                    int(value)
                    if key in ("prediction_mismatches", "samples")
                    else float(value)
                )
                for key, value in match.groupdict().items()
            }
            continue
        match = EPOCH_RE.search(line)
        if match:
            record = {}
            for key, value in match.groupdict().items():
                if key == "epoch":
                    record[key] = int(value)
                elif key == "deployable":
                    record[key] = value == "True"
                elif key == "selection_name":
                    record[key] = value
                else:
                    record[key] = float(value)
            result["epochs"].append(record)
            continue
        match = HARDWARE_RE.search(line)
        if match:
            result["first_full_hardware_check"] = {
                key: (
                    int(value)
                    if key in ("prediction_mismatches", "samples")
                    else float(value)
                )
                for key, value in match.groupdict().items()
            }
            continue
        match = COMPLETED_RE.search(line)
        if match:
            result["completed"] = {
                "selection_name": match.group("selection_name"),
                "best_acc": float(match.group("best_acc")),
                "epoch": int(match.group("epoch")),
            }
    if not result["epochs"]:
        raise RuntimeError("No Phase2.7 epoch records found in {}".format(path))
    return result


def best_record(records, metric):
    if not records:
        return None
    loss_name = metric.replace("_acc", "_loss")
    return max(
        records,
        key=lambda item: (item[metric], -item[loss_name]),
    )


def aggregate_window(records):
    if not records:
        return None
    return {
        "first_epoch": records[0]["epoch"],
        "last_epoch": records[-1]["epoch"],
        "count": len(records),
        "best_val": best_record(records, "val_acc"),
        "best_test": best_record(records, "test_acc"),
        "mean_val_acc": statistics.fmean(item["val_acc"] for item in records),
        "std_val_acc": (
            statistics.pstdev(item["val_acc"] for item in records)
            if len(records) > 1
            else 0.0
        ),
        "mean_test_acc": statistics.fmean(item["test_acc"] for item in records),
        "std_test_acc": (
            statistics.pstdev(item["test_acc"] for item in records)
            if len(records) > 1
            else 0.0
        ),
    }


def milestone_records(records):
    targets = (0.0, 0.25, 0.5, 0.75, 0.9, 1.0)
    selected = {}
    for target in targets:
        record = min(
            records,
            key=lambda item: (abs(item["rho"] - target), item["epoch"]),
        )
        selected["rho_{:.2f}".format(target)] = record
    return selected


def binary_entropy(probability):
    probability = min(max(float(probability), 0.0), 1.0)
    if probability in (0.0, 1.0):
        return 0.0
    return -(
        probability * math.log2(probability)
        + (1.0 - probability) * math.log2(1.0 - probability)
    )


def occupancy_summary(metrics):
    result = {}
    occupancy = metrics.get("two_bit_occupancy", {})
    for component in ("real", "imag"):
        values = occupancy.get(component, {})
        fractions = values.get("fractions")
        if not fractions or len(fractions) != 4:
            continue
        code_entropy = -sum(
            value * math.log2(value)
            for value in fractions
            if value > 0.0
        )
        high_fraction = fractions[0] + fractions[3]
        positive_fraction = fractions[2] + fractions[3]
        result[component] = {
            "counts": values.get("counts"),
            "fractions": fractions,
            "normalized_code_entropy": code_entropy / 2.0,
            "high_magnitude_fraction": high_fraction,
            "low_magnitude_fraction": 1.0 - high_fraction,
            "positive_fraction": positive_fraction,
            "negative_fraction": 1.0 - positive_fraction,
            "magnitude_bit_entropy": binary_entropy(high_fraction),
            "sign_bit_entropy": binary_entropy(positive_fraction),
        }
    return result


def threshold_summary(hardware_export):
    layers = hardware_export.get("activation_thresholds", {})
    if not layers:
        return {}
    layer_records = []
    all_values = []
    stage_values = {}
    for name, values in sorted(layers.items()):
        values = [float(value) for value in values]
        all_values.extend(values)
        stage = name.split(".", 1)[0]
        stage_values.setdefault(stage, []).extend(values)
        layer_records.append(
            {
                "layer": name,
                "channels": len(values),
                "min": min(values),
                "mean": statistics.fmean(values),
                "max": max(values),
            }
        )
    return {
        "channels": len(all_values),
        "min": min(all_values),
        "mean": statistics.fmean(all_values),
        "max": max(all_values),
        "layers": layer_records,
        "stages": {
            stage: {
                "channels": len(values),
                "min": min(values),
                "mean": statistics.fmean(values),
                "max": max(values),
            }
            for stage, values in sorted(stage_values.items())
        },
    }


def summarize_run(workdir):
    workdir = Path(workdir).resolve()
    parsed = parse_training_log(workdir / "logs" / "train.txt")
    epochs = parsed["epochs"]
    warmup = [item for item in epochs if item["rho"] == 0.0]
    transition = [item for item in epochs if 0.0 < item["rho"] < 1.0]
    deployable = [item for item in epochs if item["deployable"]]
    metrics = load_json(workdir / "dual_lut6_metrics.json")
    hardware_export = load_json(workdir / "dual_lut6_hardware.json")
    hardware_verification = load_json(
        workdir / "dual_lut6_hardware_verification.json"
    )

    initial = parsed["initial"]
    best_deployable_val = best_record(deployable, "val_acc")
    best_deployable_test = best_record(deployable, "test_acc")
    gain = {}
    if initial and best_deployable_val:
        gain = {
            "best_val_minus_initial_val": (
                best_deployable_val["val_acc"] - initial["val_acc"]
            ),
            "best_val_checkpoint_test_minus_initial_test": (
                best_deployable_val["test_acc"] - initial["test_acc"]
            ),
            "best_test_minus_initial_test": (
                best_deployable_test["test_acc"] - initial["test_acc"]
            ),
        }

    return {
        "workdir": str(workdir),
        "complete": epochs[-1]["epoch"] == 200 and parsed["completed"] is not None,
        "initial": initial,
        "first_full_hardware_check": parsed["first_full_hardware_check"],
        "completed": parsed["completed"],
        "last": epochs[-1],
        "best_all_val": best_record(epochs, "val_acc"),
        "best_all_test": best_record(epochs, "test_acc"),
        "best_deployable_val": best_deployable_val,
        "best_deployable_test": best_deployable_test,
        "windows": {
            "rho_zero": aggregate_window(warmup),
            "rho_transition": aggregate_window(transition),
            "deployable": aggregate_window(deployable),
            "last_10": aggregate_window(epochs[-10:]),
        },
        "milestones": milestone_records(epochs),
        "gain": gain,
        "saved_best": metrics.get("metrics", {}),
        "two_bit_diagnostics": metrics.get("two_bit_diagnostics", {}),
        "occupancy": occupancy_summary(metrics),
        "thresholds": threshold_summary(hardware_export),
        "hardware_verification": hardware_verification,
    }


def percent(value):
    return "{:.2f}%".format(100.0 * float(value))


def markdown_report(summary):
    lines = [
        "# Phase2.7 Dual LUT6_2 Run Report",
        "",
        "Workdir: {}".format(summary["workdir"]),
        "",
    ]
    initial = summary.get("initial") or {}
    best = summary.get("best_deployable_val") or {}
    best_test = summary.get("best_deployable_test") or {}
    last = summary["last"]
    lines.extend(
        [
            "## Core Result",
            "",
            "| Point | Epoch | Val | Test | rho |",
            "| --- | ---: | ---: | ---: | ---: |",
            "| Phase2 anchor | 0 | {} | {} | 0 |".format(
                percent(initial.get("val_acc", 0.0)),
                percent(initial.get("test_acc", 0.0)),
            ),
            "| Best deployable val | {} | {} | {} | {:.4f} |".format(
                best.get("epoch", "-"),
                percent(best.get("val_acc", 0.0)),
                percent(best.get("test_acc", 0.0)),
                best.get("rho", 0.0),
            ),
            "| Best deployable test | {} | {} | {} | {:.4f} |".format(
                best_test.get("epoch", "-"),
                percent(best_test.get("val_acc", 0.0)),
                percent(best_test.get("test_acc", 0.0)),
                best_test.get("rho", 0.0),
            ),
            "| Endpoint | {} | {} | {} | {:.4f} |".format(
                last["epoch"],
                percent(last["val_acc"]),
                percent(last["test_acc"]),
                last["rho"],
            ),
            "",
            "## rho Milestones",
            "",
            "| Target rho | Epoch | Actual rho | Train | Val | Test |",
            "| ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for name, record in summary["milestones"].items():
        lines.append(
            "| {} | {} | {:.4f} | {} | {} | {} |".format(
                name.replace("rho_", ""),
                record["epoch"],
                record["rho"],
                percent(record["train_acc"]),
                percent(record["val_acc"]),
                percent(record["test_acc"]),
            )
        )
    lines.extend(
        [
            "",
            "## Code Occupancy",
            "",
            "| Component | 00 | 01 | 10 | 11 | High magnitude | Code entropy |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for component, values in summary["occupancy"].items():
        fractions = values["fractions"]
        lines.append(
            "| {} | {} | {} | {} | {} | {} | {:.4f} |".format(
                component,
                *(percent(value) for value in fractions),
                percent(values["high_magnitude_fraction"]),
                values["normalized_code_entropy"],
            )
        )
    diagnostics = summary.get("two_bit_diagnostics", {})
    hardware = summary.get("first_full_hardware_check") or {}
    full_hardware = summary.get("hardware_verification") or {}
    thresholds = summary.get("thresholds") or {}
    lines.extend(
        [
            "",
            "## Quantizer And Hardware Diagnostics",
            "",
            "- Final levels: low={:.6f}, high={:.6f}.".format(
                diagnostics.get("level_low", 0.0),
                diagnostics.get("level_high", 0.0),
            ),
            "- Best thresholds: min={:.6f}, mean={:.6f}, max={:.6f}.".format(
                diagnostics.get("threshold_min", 0.0),
                diagnostics.get("threshold_mean", 0.0),
                diagnostics.get("threshold_max", 0.0),
            ),
            "- First full-rho hardware check: max diff={:.6e}, mean diff={:.6e}, mismatches={}/{}.".format(
                hardware.get("max_abs_logit_difference", 0.0),
                hardware.get("mean_abs_logit_difference", 0.0),
                hardware.get("prediction_mismatches", 0),
                hardware.get("samples", 0),
            ),
            "",
            "## Full Hardware Verification",
            "",
            "- Standard test accuracy: {}.".format(
                percent(full_hardware.get("standard_accuracy", 0.0))
            ),
            "- Hardware test accuracy: {}.".format(
                percent(full_hardware.get("hardware_accuracy", 0.0))
            ),
            "- Prediction mismatches: {}/{}; max diff={:.6e}, mean diff={:.6e}.".format(
                full_hardware.get("prediction_mismatches", 0),
                full_hardware.get("samples", 0),
                full_hardware.get("max_abs_logit_difference", 0.0),
                full_hardware.get("mean_abs_logit_difference", 0.0),
            ),
            "",
            "## Stage Thresholds",
            "",
            "| Stage | Channels | Min | Mean | Max |",
            "| --- | ---: | ---: | ---: | ---: |",
        ]
    )
    for stage, values in thresholds.get("stages", {}).items():
        lines.append(
            "| {} | {} | {:.6f} | {:.6f} | {:.6f} |".format(
                stage,
                values["channels"],
                values["min"],
                values["mean"],
                values["max"],
            )
        )
    lines.append("")
    return "\n".join(lines)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("workdir", type=Path)
    parser.add_argument(
        "--output-prefix",
        default="dual_lut6_analysis",
    )
    args = parser.parse_args(argv)
    summary = summarize_run(args.workdir)
    output_dir = args.workdir.resolve()
    json_path = output_dir / (args.output_prefix + ".json")
    markdown_path = output_dir / (args.output_prefix + ".md")
    with json_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)
    markdown_path.write_text(markdown_report(summary), encoding="utf-8")
    print("Wrote {}".format(json_path))
    print("Wrote {}".format(markdown_path))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

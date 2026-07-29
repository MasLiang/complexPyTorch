#!/usr/bin/env python3
"""Summarize a Phase 4 magnitude-STE LUT5 training directory."""

import argparse
import json
import re
import shlex
import statistics
from pathlib import Path


NUMBER = r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?"
EPOCH_RE = re.compile(
    r"Epoch\s+(?P<epoch>\d+)\s+"
    r"train_loss:\s+(?P<train_loss>" + NUMBER + r"),\s+"
    r"train_acc:\s+(?P<train_acc>" + NUMBER + r"),\s+"
    r"val_loss:\s+(?P<val_loss>" + NUMBER + r"),\s+"
    r"val_acc:\s+(?P<val_acc>" + NUMBER + r"),\s+"
    r"test_loss:\s+(?P<test_loss>" + NUMBER + r"),\s+"
    r"test_acc:\s+(?P<test_acc>" + NUMBER + r")\s+"
    r"\((?P<time_seconds>" + NUMBER + r")s\)"
)
OCCUPANCY_RE = re.compile(
    r"\[LUT5 Magnitude Occupancy\] Epoch (?P<epoch>\d+) test: "
    r"ones=(?P<ones>\d+), zeros=(?P<zeros>\d+), "
    r"one_ratio=(?P<one_ratio>" + NUMBER + r")%, "
    r"layer_ratio_range=\[(?P<layer_min>" + NUMBER + r")%, "
    r"(?P<layer_max>" + NUMBER + r")%\], "
    r"mean_abs_margin=(?P<mean_abs_margin>" + NUMBER + r"), "
    r"near_boundary_ratio=(?P<near_boundary_ratio>" + NUMBER + r")%"
)
GRADIENT_RE = re.compile(
    r"\[Magnitude Fifth-Bit Gradient\] Epoch (?P<epoch>\d+): "
    r"nonzero_batches=(?P<nonzero_batches>\d+)/(?P<total_batches>\d+) "
    r"\((?P<batch_ratio>" + NUMBER + r")%\), "
    r"nonzero_elements=(?P<nonzero_elements>\d+)/"
    r"(?P<total_elements>\d+) \((?P<element_ratio>" + NUMBER + r")%\), "
    r"mean_abs=(?P<mean_abs>" + NUMBER + r"), "
    r"max_abs=(?P<max_abs>" + NUMBER + r"); "
    r"physical_threshold\[min/mean/max\]="
    r"(?P<threshold_min>" + NUMBER + r")/"
    r"(?P<threshold_mean>" + NUMBER + r")/"
    r"(?P<threshold_max>" + NUMBER + r"); "
    r"shadow_epsilon=(?P<shadow_epsilon>" + NUMBER + r")"
)
DIAGNOSTIC_RE = re.compile(
    r"\[Phase4 Magnitude-STE LUT5\] Epoch (?P<epoch>\d+): "
    r"init_sign_diff=(?P<init_sign_diff>\d+) / (?P<entries>\d+) "
    r"\((?P<init_sign_ratio>" + NUMBER + r")%\), "
    r"magnitude_slice_hard_diff=(?P<hard_diff>\d+) / "
    r"(?P<slice_pairs>\d+) \((?P<hard_diff_ratio>" + NUMBER + r")%\), "
    r"magnitude_slice_soft_abs_diff_mean=(?P<soft_diff_mean>"
    + NUMBER
    + r"), max=(?P<soft_diff_max>"
    + NUMBER
    + r"), hard_checkpoint_eligible=(?P<eligible>True|False)"
)
SENSITIVITY_RE = re.compile(
    r"\[Phase4 Magnitude-STE LUT5 Bit Sensitivity\] Epoch (?P<epoch>\d+): "
    r"sign_r=(?P<sign_r>\d+)/(?P<pairs_r>\d+) "
    r"\((?P<sign_r_ratio>" + NUMBER + r")%\), "
    r"sign_i=(?P<sign_i>\d+)/(?P<pairs_i>\d+) "
    r"\((?P<sign_i_ratio>" + NUMBER + r")%\), "
    r"magnitude=(?P<magnitude>\d+)/(?P<pairs_magnitude>\d+) "
    r"\((?P<magnitude_ratio>" + NUMBER + r")%\)"
)
SCHEDULER_RE = re.compile(
    r"\[Phase 4 Annealing Scheduler\] Epoch (?P<epoch>\d+): "
    r"tau=(?P<tau>" + NUMBER + r"), "
    r"hard_ratio=(?P<hard_ratio>" + NUMBER + r"), "
    r"hard_mode=(?P<hard_mode>True|False)"
)
INVOCATION_RE = re.compile(r"INVOCATION:\s+(?P<invocation>.*)$")


def typed_record(match, integer_fields=(), boolean_fields=()):
    record = {}
    for key, value in match.groupdict().items():
        if key in integer_fields:
            record[key] = int(value)
        elif key in boolean_fields:
            record[key] = value == "True"
        else:
            record[key] = float(value)
    return record


def parse_invocation(path):
    if not path.is_file():
        return "", {}
    invocation = ""
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        match = INVOCATION_RE.search(line)
        if match:
            invocation = match.group("invocation").strip()
            break
    if not invocation:
        return "", {}
    try:
        tokens = shlex.split(invocation)
    except ValueError:
        tokens = invocation.split()
    options = {}
    index = 0
    while index < len(tokens):
        token = tokens[index]
        if token.startswith("--"):
            value = True
            if index + 1 < len(tokens) and not tokens[index + 1].startswith("--"):
                value = tokens[index + 1]
                index += 1
            options[token[2:]] = value
        index += 1
    return invocation, options


def load_json(path):
    if not path.is_file():
        return {}
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def parse_log(path):
    if not path.is_file():
        raise FileNotFoundError("Training log not found: {}".format(path))
    records = {
        "epochs": [],
        "occupancy": [],
        "gradients": [],
        "diagnostics": [],
        "sensitivity": [],
        "scheduler": [],
    }
    specs = (
        (
            EPOCH_RE,
            "epochs",
            ("epoch",),
            (),
        ),
        (
            OCCUPANCY_RE,
            "occupancy",
            ("epoch", "ones", "zeros"),
            (),
        ),
        (
            GRADIENT_RE,
            "gradients",
            (
                "epoch",
                "nonzero_batches",
                "total_batches",
                "nonzero_elements",
                "total_elements",
            ),
            (),
        ),
        (
            DIAGNOSTIC_RE,
            "diagnostics",
            (
                "epoch",
                "init_sign_diff",
                "entries",
                "hard_diff",
                "slice_pairs",
            ),
            ("eligible",),
        ),
        (
            SENSITIVITY_RE,
            "sensitivity",
            (
                "epoch",
                "sign_r",
                "pairs_r",
                "sign_i",
                "pairs_i",
                "magnitude",
                "pairs_magnitude",
            ),
            (),
        ),
        (
            SCHEDULER_RE,
            "scheduler",
            ("epoch",),
            ("hard_mode",),
        ),
    )
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        for regex, name, integer_fields, boolean_fields in specs:
            match = regex.search(line)
            if match:
                records[name].append(
                    typed_record(match, integer_fields, boolean_fields)
                )
                break
    if not records["epochs"]:
        raise RuntimeError("No epoch records found in {}".format(path))
    return records


def best_record(records, metric):
    loss_metric = metric.replace("_acc", "_loss")
    return max(records, key=lambda item: (item[metric], -item[loss_metric]))


def latest_at_or_before(records, epoch):
    candidates = [item for item in records if item["epoch"] <= epoch]
    return max(candidates, key=lambda item: item["epoch"]) if candidates else None


def first_matching(records, predicate):
    return next((item for item in records if predicate(item)), None)


def accuracy_window(records, size=10):
    window = records[-size:]
    return {
        "start_epoch": window[0]["epoch"],
        "end_epoch": window[-1]["epoch"],
        "count": len(window),
        "val_mean": statistics.fmean(item["val_acc"] for item in window),
        "val_std": statistics.pstdev(item["val_acc"] for item in window),
        "test_mean": statistics.fmean(item["test_acc"] for item in window),
        "test_std": statistics.pstdev(item["test_acc"] for item in window),
    }


def summarize(workdir):
    workdir = Path(workdir).resolve()
    records = parse_log(workdir / "logs" / "train.txt")
    invocation, options = parse_invocation(workdir / "logs" / "entry.txt")
    epochs = records["epochs"]
    best_saved_hard = load_json(
        workdir / "phase_metrics.json"
    ).get("phase4")
    last_epoch = epochs[-1]["epoch"]
    total_epochs = int(options.get("num-epochs", last_epoch))
    mean_epoch_seconds = statistics.fmean(
        item["time_seconds"] for item in epochs[-10:]
    )
    first_hard_diff = first_matching(
        records["diagnostics"], lambda item: item["hard_diff"] > 0
    )
    first_magnitude_sensitive = first_matching(
        records["sensitivity"], lambda item: item["magnitude"] > 0
    )
    first_eligible = first_matching(
        records["diagnostics"], lambda item: item["eligible"]
    )
    return {
        "workdir": str(workdir),
        "invocation": invocation,
        "configured_epochs": total_epochs,
        "completed_epochs": last_epoch,
        "remaining_epochs": max(total_epochs - last_epoch, 0),
        "estimated_remaining_hours": (
            max(total_epochs - last_epoch, 0) * mean_epoch_seconds / 3600.0
        ),
        "last": epochs[-1],
        "best_val": best_record(epochs, "val_acc"),
        "best_test": best_record(epochs, "test_acc"),
        "last_10": accuracy_window(epochs),
        "best_saved_hard": best_saved_hard,
        "latest_occupancy": latest_at_or_before(
            records["occupancy"], last_epoch
        ),
        "latest_gradient": latest_at_or_before(
            records["gradients"], last_epoch
        ),
        "latest_diagnostic": latest_at_or_before(
            records["diagnostics"], last_epoch
        ),
        "latest_sensitivity": latest_at_or_before(
            records["sensitivity"], last_epoch
        ),
        "latest_scheduler": latest_at_or_before(
            records["scheduler"], last_epoch
        ),
        "first_hard_diff": first_hard_diff,
        "first_magnitude_sensitive": first_magnitude_sensitive,
        "first_hard_checkpoint_eligible": first_eligible,
        "gradient_first": records["gradients"][0] if records["gradients"] else None,
        "diagnostic_first": (
            records["diagnostics"][0] if records["diagnostics"] else None
        ),
    }


def render(summary):
    lines = [
        "Magnitude-STE LUT5 run: {}".format(summary["workdir"]),
        "progress: {}/{} epochs, estimated {:.2f} h remaining".format(
            summary["completed_epochs"],
            summary["configured_epochs"],
            summary["estimated_remaining_hours"],
        ),
        "best val: {:.2%} at epoch {} (test {:.2%})".format(
            summary["best_val"]["val_acc"],
            summary["best_val"]["epoch"],
            summary["best_val"]["test_acc"],
        ),
        "best test: {:.2%} at epoch {} (val {:.2%})".format(
            summary["best_test"]["test_acc"],
            summary["best_test"]["epoch"],
            summary["best_test"]["val_acc"],
        ),
        "last: epoch {} val {:.2%} test {:.2%}".format(
            summary["last"]["epoch"],
            summary["last"]["val_acc"],
            summary["last"]["test_acc"],
        ),
        "last 10: val {:.2%} +/- {:.2%}, test {:.2%} +/- {:.2%}".format(
            summary["last_10"]["val_mean"],
            summary["last_10"]["val_std"],
            summary["last_10"]["test_mean"],
            summary["last_10"]["test_std"],
        ),
    ]
    occupancy = summary["latest_occupancy"]
    best_saved_hard = summary["best_saved_hard"]
    if best_saved_hard:
        lines.insert(
            4,
            "best saved hard: val {:.2%}, test {:.2%} at epoch {}".format(
                best_saved_hard["val_acc"],
                best_saved_hard["test_acc"],
                best_saved_hard["epoch"],
            ),
        )
    if occupancy:
        lines.append(
            "occupancy@{}: one {:.2f}%, layers [{:.2f}%, {:.2f}%], "
            "near boundary {:.2f}%".format(
                occupancy["epoch"],
                occupancy["one_ratio"],
                occupancy["layer_min"],
                occupancy["layer_max"],
                occupancy["near_boundary_ratio"],
            )
        )
    gradient = summary["latest_gradient"]
    if gradient:
        lines.append(
            "fifth-bit gradient@{}: batches {:.2f}%, elements {:.2f}%, "
            "mean {:.3e}, threshold mean {:.6f}".format(
                gradient["epoch"],
                gradient["batch_ratio"],
                gradient["element_ratio"],
                gradient["mean_abs"],
                gradient["threshold_mean"],
            )
        )
    diagnostic = summary["latest_diagnostic"]
    if diagnostic:
        lines.append(
            "LUT@{}: hard slice diff {}/{}, soft mean {:.6f}, "
            "soft max {:.6f}, eligible={}".format(
                diagnostic["epoch"],
                diagnostic["hard_diff"],
                diagnostic["slice_pairs"],
                diagnostic["soft_diff_mean"],
                diagnostic["soft_diff_max"],
                diagnostic["eligible"],
            )
        )
    sensitivity = summary["latest_sensitivity"]
    if sensitivity:
        lines.append(
            "magnitude hard sensitivity@{}: {}/{} ({:.2f}%)".format(
                sensitivity["epoch"],
                sensitivity["magnitude"],
                sensitivity["pairs_magnitude"],
                sensitivity["magnitude_ratio"],
            )
        )
    scheduler = summary["latest_scheduler"]
    if scheduler:
        lines.append(
            "scheduler@{}: tau {:.4f}, hard ratio {:.4f}, hard mode={}".format(
                scheduler["epoch"],
                scheduler["tau"],
                scheduler["hard_ratio"],
                scheduler["hard_mode"],
            )
        )
    return "\n".join(lines)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Summarize a Phase 4 magnitude-STE LUT5 run."
    )
    parser.add_argument("workdir")
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main():
    args = parse_args()
    summary = summarize(args.workdir)
    content = (
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
        if args.json
        else render(summary) + "\n"
    )
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(content, encoding="utf-8")
        print("Wrote {}".format(args.output))
    else:
        print(content, end="")


if __name__ == "__main__":
    main()

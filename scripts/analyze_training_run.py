#!/usr/bin/env python3
"""Summarize training curves, LR segments, and quantizer diagnostics."""

import argparse
import json
import math
import re
import shlex
from pathlib import Path


EPOCH_RE = re.compile(
    r"Epoch\s+(?P<epoch>\d+)\s+"
    r"train_loss:\s+(?P<train_loss>[-+0-9.eE]+),\s+"
    r"train_acc:\s+(?P<train_acc>[-+0-9.eE]+),\s+"
    r"val_loss:\s+(?P<val_loss>[-+0-9.eE]+),\s+"
    r"val_acc:\s+(?P<val_acc>[-+0-9.eE]+),\s+"
    r"test_loss:\s+(?P<test_loss>[-+0-9.eE]+),\s+"
    r"test_acc:\s+(?P<test_acc>[-+0-9.eE]+)"
)
LR_RE = re.compile(r"Current learning rate value is\s+([-+0-9.eE]+)")
INVOCATION_RE = re.compile(r"INVOCATION:\s+(.*)$")
SAVE_RE = re.compile(r"Saved best model .* at epoch\s+(\d+)")
OCCUPANCY_RE = re.compile(
    r"\[C8 Code Occupancy\] Epoch\s+(?P<epoch>\d+) test:\s+"
    r"counts=\[(?P<counts>[^]]*)\],\s+"
    r"ratios=\[(?P<ratios>[^]]*)\],\s+"
    r"active_codes=(?P<active>\d+),\s+"
    r"(?P<secondary_name>added_axis_ratio|odd_octant_ratio)="
    r"(?P<secondary>[-+0-9.eE]+)%,\s+"
    r"normalized_entropy=(?P<entropy>[-+0-9.eE]+),\s+"
    r"mean_angle_error=(?P<mean_angle>[-+0-9.eE]+)deg,\s+"
    r"max_angle_error=(?P<max_angle>[-+0-9.eE]+)deg"
)
LUT_DIAGNOSTIC_RE = re.compile(
    r"\[(?P<stage>Phase(?:2\.1 Learned LUT5|3\.1 (?:Learned LUT5|Semantic LUT5|Neural LUT5|Pure MLP5)))\] "
    r"Epoch\s+(?P<epoch>\d+):\s+"
    r"init_sign_diff=(?P<sign_diff>\d+)\s+/\s+"
    r"(?P<entries>\d+)\s+\((?P<sign_ratio>[-+0-9.eE]+)%\),\s+"
    r"(?P<slice_label>[A-Za-z0-9_]+)_slice_hard_diff="
    r"(?P<slice_diff>\d+)\s+/\s+(?P<slice_pairs>\d+)\s+"
    r"\((?P<slice_ratio>[-+0-9.eE]+)%\),\s+"
    r"(?P=slice_label)_slice_soft_abs_diff_mean="
    r"(?P<soft_mean>[-+0-9.eE]+),\s+max="
    r"(?P<soft_max>[-+0-9.eE]+),\s+"
    r"hard_checkpoint_eligible=(?P<eligible>True|False)"
)
LUT_SCHEDULER_RE = re.compile(
    r"\[(?P<stage>Phase(?:2\.1 Learned|3\.1 (?:Learned|Neural)) LUT5) "
    r"Annealing Scheduler\] Epoch\s+(?P<epoch_index>\d+):\s+"
    r"tau=(?P<tau>[-+0-9.eE]+),\s+"
    r"hard_ratio=(?P<hard_ratio>[-+0-9.eE]+),\s+"
    r"hard_mode=(?P<hard_mode>True|False)"
)
LUT_BIT_SENSITIVITY_RE = re.compile(
    r"\[(?P<stage>Phase(?:2\.1 Learned LUT5|3\.1 (?:Learned LUT5|Semantic LUT5|Neural LUT5|Pure MLP5)) "
    r"Bit Sensitivity)\] Epoch\s+(?P<epoch>\d+):\s+"
    r"(?P<body>.*)\."
)
LUT_BIT_ITEM_RE = re.compile(
    r"(?P<name>[A-Za-z0-9_]+)=(?P<diff>\d+)/(?P<pairs>\d+) "
    r"\((?P<ratio>[-+0-9.eE]+)%\)"
)
NEURAL_HARD_PROJECTION_RE = re.compile(
    r"\[Phase3\.1 (?:Neural LUT5|Pure MLP5) Hard Projection\] Epoch\s+"
    r"(?P<epoch>\d+):\s+"
    r"val_loss=(?P<val_loss>[-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?),\s+"
    r"val_acc=(?P<val_acc>[-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?),\s+"
    r"test_loss=(?P<test_loss>[-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?),\s+"
    r"test_acc=(?P<test_acc>[-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?)"
)


def phase_tag(value):
    numeric = float(value)
    if numeric.is_integer():
        return str(int(numeric))
    return str(numeric).replace(".", "p")


def phase_key(value):
    return "phase" + phase_tag(value)


def read_json(path):
    if not path.is_file():
        return {}
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def parse_float_list(text):
    if not text.strip():
        return []
    return [float(token.strip()) for token in text.split(",")]


def parse_int_list(text):
    if not text.strip():
        return []
    return [int(token.strip()) for token in text.split(",")]


def parse_invocation_phase(invocation):
    if not invocation:
        return None
    try:
        tokens = shlex.split(invocation)
    except ValueError:
        tokens = invocation.split()
    for index, token in enumerate(tokens[:-1]):
        if token == "--phase":
            return tokens[index + 1]
    return None


def parse_invocation_options(invocation):
    if not invocation:
        return {}
    try:
        tokens = shlex.split(invocation)
    except ValueError:
        tokens = invocation.split()

    options = {}
    index = 0
    while index < len(tokens):
        token = tokens[index]
        if not token.startswith("--"):
            index += 1
            continue
        if index + 1 < len(tokens) and not tokens[index + 1].startswith("--"):
            options[token] = tokens[index + 1]
            index += 2
        else:
            options[token] = True
            index += 1
    return options


def infer_piecewise_lr(epoch, invocation):
    options = parse_invocation_options(invocation)
    schedule = options.get("--schedule", "default")
    epoch_index = epoch - 1

    if schedule == "default":
        if epoch_index < 10:
            return 0.01
        if epoch_index < 100:
            return 0.1
        if epoch_index < 120:
            return 0.01
        if epoch_index < 150:
            return 0.001
        return 0.0001

    if schedule not in {"bireal", "constant"}:
        return None
    try:
        base_lr = float(options["--lr"])
    except (KeyError, TypeError, ValueError):
        return None

    if schedule == "constant":
        return base_lr

    try:
        num_epochs = int(options["--num-epochs"])
    except (KeyError, TypeError, ValueError):
        return None
    warmup_epochs = 5
    if epoch_index < warmup_epochs:
        return base_lr * float(epoch_index + 1) / float(warmup_epochs)
    if epoch_index < int(num_epochs * 0.5):
        return base_lr
    if epoch_index < int(num_epochs * 0.75):
        return base_lr * 0.1
    if epoch_index < int(num_epochs * 0.875):
        return base_lr * 0.01
    return base_lr * 0.001


def new_log_segment(invocation=None):
    return {
        "invocation": invocation,
        "epochs": [],
        "occupancy": {},
        "lut_diagnostics": {},
        "lut_schedule": {},
        "lut_bit_sensitivity": {},
        "hard_projections": {},
        "saved_best_epochs": [],
    }


def deinterleave_epoch_records(epoch_records):
    tracks = []
    assignments = {}
    for record in epoch_records:
        epoch = record["epoch"]
        candidates = [
            index
            for index, track in enumerate(tracks)
            if track["epochs"]
            and track["epochs"][-1]["epoch"] + 1 == epoch
        ]
        if candidates:
            track_index = max(
                candidates,
                key=lambda index: tracks[index]["epochs"][-1]["_line"],
            )
        else:
            track_index = len(tracks)
            tracks.append(new_log_segment())
        tracks[track_index]["epochs"].append(record)
        assignments.setdefault(epoch, []).append(
            (record["_line"], track_index)
        )
    return tracks, assignments


def parse_training_log(path):
    result = {
        "invocation": None,
        "invocations": [],
        "segments": [],
    }
    if not path.is_file():
        return result

    epoch_records = []
    occupancy_records = []
    lut_diagnostic_records = []
    lut_scheduler_records = []
    lut_bit_sensitivity_records = []
    hard_projection_records = []
    save_records = []
    lr_records = []

    with path.open("r", encoding="utf-8", errors="replace") as handle:
        for line_number, line in enumerate(handle, start=1):
            invocation_match = INVOCATION_RE.search(line)
            if invocation_match:
                invocation = invocation_match.group(1).strip()
                result["invocations"].append(invocation)
                result["invocation"] = invocation

            lr_match = LR_RE.search(line)
            if lr_match:
                lr_records.append(
                    {
                        "lr": float(lr_match.group(1)),
                        "_line": line_number,
                    }
                )

            epoch_match = EPOCH_RE.search(line)
            if epoch_match:
                epoch = int(epoch_match.group("epoch"))
                record = {
                    "epoch": epoch,
                    "lr": None,
                    "_line": line_number,
                }
                for key in (
                    "train_loss",
                    "train_acc",
                    "val_loss",
                    "val_acc",
                    "test_loss",
                    "test_acc",
                ):
                    record[key] = float(epoch_match.group(key))
                epoch_records.append(record)

            occupancy_match = OCCUPANCY_RE.search(line)
            if occupancy_match:
                epoch = int(occupancy_match.group("epoch"))
                occupancy_records.append({
                    "epoch": epoch,
                    "_line": line_number,
                    "counts": parse_int_list(occupancy_match.group("counts")),
                    "ratios": parse_float_list(occupancy_match.group("ratios")),
                    "active_codes": int(occupancy_match.group("active")),
                    "secondary_ratio_name": occupancy_match.group(
                        "secondary_name"
                    ),
                    "secondary_code_ratio": float(
                        occupancy_match.group("secondary")
                    ) / 100.0,
                    "normalized_entropy": float(occupancy_match.group("entropy")),
                    "mean_angle_error_deg": float(
                        occupancy_match.group("mean_angle")
                    ),
                    "max_angle_error_deg": float(
                        occupancy_match.group("max_angle")
                    ),
                })

            lut_diagnostic_match = LUT_DIAGNOSTIC_RE.search(line)
            if lut_diagnostic_match:
                lut_diagnostic_records.append(
                    {
                        "stage": lut_diagnostic_match.group("stage"),
                        "epoch": int(
                            lut_diagnostic_match.group("epoch")
                        ),
                        "init_sign_diff": int(
                            lut_diagnostic_match.group("sign_diff")
                        ),
                        "entries": int(
                            lut_diagnostic_match.group("entries")
                        ),
                        "init_sign_diff_ratio": float(
                            lut_diagnostic_match.group("sign_ratio")
                        ) / 100.0,
                        "slice_label": lut_diagnostic_match.group(
                            "slice_label"
                        ),
                        "slice_hard_diff": int(
                            lut_diagnostic_match.group("slice_diff")
                        ),
                        "slice_pairs": int(
                            lut_diagnostic_match.group("slice_pairs")
                        ),
                        "slice_hard_diff_ratio": float(
                            lut_diagnostic_match.group("slice_ratio")
                        ) / 100.0,
                        "slice_soft_abs_diff_mean": float(
                            lut_diagnostic_match.group("soft_mean")
                        ),
                        "slice_soft_abs_diff_max": float(
                            lut_diagnostic_match.group("soft_max")
                        ),
                        "hard_checkpoint_eligible": (
                            lut_diagnostic_match.group("eligible") == "True"
                        ),
                        "_line": line_number,
                    }
                )

            lut_scheduler_match = LUT_SCHEDULER_RE.search(line)
            if lut_scheduler_match:
                epoch_index = int(
                    lut_scheduler_match.group("epoch_index")
                )
                lut_scheduler_records.append(
                    {
                        "stage": lut_scheduler_match.group("stage"),
                        "epoch": epoch_index + 1,
                        "epoch_index": epoch_index,
                        "tau": float(lut_scheduler_match.group("tau")),
                        "hard_ratio": float(
                            lut_scheduler_match.group("hard_ratio")
                        ),
                        "hard_mode": (
                            lut_scheduler_match.group("hard_mode") == "True"
                        ),
                        "_line": line_number,
                    }
                )

            sensitivity_match = LUT_BIT_SENSITIVITY_RE.search(line)
            if sensitivity_match:
                bits = {}
                for item in LUT_BIT_ITEM_RE.finditer(
                    sensitivity_match.group("body")
                ):
                    bits[item.group("name")] = {
                        "hard_diff": int(item.group("diff")),
                        "pairs": int(item.group("pairs")),
                        "hard_diff_ratio": (
                            float(item.group("ratio")) / 100.0
                        ),
                    }
                lut_bit_sensitivity_records.append(
                    {
                        "stage": sensitivity_match.group("stage"),
                        "epoch": int(sensitivity_match.group("epoch")),
                        "bits": bits,
                        "_line": line_number,
                    }
                )

            hard_projection_match = NEURAL_HARD_PROJECTION_RE.search(line)
            if hard_projection_match:
                hard_projection_records.append(
                    {
                        "epoch": int(
                            hard_projection_match.group("epoch")
                        ),
                        "val_loss": float(
                            hard_projection_match.group("val_loss")
                        ),
                        "val_acc": float(
                            hard_projection_match.group("val_acc")
                        ),
                        "test_loss": float(
                            hard_projection_match.group("test_loss")
                        ),
                        "test_acc": float(
                            hard_projection_match.group("test_acc")
                        ),
                        "_line": line_number,
                    }
                )

            save_match = SAVE_RE.search(line)
            if save_match:
                save_records.append(
                    {
                        "epoch": int(save_match.group(1)),
                        "_line": line_number,
                    }
                )

    tracks, assignments = deinterleave_epoch_records(epoch_records)

    lr_transitions = [[] for _ in tracks]
    if epoch_records:
        first_epoch_line = min(record["_line"] for record in epoch_records)
        initial_lr_records = [
            record for record in lr_records if record["_line"] < first_epoch_line
        ]
        for track_index, record in enumerate(initial_lr_records[:len(tracks)]):
            lr_transitions[track_index].append((1, record["lr"]))

        epoch_line_assignments = sorted(
            (
                record["_line"],
                track_index,
                record["epoch"],
            )
            for track_index, track in enumerate(tracks)
            for record in track["epochs"]
        )
        for lr_record in lr_records:
            if lr_record in initial_lr_records:
                continue
            preceding = [
                assignment
                for assignment in epoch_line_assignments
                if assignment[0] < lr_record["_line"]
            ]
            if not preceding:
                continue
            _, track_index, previous_epoch = max(
                preceding,
                key=lambda assignment: assignment[0],
            )
            lr_transitions[track_index].append(
                (previous_epoch + 1, lr_record["lr"])
            )

    for track_index, track in enumerate(tracks):
        transitions = sorted(lr_transitions[track_index])
        for record in track["epochs"]:
            active = [
                value
                for start_epoch, value in transitions
                if start_epoch <= record["epoch"]
            ]
            record["lr"] = active[-1] if active else None

    used_epoch_records = set()
    for occupancy in occupancy_records:
        candidates = assignments.get(occupancy["epoch"], [])
        available = [
            candidate
            for candidate in candidates
            if candidate not in used_epoch_records
        ]
        following = [
            candidate
            for candidate in available
            if candidate[0] >= occupancy["_line"]
        ]
        if not available:
            continue
        record_line, track_index = min(
            following or available,
            key=lambda candidate: abs(candidate[0] - occupancy["_line"]),
        )
        used_epoch_records.add((record_line, track_index))
        occupancy = dict(occupancy)
        occupancy.pop("_line", None)
        tracks[track_index]["occupancy"][occupancy["epoch"]] = occupancy

    for save in save_records:
        candidates = [
            candidate
            for candidate in assignments.get(save["epoch"], [])
            if candidate[0] <= save["_line"]
        ]
        if not candidates:
            continue
        _, track_index = max(candidates, key=lambda candidate: candidate[0])
        tracks[track_index]["saved_best_epochs"].append(save["epoch"])

    for diagnostic in lut_diagnostic_records:
        candidates = [
            candidate
            for candidate in assignments.get(diagnostic["epoch"], [])
            if candidate[0] <= diagnostic["_line"]
        ]
        if not candidates:
            continue
        _, track_index = max(candidates, key=lambda candidate: candidate[0])
        record = dict(diagnostic)
        record.pop("_line", None)
        tracks[track_index]["lut_diagnostics"][record["epoch"]] = record

    for scheduler in lut_scheduler_records:
        candidates = assignments.get(scheduler["epoch"], [])
        if not candidates:
            continue
        _, track_index = min(
            candidates,
            key=lambda candidate: abs(candidate[0] - scheduler["_line"]),
        )
        record = dict(scheduler)
        record.pop("_line", None)
        tracks[track_index]["lut_schedule"][record["epoch"]] = record

    for sensitivity in lut_bit_sensitivity_records:
        candidates = assignments.get(sensitivity["epoch"], [])
        if not candidates:
            continue
        _, track_index = min(
            candidates,
            key=lambda candidate: abs(
                candidate[0] - sensitivity["_line"]
            ),
        )
        record = dict(sensitivity)
        record.pop("_line", None)
        tracks[track_index]["lut_bit_sensitivity"][
            record["epoch"]
        ] = record

    for projection in hard_projection_records:
        candidates = assignments.get(projection["epoch"], [])
        if not candidates:
            continue
        _, track_index = min(
            candidates,
            key=lambda candidate: abs(
                candidate[0] - projection["_line"]
            ),
        )
        record = dict(projection)
        record.pop("_line", None)
        tracks[track_index]["hard_projections"][record["epoch"]] = record

    for track in tracks:
        for record in track["epochs"]:
            record.pop("_line", None)
    result["segments"] = tracks

    return result


def merge_entry_invocations(workdir, parsed_log):
    entry_path = workdir / "logs" / "entry.txt"
    if not entry_path.is_file():
        return
    entry_invocations = []
    with entry_path.open("r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            match = INVOCATION_RE.search(line)
            if match:
                entry_invocations.append(match.group(1).strip())
    if not parsed_log["invocations"]:
        parsed_log["invocations"] = entry_invocations
    for segment, invocation in zip(
        parsed_log["segments"],
        parsed_log["invocations"],
    ):
        segment["invocation"] = invocation
    if parsed_log["invocations"]:
        parsed_log["invocation"] = parsed_log["invocations"][-1]


def choose_invocation(invocations, requested_phase):
    unique_invocations = list(dict.fromkeys(invocations))
    if requested_phase is not None:
        matching = [
            invocation
            for invocation in unique_invocations
            if parse_invocation_phase(invocation) == str(requested_phase)
        ]
        if matching:
            return matching[-1]
    if len(unique_invocations) == 1:
        return unique_invocations[0]
    return None


def segment_metric_distance(segment, metrics):
    target_epoch = metrics.get("epoch")
    if target_epoch is None:
        return None
    candidates = [
        record for record in segment["epochs"] if record["epoch"] == target_epoch
    ]
    if not candidates:
        return None
    distances = []
    for record in candidates:
        terms = []
        for key in ("train_acc", "val_acc", "test_acc"):
            if key in metrics:
                terms.append(abs(record[key] - float(metrics[key])))
        if terms:
            distances.append(sum(terms) / len(terms))
    return min(distances) if distances else None


def select_log_segment(segments, metrics):
    if not segments:
        return new_log_segment(), "no parsed segment", None

    scored = []
    for index, segment in enumerate(segments):
        distance = segment_metric_distance(segment, metrics)
        if distance is not None:
            scored.append((distance, -index, index, segment))
    if scored:
        distance, _, index, segment = min(
            scored,
            key=lambda item: (item[0], item[1]),
        )
        return (
            segment,
            f"matched saved metrics (mean abs acc error={distance:.6g})",
            index,
        )

    return (
        segments[-1],
        "latest epoch segment (no saved-metric match)",
        len(segments) - 1,
    )


def lr_segments(epochs):
    segments = []
    for record in epochs:
        lr = record["lr"]
        if not segments or segments[-1]["lr"] != lr:
            segments.append(
                {
                    "start_epoch": record["epoch"],
                    "end_epoch": record["epoch"],
                    "lr": lr,
                    "best_val_acc": record["val_acc"],
                    "best_val_epoch": record["epoch"],
                }
            )
        else:
            segments[-1]["end_epoch"] = record["epoch"]
        if record["val_acc"] > segments[-1]["best_val_acc"]:
            segments[-1]["best_val_acc"] = record["val_acc"]
            segments[-1]["best_val_epoch"] = record["epoch"]
    return segments


def selected_epoch_records(epochs, segments):
    if not epochs:
        return []
    by_epoch = {record["epoch"]: record for record in epochs}
    best_val = max(epochs, key=lambda item: item["val_acc"])
    best_test = max(epochs, key=lambda item: item["test_acc"])
    selected = {epochs[0]["epoch"], epochs[-1]["epoch"]}
    selected.update((best_val["epoch"], best_test["epoch"]))
    for segment in segments[1:]:
        start = segment["start_epoch"]
        selected.update((start - 1, start))
    return [by_epoch[epoch] for epoch in sorted(selected) if epoch in by_epoch]


def scalar_metrics(metrics):
    return {
        key: value
        for key, value in metrics.items()
        if isinstance(value, (int, float, str, bool)) or value is None
    }


def infer_metrics(metrics_by_phase, requested_phase, invocation):
    phase = requested_phase or parse_invocation_phase(invocation)
    if phase is not None:
        key = phase_key(phase)
        return key, metrics_by_phase.get(key, {})
    if len(metrics_by_phase) == 1:
        key = next(iter(metrics_by_phase))
        return key, metrics_by_phase[key]
    return None, {}


def analyze_workdir(workdir, requested_phase=None, segment_index=None):
    workdir = Path(workdir)
    parsed_log = parse_training_log(workdir / "logs" / "train.txt")
    merge_entry_invocations(workdir, parsed_log)
    invocation = choose_invocation(parsed_log["invocations"], requested_phase)
    metrics_by_phase = read_json(workdir / "phase_metrics.json")
    metrics_key, metrics = infer_metrics(
        metrics_by_phase,
        requested_phase,
        invocation,
    )
    if segment_index is None:
        segment, segment_selection, selected_segment_index = (
            select_log_segment(parsed_log["segments"], metrics)
        )
    else:
        if not 0 <= segment_index < len(parsed_log["segments"]):
            raise ValueError(
                "segment {} is out of range for {} parsed segment(s)".format(
                    segment_index + 1,
                    len(parsed_log["segments"]),
                )
            )
        selected_segment_index = segment_index
        segment = parsed_log["segments"][segment_index]
        segment_selection = "explicit segment {}".format(segment_index + 1)
    invocation = segment["invocation"] or invocation
    epochs = segment["epochs"]
    for record in epochs:
        inferred_lr = infer_piecewise_lr(record["epoch"], invocation)
        if inferred_lr is not None:
            record["lr"] = inferred_lr
    segments = lr_segments(epochs)

    analysis = {
        "workdir": str(workdir),
        "invocation": invocation,
        "phase_metrics_key": metrics_key,
        "phase_metrics": scalar_metrics(metrics),
        "parsed_segment_count": len(parsed_log["segments"]),
        "selected_segment": (
            None
            if selected_segment_index is None
            else selected_segment_index + 1
        ),
        "segment_selection": segment_selection,
        "epoch_count": len(epochs),
        "lr_segments": segments,
        "selected_epochs": selected_epoch_records(epochs, segments),
        "saved_best_epochs": segment["saved_best_epochs"],
        "occupancy": {},
        "lut_diagnostics": [
            segment["lut_diagnostics"][epoch]
            for epoch in sorted(segment["lut_diagnostics"])
        ],
        "lut_schedule": [
            segment["lut_schedule"][epoch]
            for epoch in sorted(segment["lut_schedule"])
        ],
        "lut_bit_sensitivity": [
            segment["lut_bit_sensitivity"][epoch]
            for epoch in sorted(segment["lut_bit_sensitivity"])
        ],
        "hard_projections": [
            segment["hard_projections"][epoch]
            for epoch in sorted(segment["hard_projections"])
        ],
    }
    if epochs:
        best_val = max(epochs, key=lambda item: item["val_acc"])
        best_test = max(epochs, key=lambda item: item["test_acc"])
        final = epochs[-1]
        analysis.update(
            {
                "best_val": best_val,
                "best_test": best_test,
                "final": final,
                "best_val_generalization_gap": (
                    best_val["train_acc"] - best_val["val_acc"]
                ),
                "final_generalization_gap": (
                    final["train_acc"] - final["val_acc"]
                ),
            }
        )

        occupancy = segment["occupancy"]
        for label, epoch in (
            ("first", epochs[0]["epoch"]),
            ("best_val", best_val["epoch"]),
            ("final", final["epoch"]),
        ):
            if epoch in occupancy:
                analysis["occupancy"][label] = occupancy[epoch]

    boundaries = []
    by_epoch = {record["epoch"]: record for record in epochs}
    for segment in segments[1:]:
        epoch = segment["start_epoch"]
        before = by_epoch.get(epoch - 1)
        after = by_epoch.get(epoch)
        if before is None or after is None:
            continue
        previous_lr = before["lr"]
        new_lr = after["lr"]
        ratio = None
        if previous_lr not in (None, 0.0) and new_lr is not None:
            ratio = new_lr / previous_lr
        boundaries.append(
            {
                "epoch": epoch,
                "previous_lr": previous_lr,
                "new_lr": new_lr,
                "lr_ratio": ratio,
                "val_acc_delta": after["val_acc"] - before["val_acc"],
                "test_acc_delta": after["test_acc"] - before["test_acc"],
                "train_acc_delta": after["train_acc"] - before["train_acc"],
            }
        )
    analysis["lr_boundaries"] = boundaries
    for diagnostic in analysis["lut_diagnostics"]:
        epoch_metrics = by_epoch.get(diagnostic["epoch"])
        if epoch_metrics is None:
            continue
        for key in ("train_acc", "val_acc", "test_acc"):
            diagnostic[key] = epoch_metrics[key]
    return analysis


def fmt_float(value, digits=4):
    if value is None:
        return "n/a"
    return f"{value:.{digits}f}"


def render_markdown(analyses):
    lines = []
    for analysis in analyses:
        lines.append(f"# Training run: {analysis['workdir']}")
        if analysis["invocation"]:
            lines.append("")
            lines.append(f"Invocation: `{analysis['invocation']}`")
        lines.append("")
        lines.append(
            f"Parsed epochs: {analysis['epoch_count']}; "
            f"segment: {analysis['selected_segment'] or 'n/a'}/"
            f"{analysis['parsed_segment_count']}; "
            f"metrics key: {analysis['phase_metrics_key'] or 'n/a'}"
        )

        metrics = analysis["phase_metrics"]
        if metrics:
            lines.append("")
            lines.append("## Saved phase metrics")
            lines.append("")
            for key in sorted(metrics):
                lines.append(f"- `{key}`: {metrics[key]}")

        if analysis["epoch_count"]:
            best_val = analysis["best_val"]
            best_test = analysis["best_test"]
            final = analysis["final"]
            lines.extend(
                [
                    "",
                    "## Curve summary",
                    "",
                    "| point | epoch | lr | train | val | test |",
                    "|---|---:|---:|---:|---:|---:|",
                    (
                        f"| best val | {best_val['epoch']} | "
                        f"{fmt_float(best_val['lr'], 6)} | "
                        f"{best_val['train_acc']:.4f} | "
                        f"{best_val['val_acc']:.4f} | "
                        f"{best_val['test_acc']:.4f} |"
                    ),
                    (
                        f"| best test | {best_test['epoch']} | "
                        f"{fmt_float(best_test['lr'], 6)} | "
                        f"{best_test['train_acc']:.4f} | "
                        f"{best_test['val_acc']:.4f} | "
                        f"{best_test['test_acc']:.4f} |"
                    ),
                    (
                        f"| final | {final['epoch']} | "
                        f"{fmt_float(final['lr'], 6)} | "
                        f"{final['train_acc']:.4f} | "
                        f"{final['val_acc']:.4f} | "
                        f"{final['test_acc']:.4f} |"
                    ),
                    "",
                    (
                        "Generalization gap (train-val): "
                        f"best={analysis['best_val_generalization_gap']:.4f}, "
                        f"final={analysis['final_generalization_gap']:.4f}"
                    ),
                ]
            )

            lines.extend(
                [
                    "",
                    "## LR segments",
                    "",
                    "| epochs | lr | best val | best epoch |",
                    "|---|---:|---:|---:|",
                ]
            )
            for segment in analysis["lr_segments"]:
                lines.append(
                    f"| {segment['start_epoch']}-{segment['end_epoch']} | "
                    f"{fmt_float(segment['lr'], 6)} | "
                    f"{segment['best_val_acc']:.4f} | "
                    f"{segment['best_val_epoch']} |"
                )

            if analysis["lr_boundaries"]:
                lines.extend(
                    [
                        "",
                        "## LR boundary impact",
                        "",
                        "| epoch | LR change | ratio | delta train | delta val | delta test |",
                        "|---:|---:|---:|---:|---:|---:|",
                    ]
                )
                for boundary in analysis["lr_boundaries"]:
                    lines.append(
                        f"| {boundary['epoch']} | "
                        f"{fmt_float(boundary['previous_lr'], 6)} -> "
                        f"{fmt_float(boundary['new_lr'], 6)} | "
                        f"{fmt_float(boundary['lr_ratio'], 2)} | "
                        f"{boundary['train_acc_delta']:+.4f} | "
                        f"{boundary['val_acc_delta']:+.4f} | "
                        f"{boundary['test_acc_delta']:+.4f} |"
                    )

            lines.extend(
                [
                    "",
                    "## Selected epochs",
                    "",
                    "| epoch | lr | train | val | test |",
                    "|---:|---:|---:|---:|---:|",
                ]
            )
            for record in analysis["selected_epochs"]:
                lines.append(
                    f"| {record['epoch']} | {fmt_float(record['lr'], 6)} | "
                    f"{record['train_acc']:.4f} | "
                    f"{record['val_acc']:.4f} | "
                    f"{record['test_acc']:.4f} |"
                )

        if analysis["occupancy"]:
            lines.extend(
                [
                    "",
                    "## C8 occupancy snapshots",
                    "",
                    "| point | epoch | active | secondary ratio | entropy | mean angle |",
                    "|---|---:|---:|---:|---:|---:|",
                ]
            )
            for label, occupancy in analysis["occupancy"].items():
                lines.append(
                    f"| {label} | {occupancy['epoch']} | "
                    f"{occupancy['active_codes']} | "
                    f"{occupancy['secondary_ratio_name']}="
                    f"{occupancy['secondary_code_ratio']:.4f} | "
                    f"{occupancy['normalized_entropy']:.4f} | "
                    f"{occupancy['mean_angle_error_deg']:.4f} deg |"
                )

        if analysis["lut_schedule"]:
            lines.extend(
                [
                    "",
                    "## LUT annealing snapshots",
                    "",
                    "| epoch | tau | hard ratio | hard mode |",
                    "|---:|---:|---:|:---:|",
                ]
            )
            for snapshot in analysis["lut_schedule"]:
                lines.append(
                    f"| {snapshot['epoch']} | {snapshot['tau']:.4f} | "
                    f"{snapshot['hard_ratio']:.4f} | "
                    f"{snapshot['hard_mode']} |"
                )

        if analysis["hard_projections"]:
            lines.extend(
                [
                    "",
                    "## MLP/LUT hard projections",
                    "",
                    "| epoch | hard val loss | hard val acc | hard test loss | hard test acc |",
                    "|---:|---:|---:|---:|---:|",
                ]
            )
            for projection in analysis["hard_projections"]:
                lines.append(
                    f"| {projection['epoch']} | "
                    f"{projection['val_loss']:.6f} | "
                    f"{projection['val_acc']:.4f} | "
                    f"{projection['test_loss']:.6f} | "
                    f"{projection['test_acc']:.4f} |"
                )

        if analysis["lut_bit_sensitivity"]:
            lines.extend(
                [
                    "",
                    "## LUT activation-bit sensitivity",
                    "",
                    "| epoch | bit | hard differences | hard sensitivity |",
                    "|---:|:---|---:|---:|",
                ]
            )
            for snapshot in analysis["lut_bit_sensitivity"]:
                for bit_name, stats in snapshot["bits"].items():
                    lines.append(
                        f"| {snapshot['epoch']} | {bit_name} | "
                        f"{stats['hard_diff']} / {stats['pairs']} | "
                        f"{stats['hard_diff_ratio']:.4%} |"
                    )

        if analysis["lut_diagnostics"]:
            lines.extend(
                [
                    "",
                    "## Learned LUT diagnostics",
                    "",
                    "| epoch | train | val | test | init sign diff | slice | hard slice diff | soft slice mean | soft slice max | checkpoint eligible |",
                    "|---:|---:|---:|---:|---:|:---|---:|---:|---:|:---:|",
                ]
            )
            for diagnostic in analysis["lut_diagnostics"]:
                lines.append(
                    f"| {diagnostic['epoch']} | "
                    f"{fmt_float(diagnostic.get('train_acc'))} | "
                    f"{fmt_float(diagnostic.get('val_acc'))} | "
                    f"{fmt_float(diagnostic.get('test_acc'))} | "
                    f"{diagnostic['init_sign_diff']} / "
                    f"{diagnostic['entries']} "
                    f"({diagnostic['init_sign_diff_ratio']:.4%}) | "
                    f"{diagnostic['slice_label']} | "
                    f"{diagnostic['slice_hard_diff']} / "
                    f"{diagnostic['slice_pairs']} "
                    f"({diagnostic['slice_hard_diff_ratio']:.4%}) | "
                    f"{diagnostic['slice_soft_abs_diff_mean']:.6f} | "
                    f"{diagnostic['slice_soft_abs_diff_max']:.6f} | "
                    f"{diagnostic['hard_checkpoint_eligible']} |"
                )
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def parse_comparison(spec):
    if "=" not in spec:
        raise argparse.ArgumentTypeError(
            "comparison must be LABEL=WORKDIR[:PHASE]"
        )
    label, value = spec.split("=", 1)
    phase = None
    workdir = value
    if ":" in value:
        candidate_workdir, candidate_phase = value.rsplit(":", 1)
        try:
            float(candidate_phase)
        except ValueError:
            pass
        else:
            workdir = candidate_workdir
            phase = candidate_phase
    return label, workdir, phase


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Summarize a training workdir and optionally compare other runs."
        )
    )
    parser.add_argument("workdir", type=Path)
    parser.add_argument("--phase", default=None)
    segment_group = parser.add_mutually_exclusive_group()
    segment_group.add_argument(
        "--segment",
        type=int,
        default=None,
        help="Analyze one 1-based deinterleaved run segment",
    )
    segment_group.add_argument(
        "--all-segments",
        action="store_true",
        help="Analyze every deinterleaved run segment in the primary workdir",
    )
    parser.add_argument(
        "--compare",
        action="append",
        default=[],
        metavar="LABEL=WORKDIR[:PHASE]",
    )
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    if args.all_segments:
        probe = analyze_workdir(args.workdir, args.phase)
        segment_count = probe["parsed_segment_count"]
        analyses = [
            analyze_workdir(args.workdir, args.phase, index)
            for index in range(segment_count)
        ]
        labels = [
            "{}#{}".format(args.workdir.name, index + 1)
            for index in range(segment_count)
        ]
    else:
        segment_index = None
        if args.segment is not None:
            if args.segment < 1:
                parser.error("--segment must be at least 1")
            segment_index = args.segment - 1
        analyses = [
            analyze_workdir(args.workdir, args.phase, segment_index)
        ]
        labels = [args.workdir.name]
    for spec in args.compare:
        label, workdir, phase = parse_comparison(spec)
        labels.append(label)
        analyses.append(analyze_workdir(workdir, phase))

    if args.json:
        rendered = json.dumps(
            {
                "labels": labels,
                "analyses": analyses,
            },
            indent=2,
            sort_keys=True,
        ) + "\n"
    else:
        rendered = render_markdown(analyses)

    if args.output is None:
        print(rendered, end="")
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
        print(args.output)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Diagnose Pol-InSAR split drift without changing the training protocol.

The report compares class balance, spatial coverage, and train-normalized
complex-channel statistics across the official-train, validation, and test
pixel sets.  It can also summarize a partially completed FP stage history.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from datasets.pol_insar_island import (
    CLASS_NAMES,
    PolInSARScene,
    official_split,
    train_scales,
)


def _sample_coordinates(coordinates, maximum, rng):
    if len(coordinates) <= maximum:
        return coordinates
    return coordinates[rng.choice(len(coordinates), size=maximum, replace=False)]


def _channel_statistics(scene, coordinates, scales, maximum, seed):
    rng = np.random.default_rng(seed)
    sampled = _sample_coordinates(coordinates, maximum, rng)
    rows, cols = sampled[:, 0], sampled[:, 1]
    result = []
    for name, channel, scale in zip(("T11", "T22", "T33", "T12", "T13", "T23"), scene.channels, scales):
        values = np.asarray(channel[rows, cols] / scale, dtype=np.complex64)
        magnitude = np.abs(values)
        result.append({
            "channel": name,
            "real_mean": float(values.real.mean()),
            "real_std": float(values.real.std()),
            "imag_mean": float(values.imag.mean()),
            "imag_std": float(values.imag.std()),
            "rms": float(np.sqrt(np.mean(np.abs(values) ** 2))),
            "magnitude_q01_q50_q99": [float(value) for value in np.quantile(magnitude, (0.01, 0.5, 0.99))],
        })
    return {"sample_size": int(len(sampled)), "channels": result}


def _split_summary(scene, coordinates, labels, scales, maximum, seed):
    counts = np.bincount(labels[coordinates[:, 0], coordinates[:, 1]], minlength=len(CLASS_NAMES) + 1)[1:]
    row_min, col_min = coordinates.min(axis=0).tolist()
    row_max, col_max = coordinates.max(axis=0).tolist()
    return {
        "pixels": int(len(coordinates)),
        "class_counts": {name: int(count) for name, count in zip(CLASS_NAMES, counts)},
        "class_priors": {name: float(count / len(coordinates)) for name, count in zip(CLASS_NAMES, counts)},
        "spatial_bounds_row_col": {"min": [int(row_min), int(col_min)], "max": [int(row_max), int(col_max)]},
        "channel_statistics": _channel_statistics(scene, coordinates, scales, maximum, seed),
    }


def _matched_random_validation(scene, validation_coordinates, scales, maximum, seed):
    """Sample official-train pixels with exactly the spatial validation class counts."""
    rng = np.random.default_rng(seed)
    selected = []
    validation_labels = scene.train_labels[validation_coordinates[:, 0], validation_coordinates[:, 1]]
    for class_id in range(1, len(CLASS_NAMES) + 1):
        required = int((validation_labels == class_id).sum())
        candidates = np.argwhere(scene.train_labels == class_id)
        selected.append(candidates[rng.choice(len(candidates), size=required, replace=False)])
    coordinates = np.concatenate(selected, axis=0).astype(np.int32, copy=False)
    return _split_summary(scene, coordinates, scene.train_labels, scales, maximum, seed)


def _history_summary(path):
    if not path.is_file():
        return None
    history = json.loads(path.read_text(encoding="utf-8"))
    if not history:
        return None
    best = max(history, key=lambda item: item["selection"]["oa"])
    final = history[-1]
    return {
        "epochs_completed": len(history),
        "best_validation": {"epoch": best["epoch"], **best["selection"]},
        "last_epoch": {
            "epoch": final["epoch"],
            "train": final["train"],
            "validation": final["selection"],
        },
    }


def _markdown(report):
    lines = ["# Pol-InSAR Split Diagnostics", ""]
    history = report.get("fp_history")
    if history:
        lines.extend([
            "## FP History", "",
            "- Completed epochs: {}".format(history["epochs_completed"]),
            "- Best validation OA: {:.4f} at epoch {}".format(history["best_validation"]["oa"], history["best_validation"]["epoch"]),
            "- Last train / validation OA: {:.4f} / {:.4f}".format(history["last_epoch"]["train"]["oa"], history["last_epoch"]["validation"]["oa"]),
            "",
        ])
    lines.extend(["## Split Overview", "", "| Split | Pixels | Row/col coverage |", "|---|---:|---|"])
    for name, summary in report["splits"].items():
        bounds = summary["spatial_bounds_row_col"]
        lines.append("| {} | {} | {} to {} |".format(name, summary["pixels"], bounds["min"], bounds["max"]))
    lines.extend(["", "## Train-normalized Channel RMS", "", "| Split | " + " | ".join(channel["channel"] for channel in report["splits"]["train"]["channel_statistics"]["channels"]) + " |", "|---|" + "|".join(["---:"] * 6) + "|"])
    for name, summary in report["splits"].items():
        values = ["{:.3f}".format(channel["rms"]) for channel in summary["channel_statistics"]["channels"]]
        lines.append("| {} | {} |".format(name, " | ".join(values)))
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", default="datasets_external/pol_insar_island")
    parser.add_argument("--patch-size", type=int, default=11)
    parser.add_argument("--val-fraction", type=float, default=0.15)
    parser.add_argument("--block-size", type=int, default=64)
    parser.add_argument("--split-seed", type=int, default=0)
    parser.add_argument("--max-samples", type=int, default=200000)
    parser.add_argument("--run-root", default=None)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    scene = PolInSARScene.load(args.data_root)
    coordinates, split_report = official_split(scene, args.patch_size, args.val_fraction, args.block_size, args.split_seed)
    scales = train_scales(scene, coordinates["train"])
    report = {
        "configuration": vars(args),
        "official_split": split_report,
        "train_rms_scales": [float(value) for value in scales],
        "splits": {
            "train": _split_summary(scene, coordinates["train"], scene.train_labels, scales, args.max_samples, args.split_seed),
            "validation": _split_summary(scene, coordinates["validation"], scene.train_labels, scales, args.max_samples, args.split_seed + 1),
            "random_reference_matched_validation": _matched_random_validation(
                scene, coordinates["validation"], scales, args.max_samples, args.split_seed + 3
            ),
            "test": _split_summary(scene, coordinates["test"], scene.test_labels, scales, args.max_samples, args.split_seed + 2),
        },
    }
    if args.run_root:
        report["fp_history"] = _history_summary(Path(args.run_root) / "fp" / "history.json")
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.with_suffix(".json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    output.with_suffix(".md").write_text(_markdown(report), encoding="utf-8")


if __name__ == "__main__":
    main()

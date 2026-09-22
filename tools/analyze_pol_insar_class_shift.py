#!/usr/bin/env python3
"""Report class-conditional T3 diagonal power drift across Pol-InSAR splits."""

import argparse
import json
import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from datasets.pol_insar_island import CLASS_NAMES, PolInSARScene, official_split


def _sample(coordinates, maximum, rng):
    if len(coordinates) <= maximum:
        return coordinates
    return coordinates[rng.choice(len(coordinates), size=maximum, replace=False)]


def _statistics(scene, coordinates, labels, class_id, maximum, rng):
    selected = coordinates[labels[coordinates[:, 0], coordinates[:, 1]] == class_id]
    selected = _sample(selected, maximum, rng)
    rows, cols = selected[:, 0], selected[:, 1]
    result = {"pixels": int(len(selected))}
    for index, name in enumerate(("T11", "T22", "T33")):
        values = np.maximum(np.asarray(scene.channels[index][rows, cols].real), 1e-8)
        result[name] = {
            "rms": float(np.sqrt(np.mean(values ** 2))),
            "median": float(np.median(values)),
            "log_mean": float(np.log(values).mean()),
            "log_std": float(np.log(values).std()),
        }
    return result


def _markdown(report):
    lines = ["# Pol-InSAR Class-Conditional T3 Shift", "", "RMS values are raw diagonal powers; each split/class is sampled independently.", ""]
    for name in CLASS_NAMES:
        lines.extend(["## {}".format(name), "", "| Split | Pixels | T11 RMS | T22 RMS | T33 RMS |", "|---|---:|---:|---:|---:|"])
        for split in ("train", "validation", "test"):
            value = report["splits"][split][name]
            lines.append("| {} | {} | {:.4g} | {:.4g} | {:.4g} |".format(split, value["pixels"], value["T11"]["rms"], value["T22"]["rms"], value["T33"]["rms"]))
        lines.append("")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", default="datasets_external/pol_insar_island")
    parser.add_argument("--patch-size", type=int, default=11)
    parser.add_argument("--val-fraction", type=float, default=0.15)
    parser.add_argument("--block-size", type=int, default=16)
    parser.add_argument("--split-seed", type=int, default=0)
    parser.add_argument("--max-samples-per-class", type=int, default=100000)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    scene = PolInSARScene.load(args.data_root)
    coordinates, split_report = official_split(scene, args.patch_size, args.val_fraction, args.block_size, args.split_seed)
    labels = {"train": scene.train_labels, "validation": scene.train_labels, "test": scene.test_labels}
    rng = np.random.default_rng(args.split_seed)
    report = {
        "configuration": vars(args),
        "split": split_report,
        "splits": {
            split: {
                name: _statistics(scene, coordinates[split], labels[split], class_id, args.max_samples_per_class, rng)
                for class_id, name in enumerate(CLASS_NAMES, start=1)
            }
            for split in ("train", "validation", "test")
        },
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.with_suffix(".json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    output.with_suffix(".md").write_text(_markdown(report), encoding="utf-8")


if __name__ == "__main__":
    main()

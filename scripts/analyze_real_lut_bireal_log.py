#!/usr/bin/env python3
"""Summarize validation accuracy and LR from the real LUT-BiReal logs."""

import argparse
import json
import re
from pathlib import Path


ACCURACY_RE = re.compile(r"^ \* acc@1 ([0-9.]+)\s*$", re.MULTILINE)
LEARNING_RATE_RE = re.compile(
    r"^learning_rate:\s*([0-9.eE+-]+)\s*$",
    re.MULTILINE,
)


def summarize(path):
    path = Path(path)
    text = path.read_text(encoding="utf-8", errors="replace")
    accuracies = [float(value) for value in ACCURACY_RE.findall(text)]
    learning_rates = [
        float(value) for value in LEARNING_RATE_RE.findall(text)
    ]
    if not accuracies:
        raise ValueError("No final validation accuracy lines found in {}".format(path))
    best_index = max(range(len(accuracies)), key=accuracies.__getitem__)
    milestone_epochs = (1, 5, 10, 20, 50, 100, 150, 200, 256)
    milestones = {
        str(epoch): accuracies[epoch - 1]
        for epoch in milestone_epochs
        if epoch <= len(accuracies)
    }
    return {
        "path": str(path),
        "epochs": len(accuracies),
        "best_accuracy_percent": accuracies[best_index],
        "best_epoch": best_index + 1,
        "last_accuracy_percent": accuracies[-1],
        "first_learning_rate": (
            learning_rates[0] if learning_rates else None
        ),
        "last_learning_rate": (
            learning_rates[-1] if learning_rates else None
        ),
        "milestone_accuracy_percent": milestones,
    }


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("logs", nargs="+")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)
    results = [summarize(path) for path in args.logs]
    if args.json:
        print(json.dumps(results, indent=2, sort_keys=True))
        return
    for result in results:
        print("# {}".format(result["path"]))
        print("epochs: {}".format(result["epochs"]))
        print(
            "best: {:.3f}% at epoch {}".format(
                result["best_accuracy_percent"],
                result["best_epoch"],
            )
        )
        print("last: {:.3f}%".format(result["last_accuracy_percent"]))
        print(
            "lr: {} -> {}".format(
                result["first_learning_rate"],
                result["last_learning_rate"],
            )
        )
        print(
            "milestones: {}".format(
                result["milestone_accuracy_percent"]
            )
        )


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Summarize a Phase 1/2 training workdir from its persistent log."""

import argparse
import json
import re
import shlex
import sys
from pathlib import Path


EPOCH_PATTERN = re.compile(
    r"Epoch\s+(?P<epoch>\d+)\s+"
    r"train_loss:\s*(?P<train_loss>[-+0-9.eE]+),\s*"
    r"train_acc:\s*(?P<train_acc>[-+0-9.eE]+),\s*"
    r"val_loss:\s*(?P<val_loss>[-+0-9.eE]+),\s*"
    r"val_acc:\s*(?P<val_acc>[-+0-9.eE]+),\s*"
    r"test_loss:\s*(?P<test_loss>[-+0-9.eE]+),\s*"
    r"test_acc:\s*(?P<test_acc>[-+0-9.eE]+)"
)
CONFIG_PATTERN = re.compile(r"Configuration:\s*(\{.*\})")
INVOCATION_PATTERN = re.compile(r"INVOCATION:\s*(.*)$", re.MULTILINE)


def parse_epoch_rows(text):
    rows = []
    for match in EPOCH_PATTERN.finditer(text):
        row = {"epoch": int(match.group("epoch"))}
        for name in (
            "train_loss",
            "train_acc",
            "val_loss",
            "val_acc",
            "test_loss",
            "test_acc",
        ):
            row[name] = float(match.group(name))
        rows.append(row)
    return rows


def parse_configuration(text):
    configurations = []
    for match in CONFIG_PATTERN.finditer(text):
        try:
            configurations.append(json.loads(match.group(1)))
        except json.JSONDecodeError:
            continue
    if configurations:
        return configurations[-1]

    invocations = INVOCATION_PATTERN.findall(text)
    if not invocations:
        return {}
    try:
        tokens = shlex.split(invocations[-1])
    except ValueError:
        tokens = invocations[-1].split()
    configuration = {}
    option_map = {
        "--phase": ("phase", int),
        "--num-epochs": ("num_epochs", int),
        "--batch-size": ("batch_size", int),
        "--lr": ("lr", float),
        "--schedule": ("schedule", str),
    }
    for index, token in enumerate(tokens[:-1]):
        if token in option_map:
            name, converter = option_map[token]
            try:
                configuration[name] = converter(tokens[index + 1])
            except (TypeError, ValueError):
                pass
    return configuration


def summarize(workdir):
    workdir = Path(workdir)
    log_path = workdir / "logs" / "train.txt"
    if not log_path.is_file():
        raise FileNotFoundError("Training log not found: {}".format(log_path))
    text = log_path.read_text(encoding="utf-8", errors="replace")
    last_invocation = text.rfind("INVOCATION:")
    run_text = text[last_invocation:] if last_invocation >= 0 else text
    rows = parse_epoch_rows(run_text)
    if not rows:
        raise RuntimeError("No epoch rows found in {}".format(log_path))
    configuration = parse_configuration(run_text)

    last = rows[-1]
    requested_epochs = configuration.get("num_epochs")
    status = "unknown"
    if requested_epochs is not None:
        status = (
            "complete"
            if last["epoch"] >= requested_epochs
            else "incomplete"
        )
    best_val = max(rows, key=lambda row: row["val_acc"])
    best_test = max(rows, key=lambda row: row["test_acc"])
    result = {
        "workdir": str(workdir),
        "log": str(log_path),
        "status": status,
        "configuration": configuration,
        "epochs_logged": len(rows),
        "last": last,
        "best_validation": best_val,
        "best_test": best_test,
    }

    metrics_path = workdir / "phase_metrics.json"
    if metrics_path.is_file():
        try:
            result["saved_metrics"] = json.loads(
                metrics_path.read_text(encoding="utf-8")
            )
        except json.JSONDecodeError:
            result["saved_metrics_error"] = "invalid JSON"
    return result


def to_markdown(result):
    configuration = result["configuration"]
    last = result["last"]
    best_val = result["best_validation"]
    best_test = result["best_test"]
    lines = [
        "# Training Run Summary",
        "",
        "- Workdir: `{}`".format(result["workdir"]),
        "- Status: `{}`".format(result["status"]),
        "- Phase: `{}`".format(configuration.get("phase", "unknown")),
        "- Epochs: `{}/{}`".format(
            last["epoch"],
            configuration.get("num_epochs", "unknown"),
        ),
        "- Best validation: `{:.4f}` at epoch `{}`; test there: `{:.4f}`".format(
            best_val["val_acc"],
            best_val["epoch"],
            best_val["test_acc"],
        ),
        "- Best test: `{:.4f}` at epoch `{}`".format(
            best_test["test_acc"],
            best_test["epoch"],
        ),
        "- Last: train `{:.4f}`, val `{:.4f}`, test `{:.4f}`".format(
            last["train_acc"],
            last["val_acc"],
            last["test_acc"],
        ),
        "",
    ]
    return "\n".join(lines)


def parse_args(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("workdir")
    parser.add_argument(
        "--format",
        choices=["markdown", "json"],
        default="markdown",
    )
    parser.add_argument("--output", default=None)
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    result = summarize(args.workdir)
    if args.format == "json":
        output = json.dumps(result, indent=2, sort_keys=True) + "\n"
    else:
        output = to_markdown(result)
    if args.output is None:
        sys.stdout.write(output)
    else:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(output, encoding="utf-8")


if __name__ == "__main__":
    main()

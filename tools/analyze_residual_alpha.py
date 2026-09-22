#!/usr/bin/env python3
"""Summarize residual-alpha checkpoint-selection candidates from a flow history."""

import argparse
import json
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("history", type=Path)
    parser.add_argument("--metric", choices=("oa", "aa", "macro_f1"), default="oa")
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def candidate(record, metric):
    selection = record["selection"]
    return {
        "epoch": record["epoch"],
        "residual_alpha": record["residual_alpha"],
        "selection_{}".format(metric): selection[metric],
        "selection_oa": selection["oa"],
        "selection_aa": selection["aa"],
        "selection_macro_f1": selection["macro_f1"],
    }


def best(records, metric):
    return max(records, key=lambda record: record["selection"][metric])


def main():
    args = parse_args()
    history = json.loads(args.history.read_text(encoding="utf-8"))
    if not history:
        raise ValueError("History is empty")
    residual_records = [record for record in history if record["residual_alpha"] is not None]
    if not residual_records:
        raise ValueError("History contains no residual-alpha records")
    alpha_end = max(record["residual_alpha"] for record in residual_records)
    end_records = [
        record
        for record in residual_records
        if abs(record["residual_alpha"] - alpha_end) <= 1e-8
    ]
    best_any = best(residual_records, args.metric)
    best_end = best(end_records, args.metric)
    result = {
        "history": str(args.history),
        "metric": args.metric,
        "alpha_end": alpha_end,
        "best_any_alpha": candidate(best_any, args.metric),
        "best_at_alpha_end": candidate(best_end, args.metric),
        "selection_gain_any_minus_end": (
            best_any["selection"][args.metric] - best_end["selection"][args.metric]
        ),
    }
    rendered = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output is None:
        print(rendered, end="")
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")


if __name__ == "__main__":
    main()

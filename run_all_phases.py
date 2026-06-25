#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import json
import subprocess
import sys
from pathlib import Path

PHASE_NAMES = {
    1: "full-precision BiReal",
    2: "binary BiReal",
    3: "LUT-aware BNN",
    4: "LUTNN",
}


def parse_phase_epochs(value):
    if value is None:
        return [200, 200, 200, 400]
    parts = [p.strip() for p in value.split(",") if p.strip()]
    if len(parts) == 1:
        return [int(parts[0])] * 4
    if len(parts) != 4:
        raise argparse.ArgumentTypeError("--phase-epochs must be one integer or four comma-separated integers")
    return [int(p) for p in parts]


def load_phase_metrics(workdir, phase):
    metrics_path = Path(workdir) / "phase_metrics.json"
    if not metrics_path.is_file():
        raise FileNotFoundError(f"Missing metrics file: {metrics_path}")
    with metrics_path.open("r", encoding="utf-8") as f:
        all_metrics = json.load(f)
    key = f"phase{phase}"
    if key not in all_metrics:
        raise KeyError(f"{metrics_path} does not contain {key}; did this phase save a best checkpoint?")
    return all_metrics[key]


def main(argv=None):
    parser = argparse.ArgumentParser(description="Run phase1->phase4 training and summarize best metrics.")
    parser.add_argument("-d", "--datadir", default=".")
    parser.add_argument("-w", "--workdir", default="bi_workdir")
    parser.add_argument("--training-script", default="training.py")
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--lut-sets", default=1, type=int,
                        help="Number of independently learnable LUT pairs per LUT layer")
    parser.add_argument("--lut-allocation", default="layer", choices=["layer", "channel"],
                        help="Allocate LUT sets from a layer-shared pool or from per-output-channel pools")
    parser.add_argument("--lut-sets-per-channel", default=1, type=int,
                        help="Number of LUT pairs owned by each output channel when --lut-allocation=channel")
    parser.add_argument("--start-phase", default=1, type=int, choices=range(1, 5),
                        help="First phase to run; it loads the previous phase-specific checkpoint")
    parser.add_argument("--phase-epochs", type=parse_phase_epochs, default=None,
                        help="One epoch count for all phases, or four comma-separated counts, e.g. 200,200,200,100")
    parser.add_argument("extra_args", nargs=argparse.REMAINDER,
                        help="Arguments after -- are forwarded to training.py for every phase")
    args = parser.parse_args(argv)

    extra_args = args.extra_args
    if extra_args and extra_args[0] == "--":
        extra_args = extra_args[1:]

    phase_epochs = parse_phase_epochs(None) if args.phase_epochs is None else args.phase_epochs
    workdir = Path(args.workdir)
    workdir.mkdir(parents=True, exist_ok=True)

    summary = {}
    for phase in range(args.start_phase, 5):
        cmd = [
            args.python,
            args.training_script,
            "--phase", str(phase),
            "--num-epochs", str(phase_epochs[phase - 1]),
            "--datadir", args.datadir,
            "--workdir", args.workdir,
            "--lut-sets", str(args.lut_sets),
            "--lut-allocation", args.lut_allocation,
            "--lut-sets-per-channel", str(args.lut_sets_per_channel),
        ] + extra_args
        print("\n==> Running phase {}: {}".format(phase, PHASE_NAMES[phase]), flush=True)
        print(" ".join(cmd), flush=True)
        subprocess.run(cmd, check=True)

        metrics = load_phase_metrics(workdir, phase)
        summary[f"phase{phase}"] = metrics
        summary_path = workdir / "phase_summary.json"
        with summary_path.open("w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, sort_keys=True)

        print(
            "==> Phase {} best: acc={:.4f}, loss={:.6f}, epoch={}".format(
                phase,
                metrics["best_acc"],
                metrics["best_loss"],
                metrics["epoch"],
            ),
            flush=True,
        )

    print("\n==> Phase summary")
    for phase in sorted(int(key.removeprefix("phase")) for key in summary):
        metrics = summary[f"phase{phase}"]
        print(
            "phase{} {:>24s}  acc={:.4f}  loss={:.6f}  epoch={}".format(
                phase,
                PHASE_NAMES[phase],
                metrics["best_acc"],
                metrics["best_loss"],
                metrics["epoch"],
            )
        )
    print("Saved summary to {}".format(workdir / "phase_summary.json"))


if __name__ == "__main__":
    main()

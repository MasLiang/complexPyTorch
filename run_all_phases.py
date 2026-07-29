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
    3.5: "LUT5-aware activation QAT",
    4: "LUTNN",
}


def phase_tag(phase):
    numeric = float(phase)
    if numeric.is_integer():
        return str(int(numeric))
    return str(numeric).replace(".", "p")


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
    key = f"phase{phase_tag(phase)}"
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
    parser.add_argument("--lut-inputs", default=4, type=int, choices=[4, 5],
                        help="Physical LUT inputs; 5 adds phase dominance abs(x_r)>=abs(x_i)")
    parser.add_argument(
        "--phase3-lut5-checkpoint",
        default=None,
        type=Path,
        help="LUT4 Bestmodel_phase3.pt used to initialize Phase3 LUT5",
    )
    parser.add_argument(
        "--phase3-lut5-transition-epochs",
        default=None,
        type=int,
        help="Epochs used to mix LUT4 into the hard phase-conditioned LUT5 operation",
    )
    parser.add_argument(
        "--include-phase3p5",
        action="store_true",
        help=(
            "Run the optional LUT5-aware activation-QAT branch after Phase 3; "
            "Phase3.5 initializes from the Phase2 checkpoint"
        ),
    )
    parser.add_argument("--phase3p5-epochs", default=200, type=int)
    parser.add_argument("--phase3p5-warmup-epochs", default=5, type=int)
    parser.add_argument("--phase3p5-transition-epochs", default=None, type=int)
    parser.add_argument("--phase3p5-beta-start", default=1.0, type=float)
    parser.add_argument("--phase3p5-beta-end", default=8.0, type=float)
    parser.add_argument(
        "--phase3p5-schedule",
        default="cosine",
        choices=["constant", "cosine"],
    )
    parser.add_argument("--lut-logit-init", default=5.0, type=float)
    parser.add_argument("--lut-init-mode", default="binary", choices=["binary", "raw"])
    parser.add_argument("--lut-lr", default=None, type=float)
    parser.add_argument("--lut-tau-min", default=1.0, type=float)
    parser.add_argument("--lut-tau-max", default=10.0, type=float)
    parser.add_argument("--lut-anneal-epochs", default=None, type=int)
    parser.add_argument("--lut-hard-ste", action="store_true")
    parser.add_argument("--lut-hard-epoch-fraction", default=0.9, type=float)
    parser.add_argument("--schedule", default="default", choices=["default", "constant", "cosine", "bireal"])
    parser.add_argument("--min-lr-factor", default=0.1, type=float)
    parser.add_argument("--start-phase", default=1, type=int, choices=range(1, 5),
                        help="First phase to run; it loads the previous phase-specific checkpoint")
    parser.add_argument("--phase-epochs", type=parse_phase_epochs, default=None,
                        help="One epoch count for all phases, or four comma-separated counts, e.g. 200,200,200,100")
    parser.add_argument("extra_args", nargs=argparse.REMAINDER,
                        help="Arguments after -- are forwarded to training.py for every phase")
    args = parser.parse_args(argv)

    if args.include_phase3p5 and args.lut_inputs != 5:
        parser.error("--include-phase3p5 requires --lut-inputs 5")

    runs_phase3_lut5 = (
        args.lut_inputs == 5
        and args.start_phase <= 3
        and not args.include_phase3p5
    )
    if runs_phase3_lut5:
        if args.phase3_lut5_checkpoint is None:
            parser.error(
                "--lut-inputs 5 requires --phase3-lut5-checkpoint when Phase 3 is run"
            )
        if not args.phase3_lut5_checkpoint.is_file():
            parser.error(
                "Phase3 LUT5 checkpoint not found: {}".format(
                    args.phase3_lut5_checkpoint
                )
            )
        if args.schedule not in ("constant", "cosine"):
            parser.error(
                "Phase3 LUT5 requires --schedule constant or cosine"
            )

    extra_args = args.extra_args
    if extra_args and extra_args[0] == "--":
        extra_args = extra_args[1:]

    phase_epochs = parse_phase_epochs(None) if args.phase_epochs is None else args.phase_epochs
    workdir = Path(args.workdir)
    workdir.mkdir(parents=True, exist_ok=True)

    phases = list(range(args.start_phase, 5))
    if args.include_phase3p5 and args.start_phase <= 3:
        phases.insert(phases.index(4), 3.5)

    summary = {}
    for phase in phases:
        if phase == 3.5:
            num_epochs = args.phase3p5_epochs
            phase_schedule = args.phase3p5_schedule
            phase_lut_inputs = 5
        else:
            num_epochs = phase_epochs[int(phase) - 1]
            phase_schedule = args.schedule
            phase_lut_inputs = (
                4
                if args.include_phase3p5 and phase <= 3
                else args.lut_inputs
            )
        cmd = [
            args.python,
            args.training_script,
            "--phase", str(phase),
            "--num-epochs", str(num_epochs),
            "--datadir", args.datadir,
            "--workdir", args.workdir,
            "--lut-sets", str(args.lut_sets),
            "--lut-allocation", args.lut_allocation,
            "--lut-sets-per-channel", str(args.lut_sets_per_channel),
            "--lut-inputs", str(phase_lut_inputs),
            "--lut-logit-init", str(args.lut_logit_init),
            "--lut-init-mode", args.lut_init_mode,
            "--lut-tau-min", str(args.lut_tau_min),
            "--lut-tau-max", str(args.lut_tau_max),
            "--lut-hard-epoch-fraction", str(args.lut_hard_epoch_fraction),
            "--schedule", phase_schedule,
            "--min-lr-factor", str(args.min_lr_factor),
        ]
        if args.lut_anneal_epochs is not None:
            cmd.extend(["--lut-anneal-epochs", str(args.lut_anneal_epochs)])
        if args.lut_hard_ste:
            cmd.append("--lut-hard-ste")
        if args.lut_lr is not None:
            cmd.extend(["--lut-lr", str(args.lut_lr)])
        if phase == 3 and runs_phase3_lut5:
            cmd.extend([
                "--checkpoint",
                str(args.phase3_lut5_checkpoint),
            ])
            if args.phase3_lut5_transition_epochs is not None:
                cmd.extend([
                    "--phase3-lut5-transition-epochs",
                    str(args.phase3_lut5_transition_epochs),
                ])
        if phase == 3.5:
            cmd.extend([
                "--phase3p5-warmup-epochs",
                str(args.phase3p5_warmup_epochs),
                "--phase3p5-beta-start",
                str(args.phase3p5_beta_start),
                "--phase3p5-beta-end",
                str(args.phase3p5_beta_end),
            ])
            if args.phase3p5_transition_epochs is not None:
                cmd.extend([
                    "--phase3p5-transition-epochs",
                    str(args.phase3p5_transition_epochs),
                ])
        cmd += extra_args
        print("\n==> Running phase {}: {}".format(phase, PHASE_NAMES[phase]), flush=True)
        print(" ".join(cmd), flush=True)
        subprocess.run(cmd, check=True)

        metrics = load_phase_metrics(workdir, phase)
        summary[f"phase{phase_tag(phase)}"] = metrics
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
    for phase in phases:
        key = f"phase{phase_tag(phase)}"
        if key not in summary:
            continue
        metrics = summary[key]
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

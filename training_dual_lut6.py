#!/usr/bin/env python3
"""Train the isolated Phase2.7 dual-LUT6_2 complex-product flow."""

import argparse
import json
import math
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from complexPyTorch.dualLut6Flow import (
    DUAL_LUT6_PHASE,
    DualLUT6ComplexResNet,
    TwoBitOccupancyTracker,
    dual_lut6_hardware_spec,
    iter_two_bit_activations,
    set_dual_lut6_implementation,
    set_two_bit_rho,
    two_bit_parameter_diagnostics,
)
from training import (
    build_datasets,
    evaluate,
    get_lr_for_epoch,
    save_checkpoint,
    set_seed,
    setup_logging,
)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Phase2.7: 2-bit complex activation and exact dual-LUT6_2 product"
    )
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("-d", "--datadir", default="data")
    parser.add_argument("-w", "--workdir", required=True)
    parser.add_argument(
        "--dataset",
        default="cifar10",
        choices=["cifar10", "cifar100", "svhn"],
    )
    parser.add_argument("--num-epochs", default=200, type=int)
    parser.add_argument("--batch-size", default=128, type=int)
    parser.add_argument("--num-workers", default=4, type=int)
    parser.add_argument("--start-filter", default=11, type=int)
    parser.add_argument("--num-blocks", default=3, type=int)
    parser.add_argument(
        "--spectral-pool-scheme",
        default="none",
        choices=["none", "stagemiddle", "proj", "nodownsample"],
    )
    parser.add_argument("--spectral-pool-gamma", default=0.5, type=float)

    parser.add_argument("--threshold-init", default=0.675, type=float)
    parser.add_argument("--threshold-lr", default=0.001, type=float)
    parser.add_argument("--magnitude-beta", default=4.0, type=float)
    parser.add_argument("--high-ratio", default=3.0, type=float)
    parser.add_argument("--rho-warmup-epochs", default=10, type=int)
    parser.add_argument("--rho-transition-epochs", default=80, type=int)
    parser.add_argument(
        "--rho-schedule",
        default="cosine",
        choices=["linear", "cosine"],
    )

    parser.add_argument("--optimizer", default="sgd", choices=["sgd", "adamw"])
    parser.add_argument("--lr", default=0.01, type=float)
    parser.add_argument("--momentum", default=0.9, type=float)
    parser.add_argument("--weight-decay", default=0.0, type=float)
    parser.add_argument(
        "--schedule",
        default="cosine",
        choices=["default", "constant", "cosine", "bireal"],
    )
    parser.add_argument("--min-lr-factor", default=0.1, type=float)
    parser.add_argument("--clipnorm", default=1.0, type=float)
    parser.add_argument("--clipval", default=1.0, type=float)

    parser.add_argument("--seed", default=0xE4223644E98B8E64, type=int)
    parser.add_argument("--no-validation", action="store_true")
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument(
        "--loglevel",
        default="info",
        choices=["none", "debug", "info", "warn", "err", "crit"],
    )
    return parser.parse_args(argv)


def validate_args(args):
    if args.num_epochs < 1:
        raise ValueError("num_epochs must be positive")
    if args.threshold_init <= 0.0:
        raise ValueError("threshold_init must be positive")
    if args.threshold_lr < 0.0:
        raise ValueError("threshold_lr must be non-negative")
    if args.magnitude_beta <= 0.0:
        raise ValueError("magnitude_beta must be positive")
    if args.high_ratio <= 1.0:
        raise ValueError("high_ratio must be greater than one")
    if args.rho_warmup_epochs < 0 or args.rho_transition_epochs < 1:
        raise ValueError("rho schedule lengths are invalid")
    first_full_epoch = args.rho_warmup_epochs + args.rho_transition_epochs
    if args.num_epochs < first_full_epoch:
        raise ValueError(
            "num_epochs must be at least {} so training reaches rho=1".format(
                first_full_epoch
            )
        )
    if args.lr <= 0.0:
        raise ValueError("lr must be positive")


def _strip_wrappers(key):
    changed = True
    while changed:
        changed = False
        for prefix in ("_orig_mod.", "module."):
            if key.startswith(prefix):
                key = key[len(prefix):]
                changed = True
    return key


def _checkpoint_phase(checkpoint):
    for container in (
        checkpoint,
        checkpoint.get("metrics", {}) if isinstance(checkpoint, dict) else {},
        checkpoint.get("args", {}) if isinstance(checkpoint, dict) else {},
    ):
        if isinstance(container, dict) and container.get("phase") is not None:
            return float(container["phase"])
    return None


def load_phase2_anchor(model, checkpoint_path):
    path = Path(checkpoint_path)
    if not path.is_file():
        raise FileNotFoundError("Checkpoint not found: {}".format(path))
    checkpoint = torch.load(path, map_location="cpu")
    phase = _checkpoint_phase(checkpoint)
    if phase is not None and phase != 2.0:
        raise ValueError(
            "Phase2.7 requires a Phase2 checkpoint, but {} stores Phase {}".format(
                path, phase
            )
        )
    source = checkpoint.get(
        "model", checkpoint.get("state_dict", checkpoint)
    )
    target = model.state_dict()
    target_by_clean_key = {
        _strip_wrappers(key): key for key in target
    }
    compatible = {}
    skipped = []
    for key, value in source.items():
        target_key = target_by_clean_key.get(_strip_wrappers(key))
        if target_key is None or target[target_key].shape != value.shape:
            skipped.append(_strip_wrappers(key))
            continue
        compatible[target_key] = value

    missing, unexpected = model.load_state_dict(compatible, strict=False)
    allowed_missing_suffixes = (
        "threshold_unconstrained",
        "rho",
        "beta",
        "high_ratio",
    )
    unexpected_missing = [
        key for key in missing
        if not key.endswith(allowed_missing_suffixes)
    ]
    if unexpected_missing or unexpected:
        raise RuntimeError(
            "Phase2 mapping is incomplete: missing={}, unexpected={}".format(
                unexpected_missing,
                unexpected,
            )
        )
    source_tensor_count = len(source)
    if skipped:
        raise RuntimeError(
            "Phase2 mapping skipped {} of {} source tensors: {}".format(
                len(skipped),
                source_tensor_count,
                skipped[:20],
            )
        )
    return checkpoint, len(compatible), len(missing)


def build_model(args, num_classes):
    return DualLUT6ComplexResNet(
        num_blocks=args.num_blocks,
        start_filters=args.start_filter,
        num_classes=num_classes,
        spectral_pool_scheme=args.spectral_pool_scheme,
        spectral_pool_gamma=args.spectral_pool_gamma,
        is_sar_input=False,
        threshold_init=args.threshold_init,
        high_ratio=args.high_ratio,
        magnitude_beta=args.magnitude_beta,
    )


def build_optimizer(model, args):
    spatial_parameters = []
    threshold_parameters = []
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        if name.endswith("threshold_unconstrained"):
            threshold_parameters.append(parameter)
        else:
            spatial_parameters.append(parameter)
    groups = [
        {
            "params": spatial_parameters,
            "lr": args.lr,
            "weight_decay": args.weight_decay,
            "group_name": "spatial",
        },
        {
            "params": threshold_parameters,
            "lr": args.threshold_lr,
            "weight_decay": 0.0,
            "group_name": "threshold",
        },
    ]
    if args.optimizer == "sgd":
        return torch.optim.SGD(groups, momentum=args.momentum)
    return torch.optim.AdamW(groups)


def rho_for_epoch(epoch, args):
    if epoch < args.rho_warmup_epochs:
        return 0.0
    offset = epoch - args.rho_warmup_epochs + 1
    progress = min(
        max(offset / float(args.rho_transition_epochs), 0.0),
        1.0,
    )
    if args.rho_schedule == "cosine":
        progress = 0.5 - 0.5 * math.cos(math.pi * progress)
    return float(progress)


def update_learning_rates(optimizer, epoch, args):
    base_lr = get_lr_for_epoch(epoch, args)
    threshold_scale = args.threshold_lr / args.lr
    for group in optimizer.param_groups:
        if group["group_name"] == "threshold":
            group["lr"] = base_lr * threshold_scale
        else:
            group["lr"] = base_lr
    return {
        group["group_name"]: float(group["lr"])
        for group in optimizer.param_groups
    }


def train_epoch(model, loader, optimizer, device, args):
    model.train()
    loss_sum = 0.0
    correct = 0
    count = 0
    for data, target in loader:
        data = data.to(device)
        target = target.to(device)
        optimizer.zero_grad(set_to_none=True)
        output = model(data)
        loss = F.cross_entropy(output, target)
        loss.backward()
        if args.clipnorm > 0.0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.clipnorm)
        if args.clipval > 0.0:
            torch.nn.utils.clip_grad_value_(model.parameters(), args.clipval)
        optimizer.step()
        loss_sum += loss.item() * data.size(0)
        correct += (output.argmax(dim=1) == target).sum().item()
        count += data.size(0)
    return (
        loss_sum / float(count),
        correct / float(count),
    )


def evaluate_all(model, val_loader, test_loader, device, track_occupancy=False):
    val_loss, val_acc = (
        evaluate(model, val_loader, device)
        if val_loader is not None
        else (0.0, 0.0)
    )
    tracker = TwoBitOccupancyTracker(model) if track_occupancy else None
    test_loss, test_acc = evaluate(model, test_loader, device)
    occupancy = tracker.finish() if tracker is not None else None
    return {
        "val_loss": float(val_loss),
        "val_acc": float(val_acc),
        "test_loss": float(test_loss),
        "test_acc": float(test_acc),
    }, occupancy


@torch.no_grad()
def verify_implementation_equivalence(model, loader, device):
    data, _ = next(iter(loader))
    data = data.to(device)
    training = model.training
    model.eval()
    set_dual_lut6_implementation(model, "standard")
    standard = model(data)
    set_dual_lut6_implementation(model, "hardware")
    hardware = model(data)
    set_dual_lut6_implementation(model, "standard")
    model.train(training)
    difference = (standard - hardware).abs()
    return {
        "max_abs_logit_difference": float(difference.max().item()),
        "mean_abs_logit_difference": float(difference.mean().item()),
        "prediction_mismatches": int(
            (standard.argmax(dim=1) != hardware.argmax(dim=1)).sum().item()
        ),
        "samples": int(data.size(0)),
    }


def checkpoint_state(
    model,
    optimizer,
    args,
    epoch,
    metrics,
    diagnostics,
    occupancy,
    equivalence,
):
    return {
        "phase": DUAL_LUT6_PHASE,
        "flow_phase": DUAL_LUT6_PHASE,
        "flow_name": "dual LUT6_2 exact 2-bit complex product",
        "model": {
            key: value.detach().clone()
            for key, value in model.state_dict().items()
        },
        "optimizer": optimizer.state_dict(),
        "epoch": int(epoch),
        "args": vars(args),
        "metrics": metrics,
        "two_bit_diagnostics": diagnostics,
        "two_bit_occupancy": occupancy,
        "implementation_equivalence": equivalence,
        "hardware_spec": dual_lut6_hardware_spec(args.high_ratio),
        "hardware_deployable": diagnostics["rho"] >= 1.0 - 1e-8,
        "source_checkpoint": str(args.checkpoint),
    }


def update_metrics_file(workdir, metrics, diagnostics, occupancy, checkpoint):
    payload = {
        "phase": DUAL_LUT6_PHASE,
        "flow_name": "dual LUT6_2 exact 2-bit complex product",
        "checkpoint": str(checkpoint),
        "metrics": metrics,
        "two_bit_diagnostics": diagnostics,
        "two_bit_occupancy": occupancy,
        "hardware_spec": dual_lut6_hardware_spec(
            diagnostics.get("high_ratio", 3.0)
        ),
    }
    path = Path(workdir) / "dual_lut6_metrics.json"
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)


def main(argv=None):
    args = parse_args(argv)
    validate_args(args)
    set_seed(args.seed)
    workdir = Path(args.workdir)
    workdir.mkdir(parents=True, exist_ok=True)
    _, train_logger = setup_logging(str(workdir), args.loglevel, True)
    train_logger.info("INVOCATION: %s", " ".join([sys.executable, *sys.argv]))
    train_logger.info(
        "Phase2.7 dual LUT6_2 flow: threshold_init=%.4f threshold_lr=%.6f high_ratio=%.3f beta=%.3f rho=%s(%d warmup + %d transition)",
        args.threshold_init,
        args.threshold_lr,
        args.high_ratio,
        args.magnitude_beta,
        args.rho_schedule,
        args.rho_warmup_epochs,
        args.rho_transition_epochs,
    )

    train_set, val_set, test_set, num_classes, _, _ = build_datasets(
        args,
        train_logger,
        ddp_enabled=False,
        is_main=True,
    )
    loader_kwargs = {
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
        "pin_memory": not args.cpu,
        "persistent_workers": args.num_workers > 0,
    }
    train_loader = DataLoader(train_set, shuffle=True, **loader_kwargs)
    val_loader = (
        DataLoader(val_set, shuffle=False, **loader_kwargs)
        if val_set is not None
        else None
    )
    test_loader = DataLoader(test_set, shuffle=False, **loader_kwargs)

    model = build_model(args, num_classes)
    _, mapped, missing = load_phase2_anchor(model, args.checkpoint)
    train_logger.info(
        "Loaded Phase2 anchor: mapped=%d, new_phase2p7_state=%d",
        mapped,
        missing,
    )
    set_two_bit_rho(model, 0.0)
    device = torch.device(
        "cpu" if args.cpu or not torch.cuda.is_available() else "cuda"
    )
    model = model.to(device)
    optimizer = build_optimizer(model, args)
    train_logger.info(
        "Device=%s trainable_parameters=%d two_bit_activations=%d",
        device,
        sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad),
        sum(1 for _ in iter_two_bit_activations(model)),
    )

    metric_name = "test" if args.no_validation else "val"
    initial_metrics, occupancy = evaluate_all(
        model,
        val_loader,
        test_loader,
        device,
        track_occupancy=True,
    )
    diagnostics = two_bit_parameter_diagnostics(model)
    diagnostics["high_ratio"] = float(args.high_ratio)
    equivalence = verify_implementation_equivalence(model, test_loader, device)
    train_logger.info(
        "Initial Phase2 anchor: val_acc=%.4f test_acc=%.4f rho=%.4f levels=(%.6f, %.6f) hardware_max_diff=%.3e pred_mismatch=%d/%d",
        initial_metrics["val_acc"],
        initial_metrics["test_acc"],
        diagnostics["rho"],
        diagnostics["level_low"],
        diagnostics["level_high"],
        equivalence["max_abs_logit_difference"],
        equivalence["prediction_mismatches"],
        equivalence["samples"],
    )

    last_filename = "Lastmodel_phase2p7.pt"
    initial_state = checkpoint_state(
        model,
        optimizer,
        args,
        0,
        initial_metrics,
        diagnostics,
        occupancy,
        equivalence,
    )
    last_checkpoint = save_checkpoint(
        initial_state,
        str(workdir),
        last_filename,
    )
    best_acc = -1.0
    best_loss = float("inf")
    best_metrics = None
    best_checkpoint = None
    first_full_rho_checked = False
    last_occupancy = occupancy
    last_equivalence = equivalence

    for epoch in range(args.num_epochs):
        epoch_start = time.time()
        rho = rho_for_epoch(epoch, args)
        set_two_bit_rho(model, rho)
        group_lrs = update_learning_rates(optimizer, epoch, args)
        train_loss, train_acc = train_epoch(
            model,
            train_loader,
            optimizer,
            device,
            args,
        )
        track_occupancy = (
            epoch == args.num_epochs - 1
            or (epoch + 1) % 10 == 0
            or (rho >= 1.0 - 1e-8 and not first_full_rho_checked)
        )
        metrics, current_occupancy = evaluate_all(
            model,
            val_loader,
            test_loader,
            device,
            track_occupancy=track_occupancy,
        )
        if current_occupancy is not None:
            last_occupancy = current_occupancy
        diagnostics = two_bit_parameter_diagnostics(model)
        diagnostics["high_ratio"] = float(args.high_ratio)

        if rho >= 1.0 - 1e-8 and not first_full_rho_checked:
            last_equivalence = verify_implementation_equivalence(
                model, test_loader, device
            )
            first_full_rho_checked = True
            train_logger.info(
                "First full 2-bit hardware check: max_diff=%.3e mean_diff=%.3e pred_mismatch=%d/%d",
                last_equivalence["max_abs_logit_difference"],
                last_equivalence["mean_abs_logit_difference"],
                last_equivalence["prediction_mismatches"],
                last_equivalence["samples"],
            )

        selection_acc = metrics[
            "test_acc" if args.no_validation else "val_acc"
        ]
        selection_loss = metrics[
            "test_loss" if args.no_validation else "val_loss"
        ]
        deployable = rho >= 1.0 - 1e-8
        improved = deployable and (
            selection_acc > best_acc
            or (
                selection_acc == best_acc
                and selection_loss < best_loss
            )
        )
        epoch_metrics = {
            "phase": DUAL_LUT6_PHASE,
            "epoch": epoch + 1,
            "train_loss": float(train_loss),
            "train_acc": float(train_acc),
            "best_acc_so_far": float(best_acc),
            **metrics,
        }
        if improved:
            best_acc = selection_acc
            best_loss = selection_loss
            epoch_metrics["best_acc_so_far"] = float(best_acc)
            best_metrics = dict(epoch_metrics)
            best_state = checkpoint_state(
                model,
                optimizer,
                args,
                epoch + 1,
                best_metrics,
                diagnostics,
                last_occupancy,
                last_equivalence,
            )
            best_checkpoint = save_checkpoint(
                best_state,
                str(workdir),
                "Bestmodel_phase2p7.pt",
            )
            update_metrics_file(
                workdir,
                best_metrics,
                diagnostics,
                last_occupancy,
                best_checkpoint,
            )

        last_state = checkpoint_state(
            model,
            optimizer,
            args,
            epoch + 1,
            epoch_metrics,
            diagnostics,
            last_occupancy,
            last_equivalence,
        )
        last_checkpoint = save_checkpoint(
            last_state,
            str(workdir),
            last_filename,
        )
        train_logger.info(
            "Epoch %d train_loss=%.6f train_acc=%.4f val_loss=%.6f val_acc=%.4f test_loss=%.6f test_acc=%.4f rho=%.4f levels=(%.5f,%.5f) threshold=(%.4f,%.4f,%.4f) lrs=%s deployable=%s best_%s=%.4f time=%.2fs",
            epoch + 1,
            train_loss,
            train_acc,
            metrics["val_loss"],
            metrics["val_acc"],
            metrics["test_loss"],
            metrics["test_acc"],
            diagnostics["rho"],
            diagnostics["level_low"],
            diagnostics["level_high"],
            diagnostics["threshold_min"],
            diagnostics["threshold_mean"],
            diagnostics["threshold_max"],
            group_lrs,
            deployable,
            metric_name,
            best_acc,
            time.time() - epoch_start,
        )

    if best_checkpoint is None or best_metrics is None:
        raise RuntimeError("Training reached no deployable rho=1 checkpoint")
    train_logger.info(
        "Completed Phase2.7: best_%s_acc=%.4f at epoch=%d best_checkpoint=%s endpoint_checkpoint=%s",
        metric_name,
        best_acc,
        best_metrics["epoch"],
        best_checkpoint,
        last_checkpoint,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

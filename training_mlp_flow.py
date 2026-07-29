#!/usr/bin/env python3
"""Train the isolated Phase2.2 -> Phase4.2 binary-MLP flow."""

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from complexPyTorch.mlpFlow import (
    MLPFlowComplexResNet,
    clamp_binary_mlp_latents,
    export_hard_lut_tables,
    initialize_phase4_luts,
    iter_mlp_operation_layers,
    operation_diagnostics,
    set_phase4_lut_state,
)
from training import (
    build_datasets,
    evaluate,
    get_lr_for_epoch,
    load_phase_checkpoint,
    save_checkpoint,
    set_seed,
    setup_logging,
)


FLOW_PHASE_NAMES = {
    1.2: "legacy floating MLP operation",
    2.2: "binary MLP BNN with multi-level local scores",
    3.2: "binary MLP BNN with local output truncation",
    4.2: "compiled and trainable LUT5",
}
FLOW_PREDECESSORS = {1.2: 1.0, 2.2: 1.0, 3.2: 2.2, 4.2: 3.2}


def phase_tag(phase):
    return str(float(phase)).replace(".", "p")


def parse_flow_phase(value):
    phase = float(value)
    if phase not in FLOW_PHASE_NAMES:
        raise argparse.ArgumentTypeError(
            "flow phase must be one of {}".format(sorted(FLOW_PHASE_NAMES))
        )
    return phase


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Isolated MLP-operation complex network training"
    )
    parser.add_argument("--flow-phase", required=True, type=parse_flow_phase)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("-d", "--datadir", default="data")
    parser.add_argument("-w", "--workdir", required=True)
    parser.add_argument("--dataset", default="cifar10", choices=["cifar10", "cifar100", "svhn"])
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
    parser.add_argument("--operation-sets", default=1, type=int)
    parser.add_argument(
        "--operation-allocation",
        default="layer",
        choices=["layer", "channel"],
    )
    parser.add_argument("--operation-sets-per-channel", default=1, type=int)
    parser.add_argument("--dominance-beta", default=2.0, type=float)
    parser.add_argument("--binary-mlp-hidden", default=64, type=int)
    parser.add_argument("--binary-mlp-logit-init", default=0.5, type=float)
    parser.add_argument("--lut-logit-init", default=2.0, type=float)
    parser.add_argument("--phase3p2-score-surrogate", action="store_true")
    parser.add_argument(
        "--phase3p2-freeze-operation-epochs",
        default=0,
        type=int,
    )
    parser.add_argument(
        "--phase3p2-bn-calibration-batches",
        default=0,
        type=int,
        help="0 disables calibration; -1 uses the complete training loader",
    )

    parser.add_argument("--optimizer", default="sgd", choices=["sgd", "adamw"])
    parser.add_argument("--lr", default=0.001, type=float)
    parser.add_argument("--operation-lr", default=0.001, type=float)
    parser.add_argument("--lut-lr", default=0.001, type=float)
    parser.add_argument("--momentum", default=0.9, type=float)
    parser.add_argument("--weight-decay", default=0.0, type=float)
    parser.add_argument(
        "--schedule",
        default="cosine",
        choices=["default", "constant", "cosine", "bireal"],
    )
    parser.add_argument("--min-lr-factor", default=0.1, type=float)
    parser.add_argument(
        "--operation-schedule",
        default="constant",
        choices=["constant", "follow-base"],
    )
    parser.add_argument(
        "--lut-schedule",
        default="constant",
        choices=["constant", "follow-base"],
    )
    parser.add_argument("--clipnorm", default=1.0, type=float)
    parser.add_argument("--clipval", default=1.0, type=float)

    parser.add_argument("--lut-tau-min", default=1.0, type=float)
    parser.add_argument("--lut-tau-max", default=10.0, type=float)
    parser.add_argument("--lut-soft-warmup-epochs", default=10, type=int)
    parser.add_argument("--lut-anneal-epochs", default=140, type=int)
    parser.add_argument("--lut-hard-transition-epochs", default=20, type=int)

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
    if args.phase3p2_freeze_operation_epochs < 0:
        raise ValueError("phase3p2_freeze_operation_epochs must be non-negative")
    if args.phase3p2_bn_calibration_batches < -1:
        raise ValueError(
            "phase3p2_bn_calibration_batches must be -1 or non-negative"
        )
    phase3_strategy_enabled = (
        args.phase3p2_score_surrogate
        or args.phase3p2_freeze_operation_epochs > 0
        or args.phase3p2_bn_calibration_batches != 0
    )
    if phase3_strategy_enabled and args.flow_phase != 3.2:
        raise ValueError(
            "Phase3.2 recovery options can only be used with --flow-phase 3.2"
        )


def _strip_wrappers(key):
    changed = True
    while changed:
        changed = False
        for prefix in ("_orig_mod.", "module."):
            if key.startswith(prefix):
                key = key[len(prefix):]
                changed = True
    return key


def load_flow_checkpoint(model, path, expected_phase, device="cpu"):
    checkpoint = torch.load(path, map_location=device)
    stored_phase = checkpoint.get("flow_phase")
    if stored_phase is None:
        stored_phase = checkpoint.get("phase")
    if stored_phase is None or float(stored_phase) != float(expected_phase):
        raise ValueError(
            "Checkpoint {} is Phase {}, but Phase {} is required".format(
                path,
                stored_phase,
                expected_phase,
            )
        )
    stored_args = checkpoint.get("args", {})
    for key, expected in (
        ("operation_allocation", model.operation_allocation),
        ("operation_sets", model.operation_sets),
        ("operation_sets_per_channel", model.operation_sets_per_channel),
        ("binary_mlp_hidden", model.binary_mlp_hidden),
        ("binary_mlp_logit_init", model.binary_mlp_logit_init),
    ):
        stored = stored_args.get(key)
        if stored is not None and stored != expected:
            raise ValueError(
                "Checkpoint {} uses {}={}, but {} is required".format(
                    path,
                    key,
                    stored,
                    expected,
                )
            )

    source = checkpoint.get("model", checkpoint.get("state_dict", checkpoint))
    target = model.state_dict()
    clean_target = {_strip_wrappers(key): key for key in target}
    compatible = {}
    skipped = []
    for key, value in source.items():
        target_key = clean_target.get(_strip_wrappers(key))
        if target_key is None or target[target_key].shape != value.shape:
            skipped.append(_strip_wrappers(key))
            continue
        compatible[target_key] = value
    missing, unexpected = model.load_state_dict(compatible, strict=False)
    print(
        "==> Loaded {} compatible tensors from {} (missing={}, skipped={}, unexpected={})".format(
            len(compatible),
            path,
            len(missing),
            len(skipped),
            len(unexpected),
        )
    )
    return checkpoint


def build_model(args, num_classes):
    return MLPFlowComplexResNet(
        num_blocks=args.num_blocks,
        start_filters=args.start_filter,
        num_classes=num_classes,
        phase=args.flow_phase,
        spectral_pool_scheme=args.spectral_pool_scheme,
        spectral_pool_gamma=args.spectral_pool_gamma,
        is_sar_input=False,
        operation_sets=args.operation_sets,
        operation_allocation=args.operation_allocation,
        operation_sets_per_channel=args.operation_sets_per_channel,
        dominance_beta=args.dominance_beta,
        binary_mlp_hidden=args.binary_mlp_hidden,
        binary_mlp_logit_init=args.binary_mlp_logit_init,
        lut_logit_init=args.lut_logit_init,
        phase3p2_score_surrogate=args.phase3p2_score_surrogate,
    )


def initialize_from_predecessor(model, args):
    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.is_file():
        raise FileNotFoundError("Checkpoint not found: {}".format(checkpoint_path))
    predecessor = FLOW_PREDECESSORS[args.flow_phase]
    if args.flow_phase in (1.2, 2.2):
        load_phase_checkpoint(
            model,
            str(checkpoint_path),
            device="cpu",
            expected_phase=1,
        )
    else:
        load_flow_checkpoint(
            model,
            str(checkpoint_path),
            expected_phase=predecessor,
            device="cpu",
        )
    if args.flow_phase == 4.2:
        initialized = initialize_phase4_luts(model)
        for module in iter_mlp_operation_layers(model):
            for parameter in (
                module.operation_coefficients,
                module.binary_mlp_hidden_weight,
                module.binary_mlp_output_weight,
            ):
                if parameter is not None:
                    parameter.requires_grad_(False)
        print("==> Compiled {} trained binary MLP operators into direct LUT5 tables".format(initialized))


def build_optimizer(model, args):
    spatial_parameters = []
    operation_parameters = []
    lut_parameters = []
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        if (
            name.endswith("operation_coefficients")
            or name.endswith("binary_mlp_hidden_weight")
            or name.endswith("binary_mlp_output_weight")
        ):
            operation_parameters.append(parameter)
        elif name.endswith("lut_r") or name.endswith("lut_i"):
            lut_parameters.append(parameter)
        else:
            spatial_parameters.append(parameter)

    groups = []
    if spatial_parameters:
        groups.append(
            {
                "params": spatial_parameters,
                "lr": args.lr,
                "weight_decay": args.weight_decay,
                "group_name": "spatial",
            }
        )
    if operation_parameters:
        groups.append(
            {
                "params": operation_parameters,
                "lr": args.operation_lr,
                "weight_decay": 0.0,
                "group_name": "operation",
            }
        )
    if lut_parameters:
        groups.append(
            {
                "params": lut_parameters,
                "lr": args.lut_lr,
                "weight_decay": 0.0,
                "group_name": "lut",
            }
        )
    if args.optimizer == "sgd":
        return torch.optim.SGD(groups, momentum=args.momentum)
    return torch.optim.AdamW(groups)


def train_flow_epoch(model, loader, optimizer, device, clipnorm, clipval):
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
        if clipnorm is not None and clipnorm > 0.0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), clipnorm)
        if clipval is not None and clipval > 0.0:
            torch.nn.utils.clip_grad_value_(model.parameters(), clipval)
        optimizer.step()
        clamp_binary_mlp_latents(model)

        loss_sum += loss.item() * data.size(0)
        correct += (output.argmax(dim=1) == target).sum().item()
        count += data.size(0)
    return loss_sum, correct, count


@torch.no_grad()
def recalibrate_batch_norm(model, loader, device, max_batches):
    if max_batches == 0:
        return 0, 0, 0
    batch_norms = [
        module
        for module in model.modules()
        if getattr(module, "track_running_stats", False)
        and callable(getattr(module, "reset_running_stats", None))
    ]
    if not batch_norms:
        return 0, 0, 0

    training_states = {
        module: module.training for module in model.modules()
    }
    momenta = {
        module: module.momentum
        for module in batch_norms
        if hasattr(module, "momentum")
    }
    for module in batch_norms:
        module.reset_running_stats()
        if hasattr(module, "momentum"):
            module.momentum = None
    model.train()

    batches = 0
    samples = 0
    try:
        for data, _ in loader:
            if max_batches > 0 and batches >= max_batches:
                break
            data = data.to(device)
            model(data)
            batches += 1
            samples += data.size(0)
    finally:
        for module, momentum in momenta.items():
            module.momentum = momentum
        for module, training in training_states.items():
            module.training = training
    return len(batch_norms), batches, samples


def update_learning_rates(optimizer, epoch, args):
    base_lr = get_lr_for_epoch(epoch, args)
    for group in optimizer.param_groups:
        name = group.get("group_name")
        if name == "operation":
            if (
                args.flow_phase == 3.2
                and epoch < args.phase3p2_freeze_operation_epochs
            ):
                group["lr"] = 0.0
            else:
                group["lr"] = (
                    args.operation_lr
                    if args.operation_schedule == "constant"
                    else base_lr * (args.operation_lr / args.lr)
                )
        elif name == "lut":
            group["lr"] = (
                args.lut_lr
                if args.lut_schedule == "constant"
                else base_lr * (args.lut_lr / args.lr)
            )
        else:
            group["lr"] = base_lr
    return {group["group_name"]: group["lr"] for group in optimizer.param_groups}


def phase4_lut_state(epoch, args):
    if args.lut_tau_min <= 0.0 or args.lut_tau_max < args.lut_tau_min:
        raise ValueError("Invalid LUT tau range")
    if min(
        args.lut_soft_warmup_epochs,
        args.lut_anneal_epochs,
        args.lut_hard_transition_epochs,
    ) < 0:
        raise ValueError("LUT schedule epoch counts must be non-negative")

    anneal_start = args.lut_soft_warmup_epochs
    hard_start = anneal_start + args.lut_anneal_epochs
    hard_end = hard_start + args.lut_hard_transition_epochs
    if epoch < anneal_start:
        tau = args.lut_tau_min
        hard_ratio = 0.0
    elif epoch < hard_start and args.lut_anneal_epochs > 0:
        progress = (epoch - anneal_start) / float(max(args.lut_anneal_epochs - 1, 1))
        progress = min(max(progress, 0.0), 1.0)
        progress = 0.5 - 0.5 * math.cos(math.pi * progress)
        tau = args.lut_tau_min + (args.lut_tau_max - args.lut_tau_min) * progress
        hard_ratio = 0.0
    elif epoch < hard_end and args.lut_hard_transition_epochs > 0:
        tau = args.lut_tau_max
        hard_ratio = (epoch - hard_start + 1) / float(args.lut_hard_transition_epochs)
    else:
        tau = args.lut_tau_max
        hard_ratio = 1.0
    return float(tau), float(min(max(hard_ratio, 0.0), 1.0))


def _metric_loaders(val_loader, test_loader, no_validation):
    return (test_loader, "test") if no_validation else (val_loader, "val")


def evaluate_phase(model, val_loader, test_loader, device, phase):
    val_loss, val_acc = (
        evaluate(model, val_loader, device)
        if val_loader is not None
        else (0.0, 0.0)
    )
    test_loss, test_acc = evaluate(model, test_loader, device)
    result = {
        "val_loss": float(val_loss),
        "val_acc": float(val_acc),
        "test_loss": float(test_loss),
        "test_acc": float(test_acc),
    }
    if phase != 4.2:
        return result

    modules = list(iter_mlp_operation_layers(model))
    saved_states = [(module.tau.clone(), module.hard.clone()) for module in modules]
    try:
        for module in modules:
            module.hard.fill_(1.0)
        hard_val_loss, hard_val_acc = (
            evaluate(model, val_loader, device)
            if val_loader is not None
            else (0.0, 0.0)
        )
        hard_test_loss, hard_test_acc = evaluate(model, test_loader, device)
    finally:
        for module, (tau, hard) in zip(modules, saved_states):
            module.tau.copy_(tau)
            module.hard.copy_(hard)
    result.update(
        {
            "soft_val_loss": result["val_loss"],
            "soft_val_acc": result["val_acc"],
            "soft_test_loss": result["test_loss"],
            "soft_test_acc": result["test_acc"],
            "val_loss": float(hard_val_loss),
            "val_acc": float(hard_val_acc),
            "test_loss": float(hard_test_loss),
            "test_acc": float(hard_test_acc),
        }
    )
    return result


def checkpoint_state(model, optimizer, args, epoch, metrics, diagnostics):
    modules = list(iter_mlp_operation_layers(model))
    saved_hard = [module.hard.clone() for module in modules]
    try:
        if args.flow_phase == 4.2:
            for module in modules:
                module.hard.fill_(1.0)
        state = {
            "phase": args.flow_phase,
            "flow_phase": args.flow_phase,
            "flow_name": FLOW_PHASE_NAMES[args.flow_phase],
            "model": {
                key: value.detach().clone()
                for key, value in model.state_dict().items()
            },
            "optimizer": optimizer.state_dict(),
            "epoch": epoch,
            "args": vars(args),
            "metrics": metrics,
            "operation_diagnostics": diagnostics,
            "hard_lut_tables": export_hard_lut_tables(model),
            "hard_projection_saved": True,
        }
    finally:
        for module, hard in zip(modules, saved_hard):
            module.hard.copy_(hard)
    return state


def update_metrics_file(workdir, phase, metrics, diagnostics, checkpoint_path):
    path = Path(workdir) / "mlp_flow_metrics.json"
    if path.is_file():
        with path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
    else:
        payload = {}
    payload["phase{}".format(phase_tag(phase))] = {
        **metrics,
        "checkpoint": str(checkpoint_path),
        "operation_diagnostics": diagnostics,
    }
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
        "MLP flow Phase %.1f (%s), allocation=%s, layer_sets=%d, sets_per_channel=%d, binary_hidden=%d, binary_logit_init=%.3f",
        args.flow_phase,
        FLOW_PHASE_NAMES[args.flow_phase],
        args.operation_allocation,
        args.operation_sets,
        args.operation_sets_per_channel,
        args.binary_mlp_hidden,
        args.binary_mlp_logit_init,
    )
    if args.flow_phase == 3.2:
        train_logger.info(
            "Phase3.2 recovery: score_surrogate=%s, freeze_operation_epochs=%d, bn_calibration_batches=%d, operation_schedule=%s",
            args.phase3p2_score_surrogate,
            args.phase3p2_freeze_operation_epochs,
            args.phase3p2_bn_calibration_batches,
            args.operation_schedule,
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
    initialize_from_predecessor(model, args)
    device = torch.device(
        "cpu" if args.cpu or not torch.cuda.is_available() else "cuda"
    )
    model = model.to(device)
    if args.phase3p2_bn_calibration_batches != 0:
        bn_count, calibration_batches, calibration_samples = (
            recalibrate_batch_norm(
                model,
                train_loader,
                device,
                args.phase3p2_bn_calibration_batches,
            )
        )
        train_logger.info(
            "Recalibrated %d BN modules with hard Phase3.2 forward over %d batches (%d samples)",
            bn_count,
            calibration_batches,
            calibration_samples,
        )
    optimizer = build_optimizer(model, args)
    trainable_parameters = sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )
    total_operation_sets = sum(
        module.operation_sets for module in iter_mlp_operation_layers(model)
    )
    train_logger.info(
        "Device=%s, trainable_parameters=%d, total_operation_sets=%d",
        device,
        trainable_parameters,
        total_operation_sets,
    )

    metric_loader, metric_name = _metric_loaders(
        val_loader,
        test_loader,
        args.no_validation,
    )
    del metric_loader
    if args.flow_phase == 4.2:
        tau, hard_ratio = phase4_lut_state(0, args)
        set_phase4_lut_state(model, tau=tau, hard_ratio=hard_ratio)
    initial_metrics = evaluate_phase(
        model,
        val_loader,
        test_loader,
        device,
        args.flow_phase,
    )
    initial_diagnostics = operation_diagnostics(model)
    train_logger.info(
        "Initial %s_acc=%.4f test_acc=%.4f hard_changed=%.4f d_sensitive=%.4f d_score=%.6f zero_score=%.4f",
        metric_name,
        initial_metrics["test_acc" if args.no_validation else "val_acc"],
        initial_metrics["test_acc"],
        initial_diagnostics["hard_changed_ratio"],
        initial_diagnostics["d_sensitive_ratio"],
        initial_diagnostics["d_coefficient_abs_mean"],
        initial_diagnostics["zero_score_ratio"],
    )
    best_acc = initial_metrics[
        "test_acc" if args.no_validation else "val_acc"
    ]
    best_loss = initial_metrics[
        "test_loss" if args.no_validation else "val_loss"
    ]
    best_metrics = {
        "flow_phase": args.flow_phase,
        "flow_name": FLOW_PHASE_NAMES[args.flow_phase],
        "epoch": 0,
        "best_acc": best_acc,
        "best_loss": best_loss,
        "train_loss": None,
        "train_acc": None,
        **initial_metrics,
    }
    filename = "Bestmodel_phase{}.pt".format(phase_tag(args.flow_phase))
    initial_state = checkpoint_state(
        model,
        optimizer,
        args,
        0,
        best_metrics,
        initial_diagnostics,
    )
    best_checkpoint = save_checkpoint(
        initial_state,
        str(workdir),
        filename,
    )
    last_filename = "Lastmodel_phase{}.pt".format(
        phase_tag(args.flow_phase)
    )
    last_checkpoint = save_checkpoint(
        initial_state,
        str(workdir),
        last_filename,
    )
    update_metrics_file(
        workdir,
        args.flow_phase,
        best_metrics,
        initial_diagnostics,
        best_checkpoint,
    )

    for epoch in range(args.num_epochs):
        epoch_start = time.time()
        group_lrs = update_learning_rates(optimizer, epoch, args)
        tau = None
        hard_ratio = None
        if args.flow_phase == 4.2:
            tau, hard_ratio = phase4_lut_state(epoch, args)
            set_phase4_lut_state(model, tau=tau, hard_ratio=hard_ratio)

        train_loss_sum, train_correct, train_count = train_flow_epoch(
            model,
            train_loader,
            optimizer,
            device,
            args.clipnorm,
            args.clipval,
        )
        train_loss = train_loss_sum / float(train_count)
        train_acc = train_correct / float(train_count)
        metrics = evaluate_phase(
            model,
            val_loader,
            test_loader,
            device,
            args.flow_phase,
        )
        diagnostics = operation_diagnostics(model)
        selection_acc = metrics["test_acc" if args.no_validation else "val_acc"]
        selection_loss = metrics["test_loss" if args.no_validation else "val_loss"]
        improved = selection_acc > best_acc or (
            selection_acc == best_acc and selection_loss < best_loss
        )
        if improved:
            best_acc = selection_acc
            best_loss = selection_loss
            best_metrics = {
                "flow_phase": args.flow_phase,
                "flow_name": FLOW_PHASE_NAMES[args.flow_phase],
                "epoch": epoch + 1,
                "best_acc": best_acc,
                "best_loss": best_loss,
                "train_loss": train_loss,
                "train_acc": train_acc,
                **metrics,
            }
            filename = "Bestmodel_phase{}.pt".format(phase_tag(args.flow_phase))
            state = checkpoint_state(
                model,
                optimizer,
                args,
                epoch + 1,
                best_metrics,
                diagnostics,
            )
            best_checkpoint = save_checkpoint(state, str(workdir), filename)
            update_metrics_file(
                workdir,
                args.flow_phase,
                best_metrics,
                diagnostics,
                best_checkpoint,
            )

        endpoint_metrics = {
            "flow_phase": args.flow_phase,
            "flow_name": FLOW_PHASE_NAMES[args.flow_phase],
            "epoch": epoch + 1,
            "best_acc_so_far": best_acc,
            "best_loss_so_far": best_loss,
            "train_loss": train_loss,
            "train_acc": train_acc,
            **metrics,
        }
        endpoint_state = checkpoint_state(
            model,
            optimizer,
            args,
            epoch + 1,
            endpoint_metrics,
            diagnostics,
        )
        last_checkpoint = save_checkpoint(
            endpoint_state,
            str(workdir),
            last_filename,
        )

        schedule_text = ""
        if args.flow_phase == 4.2:
            schedule_text = " tau={:.4f} hard_ratio={:.4f}".format(tau, hard_ratio)
        train_logger.info(
            "Epoch %d train_loss=%.6f train_acc=%.4f val_loss=%.6f val_acc=%.4f test_loss=%.6f test_acc=%.4f lrs=%s%s hard_changed=%.4f d_sensitive=%.4f d_score=%.6f zero_score=%.4f best_%s=%.4f time=%.2fs",
            epoch + 1,
            train_loss,
            train_acc,
            metrics["val_loss"],
            metrics["val_acc"],
            metrics["test_loss"],
            metrics["test_acc"],
            group_lrs,
            schedule_text,
            diagnostics["hard_changed_ratio"],
            diagnostics["d_sensitive_ratio"],
            diagnostics["d_coefficient_abs_mean"],
            diagnostics["zero_score_ratio"],
            metric_name,
            best_acc,
            time.time() - epoch_start,
        )

    if best_checkpoint is None or best_metrics is None:
        raise RuntimeError("Training completed without a checkpoint")
    train_logger.info(
        "Completed Phase %.1f: best_%s_acc=%.4f at epoch=%d, best_checkpoint=%s, endpoint_checkpoint=%s",
        args.flow_phase,
        metric_name,
        best_metrics["best_acc"],
        best_metrics["epoch"],
        best_checkpoint,
        last_checkpoint,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Train the Phase1 -> fixed analytic dominance comparator flow."""

import argparse
import json
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from complexPyTorch.complexBinaryResNet import BinaryComplexResNet
from complexPyTorch.fixedAnalyticDominanceFlow import (
    FIXED_ANALYTIC_BIT_NAMES,
    FIXED_ANALYTIC_PHASE,
    FixedAnalyticBitGradientTracker,
    FixedAnalyticBitOccupancyTracker,
    FixedAnalyticDominanceComplexResNet,
    export_fixed_dominance_lut5,
    fixed_analytic_diagnostics,
    fixed_analytic_hardware_spec,
    iter_fixed_analytic_convs,
)
from training import (
    build_datasets,
    collect_batchnorm_params,
    evaluate,
    get_lr_for_epoch,
    load_phase_checkpoint,
    save_checkpoint,
    set_seed,
    setup_logging,
)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description=(
            "Phase2.6: Phase1-anchored fixed analytic dominance training "
            "with hard local comparators and comparator-aware gradients"
        )
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

    parser.add_argument("--dominance-beta", default=2.0, type=float)
    parser.add_argument("--dominance-phase-normalized", action="store_true")
    parser.add_argument("--comparator-beta", default=1.0, type=float)
    parser.add_argument("--comparator-scale", default=1.0, type=float)

    parser.add_argument("--distill-weight", default=0.5, type=float)
    parser.add_argument("--distill-temperature", default=2.0, type=float)
    parser.add_argument(
        "--bn-calibration-batches",
        default=-1,
        type=int,
        help="0 disables calibration; -1 uses the complete training loader",
    )
    parser.add_argument("--occupancy-interval", default=10, type=int)

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
    if args.lr <= 0.0:
        raise ValueError("lr must be positive")
    if args.dominance_beta <= 0.0:
        raise ValueError("dominance_beta must be positive")
    if args.comparator_beta <= 0.0 or args.comparator_scale <= 0.0:
        raise ValueError("comparator beta and scale must be positive")
    if args.distill_weight < 0.0 or args.distill_temperature <= 0.0:
        raise ValueError("distillation settings are invalid")
    if args.bn_calibration_batches < -1:
        raise ValueError("bn_calibration_batches must be -1 or non-negative")
    if args.occupancy_interval < 1:
        raise ValueError("occupancy_interval must be positive")


def build_student(args, num_classes):
    return FixedAnalyticDominanceComplexResNet(
        num_blocks=args.num_blocks,
        start_filters=args.start_filter,
        num_classes=num_classes,
        spectral_pool_scheme=args.spectral_pool_scheme,
        spectral_pool_gamma=args.spectral_pool_gamma,
        is_sar_input=False,
        dominance_beta=args.dominance_beta,
        dominance_phase_normalized=args.dominance_phase_normalized,
        comparator_beta=args.comparator_beta,
        comparator_scale=args.comparator_scale,
    )


def build_teacher(args, num_classes):
    return BinaryComplexResNet(
        num_blocks=args.num_blocks,
        start_filters=args.start_filter,
        num_classes=num_classes,
        spectral_pool_scheme=args.spectral_pool_scheme,
        spectral_pool_gamma=args.spectral_pool_gamma,
        is_sar_input=False,
        is_binary=False,
        phase=1,
    )


def build_optimizer(model, args):
    forbidden = [
        name
        for name, _ in model.named_parameters()
        if "lut" in name.lower()
    ]
    if forbidden:
        raise RuntimeError(
            "fixed analytic flow unexpectedly contains LUT parameters: {}".format(
                forbidden
            )
        )

    bn_parameters = collect_batchnorm_params(model)
    bn_ids = {id(parameter) for parameter in bn_parameters}
    decay_parameters = []
    no_decay_parameters = []
    for name, parameter in model.named_parameters():
        if id(parameter) in bn_ids:
            continue
        if (
            parameter.ndim <= 1
            or name.endswith(("weight_r", "weight_i"))
            or ".conv_r.weight" in name
            or ".conv_i.weight" in name
        ):
            no_decay_parameters.append(parameter)
        else:
            decay_parameters.append(parameter)

    groups = []
    for name, parameters, weight_decay in (
        ("decay", decay_parameters, args.weight_decay),
        ("no_decay", no_decay_parameters, 0.0),
        ("bn", bn_parameters, 0.0),
    ):
        if parameters:
            groups.append(
                {
                    "params": parameters,
                    "lr": args.lr,
                    "weight_decay": weight_decay,
                    "group_name": name,
                }
            )
    if args.optimizer == "sgd":
        return torch.optim.SGD(groups, momentum=args.momentum)
    return torch.optim.AdamW(groups)


def update_learning_rates(optimizer, epoch, args):
    learning_rate = get_lr_for_epoch(epoch, args)
    for group in optimizer.param_groups:
        group["lr"] = learning_rate
    return {
        group["group_name"]: float(group["lr"])
        for group in optimizer.param_groups
    }


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
            model(data.to(device))
            batches += 1
            samples += data.size(0)
    finally:
        for module, momentum in momenta.items():
            module.momentum = momentum
        for module, training in training_states.items():
            module.training = training
    return len(batch_norms), batches, samples


def distillation_loss(student_logits, teacher_logits, temperature):
    temperature = float(temperature)
    return F.kl_div(
        F.log_softmax(student_logits / temperature, dim=1),
        F.softmax(teacher_logits / temperature, dim=1),
        reduction="batchmean",
    ) * (temperature * temperature)


def train_epoch(model, teacher, loader, optimizer, device, args):
    model.train()
    teacher.eval()
    totals = {
        "loss": 0.0,
        "ce": 0.0,
        "distill": 0.0,
        "correct": 0,
        "count": 0,
    }
    for data, target in loader:
        data = data.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        output = model(data)
        with torch.no_grad():
            teacher_output = teacher(data)
        ce = F.cross_entropy(output, target)
        distill = distillation_loss(
            output, teacher_output, args.distill_temperature
        )
        loss = ce + args.distill_weight * distill
        if not torch.isfinite(loss):
            raise RuntimeError("fixed analytic training produced non-finite loss")
        loss.backward()
        if args.clipnorm > 0.0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.clipnorm)
        if args.clipval > 0.0:
            torch.nn.utils.clip_grad_value_(model.parameters(), args.clipval)
        optimizer.step()

        batch = data.size(0)
        totals["loss"] += float(loss.item()) * batch
        totals["ce"] += float(ce.item()) * batch
        totals["distill"] += float(distill.item()) * batch
        totals["correct"] += int((output.argmax(dim=1) == target).sum().item())
        totals["count"] += batch

    count = float(totals["count"])
    return {
        "train_loss": totals["loss"] / count,
        "train_ce": totals["ce"] / count,
        "train_distill": totals["distill"] / count,
        "train_acc": totals["correct"] / count,
    }


def evaluate_all(model, val_loader, test_loader, device, track_occupancy=False):
    val_loss, val_acc = (
        evaluate(model, val_loader, device)
        if val_loader is not None
        else (0.0, 0.0)
    )
    tracker = (
        FixedAnalyticBitOccupancyTracker(model) if track_occupancy else None
    )
    test_loss, test_acc = evaluate(model, test_loader, device)
    occupancy = tracker.finish() if tracker is not None else None
    return {
        "val_loss": float(val_loss),
        "val_acc": float(val_acc),
        "test_loss": float(test_loss),
        "test_acc": float(test_acc),
    }, occupancy


def checkpoint_state(
    model,
    optimizer,
    args,
    epoch,
    metrics,
    diagnostics,
    occupancy,
    bit_gradients,
):
    return {
        "phase": FIXED_ANALYTIC_PHASE,
        "flow_phase": FIXED_ANALYTIC_PHASE,
        "flow_name": "fixed analytic dominance comparator from Phase1",
        "model": {
            key: value.detach().cpu().clone()
            for key, value in model.state_dict().items()
        },
        "optimizer": optimizer.state_dict(),
        "epoch": int(epoch),
        "args": vars(args),
        "metrics": metrics,
        "analytic_diagnostics": diagnostics,
        "bit_occupancy": occupancy,
        "bit_gradients": bit_gradients,
        "fixed_lut5_tables": export_fixed_dominance_lut5(),
        "hardware_spec": fixed_analytic_hardware_spec(),
        "hardware_deployable": bool(diagnostics["hardware_deployable"]),
        "training_uses_lut": False,
        "truth_table_trainable": False,
        "source_checkpoint": str(args.checkpoint),
    }


def update_metrics_files(
    workdir,
    args,
    best_checkpoint,
    best_metrics,
    last_metrics,
    diagnostics,
    occupancy,
    bit_gradients,
):
    payload = {
        "phase": FIXED_ANALYTIC_PHASE,
        "flow_name": "fixed analytic dominance comparator from Phase1",
        "best_checkpoint": str(best_checkpoint),
        "best_metrics": best_metrics,
        "last_metrics": last_metrics,
        "analytic_diagnostics": diagnostics,
        "bit_occupancy": occupancy,
        "bit_gradients": bit_gradients,
        "hardware_spec": fixed_analytic_hardware_spec(),
        "training_uses_lut": False,
        "truth_table_trainable": False,
        "args": vars(args),
    }
    with (Path(workdir) / "fixed_analytic_metrics.json").open(
        "w", encoding="utf-8"
    ) as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)

    phase_metrics = {
        "phase2p6": {
            **best_metrics,
            "best_checkpoint": str(best_checkpoint),
            "hardware_deployable": bool(
                diagnostics["hardware_deployable"]
            ),
            "training_uses_lut": False,
        }
    }
    with (Path(workdir) / "phase_metrics.json").open(
        "w", encoding="utf-8"
    ) as handle:
        json.dump(phase_metrics, handle, indent=2, sort_keys=True)


def log_diagnostics(logger, epoch, diagnostics, gradients, occupancy):
    logger.info(
        "[Fixed Analytic Dominance] Epoch %d: analytic_convs=%d, LUT_modules=%d, training_uses_LUT=%s, d_slice_diff=%d/%d (%.6f%%), comparator_beta=%.4f, comparator_scale=%.4f, deployable=%s",
        epoch,
        diagnostics["analytic_conv_modules"],
        diagnostics["lut_modules"],
        diagnostics["training_uses_lut"],
        diagnostics["extra_bit_slice_hard_diff"],
        diagnostics["extra_bit_slice_pairs"],
        100.0 * diagnostics["extra_bit_slice_hard_diff_ratio"],
        diagnostics["comparator_beta"],
        diagnostics["comparator_scale"],
        diagnostics["hardware_deployable"],
    )
    if gradients is not None:
        gradient_text = ", ".join(
            "{}:nz={:.4f},mean={:.3e},max={:.3e}".format(
                name,
                values["nonzero_ratio"],
                values["mean_abs"],
                values["max_abs"],
            )
            for name, values in gradients["bits"].items()
        )
        logger.info(
            "[Fixed Analytic Bit Gradients] Epoch %d: tensors=%d, %s",
            epoch,
            gradients["captured_tensors"],
            gradient_text,
        )
    if occupancy is not None:
        logger.info(
            "[Fixed Analytic Occupancy] Epoch %d: counts=%s, active_codes=%d, dominance_one_ratio=%.6f, normalized_entropy=%.6f",
            epoch,
            occupancy["counts"],
            occupancy["active_codes"],
            occupancy["extra_bit_one_ratio"],
            occupancy["normalized_entropy"],
        )


def main(argv=None):
    args = parse_args(argv)
    validate_args(args)
    set_seed(args.seed)
    workdir = Path(args.workdir)
    workdir.mkdir(parents=True, exist_ok=True)
    _, train_logger = setup_logging(str(workdir), args.loglevel, True)
    train_logger.info("INVOCATION: %s", " ".join([sys.executable, *sys.argv]))
    train_logger.info(
        "Fixed analytic dominance flow: Phase1 checkpoint=%s, hard_forward_from_epoch0=True, training_uses_LUT=False, dominance_beta=%.4f, comparator_beta=%.4f, comparator_scale=%.4f, base_lr=%.6f (%s), distill_weight=%.3f",
        args.checkpoint,
        args.dominance_beta,
        args.comparator_beta,
        args.comparator_scale,
        args.lr,
        args.schedule,
        args.distill_weight,
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

    model = build_student(args, num_classes)
    teacher = build_teacher(args, num_classes)
    load_phase_checkpoint(model, args.checkpoint, expected_phase=1)
    load_phase_checkpoint(teacher, args.checkpoint, expected_phase=1)

    device = torch.device(
        "cpu" if args.cpu or not torch.cuda.is_available() else "cuda"
    )
    model = model.to(device)
    teacher = teacher.to(device)
    teacher.eval()
    for parameter in teacher.parameters():
        parameter.requires_grad_(False)

    diagnostics = fixed_analytic_diagnostics(model)
    if diagnostics["lut_modules"] != 0 or not diagnostics["hardware_deployable"]:
        raise RuntimeError(
            "fixed analytic model failed the no-LUT/deployability audit"
        )
    optimizer = build_optimizer(model, args)
    bn_count, calibration_batches, calibration_samples = recalibrate_batch_norm(
        model,
        train_loader,
        device,
        args.bn_calibration_batches,
    )
    train_logger.info(
        "Recalibrated %d BN modules under fixed analytic hard forward with %d batches (%d samples)",
        bn_count,
        calibration_batches,
        calibration_samples,
    )
    train_logger.info(
        "Device=%s, trainable_parameters=%d, analytic_conv_modules=%d, LUT_modules=%d, bit_names=%s",
        device,
        sum(parameter.numel() for parameter in model.parameters()),
        sum(1 for _ in iter_fixed_analytic_convs(model)),
        diagnostics["lut_modules"],
        FIXED_ANALYTIC_BIT_NAMES,
    )

    initial_metrics, occupancy = evaluate_all(
        model,
        val_loader,
        test_loader,
        device,
        track_occupancy=True,
    )
    initial_epoch_metrics = {
        "phase": FIXED_ANALYTIC_PHASE,
        "epoch": 0,
        "train_loss": None,
        "train_acc": None,
        **initial_metrics,
    }
    metric_name = "test" if args.no_validation else "val"
    best_acc = initial_metrics["{}_acc".format(metric_name)]
    best_loss = initial_metrics["{}_loss".format(metric_name)]
    best_metrics = dict(initial_epoch_metrics)
    bit_gradients = None
    best_filename = "Bestmodel_fixed_analytic_dominance.pt"
    last_filename = "Lastmodel_fixed_analytic_dominance.pt"
    best_checkpoint = save_checkpoint(
        checkpoint_state(
            model,
            optimizer,
            args,
            0,
            best_metrics,
            diagnostics,
            occupancy,
            bit_gradients,
        ),
        str(workdir),
        best_filename,
    )
    last_checkpoint = save_checkpoint(
        checkpoint_state(
            model,
            optimizer,
            args,
            0,
            initial_epoch_metrics,
            diagnostics,
            occupancy,
            bit_gradients,
        ),
        str(workdir),
        last_filename,
    )
    update_metrics_files(
        workdir,
        args,
        best_checkpoint,
        best_metrics,
        initial_epoch_metrics,
        diagnostics,
        occupancy,
        bit_gradients,
    )
    train_logger.info(
        "Initial fixed analytic anchor after BN calibration: val_loss=%.6f val_acc=%.4f test_loss=%.6f test_acc=%.4f",
        initial_metrics["val_loss"],
        initial_metrics["val_acc"],
        initial_metrics["test_loss"],
        initial_metrics["test_acc"],
    )
    log_diagnostics(train_logger, 0, diagnostics, bit_gradients, occupancy)

    gradient_tracker = FixedAnalyticBitGradientTracker(model)
    last_occupancy = occupancy
    for epoch in range(args.num_epochs):
        epoch_start = time.time()
        group_lrs = update_learning_rates(optimizer, epoch, args)
        gradient_tracker.reset()
        train_metrics = train_epoch(
            model,
            teacher,
            train_loader,
            optimizer,
            device,
            args,
        )
        bit_gradients = gradient_tracker.finish()
        track_occupancy = (
            epoch == args.num_epochs - 1
            or (epoch + 1) % args.occupancy_interval == 0
        )
        eval_metrics, current_occupancy = evaluate_all(
            model,
            val_loader,
            test_loader,
            device,
            track_occupancy=track_occupancy,
        )
        if current_occupancy is not None:
            last_occupancy = current_occupancy
        diagnostics = fixed_analytic_diagnostics(model)
        epoch_metrics = {
            "phase": FIXED_ANALYTIC_PHASE,
            "epoch": epoch + 1,
            **train_metrics,
            **eval_metrics,
            "learning_rates": group_lrs,
            "training_uses_lut": False,
        }

        selection_acc = eval_metrics["{}_acc".format(metric_name)]
        selection_loss = eval_metrics["{}_loss".format(metric_name)]
        improved = selection_acc > best_acc or (
            selection_acc == best_acc and selection_loss < best_loss
        )
        if improved:
            best_acc = selection_acc
            best_loss = selection_loss
            best_metrics = dict(epoch_metrics)
            best_checkpoint = save_checkpoint(
                checkpoint_state(
                    model,
                    optimizer,
                    args,
                    epoch + 1,
                    best_metrics,
                    diagnostics,
                    last_occupancy,
                    bit_gradients,
                ),
                str(workdir),
                best_filename,
            )
            train_logger.info(
                "Saved best fixed analytic model at epoch %d: %s_acc=%.4f",
                epoch + 1,
                metric_name,
                best_acc,
            )

        last_checkpoint = save_checkpoint(
            checkpoint_state(
                model,
                optimizer,
                args,
                epoch + 1,
                epoch_metrics,
                diagnostics,
                last_occupancy,
                bit_gradients,
            ),
            str(workdir),
            last_filename,
        )
        update_metrics_files(
            workdir,
            args,
            best_checkpoint,
            best_metrics,
            epoch_metrics,
            diagnostics,
            last_occupancy,
            bit_gradients,
        )
        train_logger.info(
            "Epoch %5d train_loss: %.6f, train_acc: %.4f, val_loss: %.6f, val_acc: %.4f, test_loss: %.6f, test_acc: %.4f",
            epoch + 1,
            train_metrics["train_loss"],
            train_metrics["train_acc"],
            eval_metrics["val_loss"],
            eval_metrics["val_acc"],
            eval_metrics["test_loss"],
            eval_metrics["test_acc"],
        )
        train_logger.info(
            "Fixed analytic epoch %d details: ce=%.6f distill=%.6f lrs=%s best_%s=%.4f time=%.2fs",
            epoch + 1,
            train_metrics["train_ce"],
            train_metrics["train_distill"],
            group_lrs,
            metric_name,
            best_acc,
            time.time() - epoch_start,
        )
        log_diagnostics(
            train_logger,
            epoch + 1,
            diagnostics,
            bit_gradients,
            current_occupancy,
        )

    gradient_tracker.close()
    train_logger.info(
        "Completed fixed analytic dominance: best_%s_acc=%.4f at epoch=%d, best=%s, last=%s",
        metric_name,
        best_acc,
        best_metrics["epoch"],
        best_checkpoint,
        last_checkpoint,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

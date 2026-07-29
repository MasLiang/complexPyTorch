#!/usr/bin/env python3
"""Train the isolated Phase1 -> hard Direct LUT5 flow."""

import argparse
import json
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from complexPyTorch.complexBinaryResNet import BinaryComplexResNet
from complexPyTorch.directLut5Flow import (
    DIRECT_LUT5_ENCODERS,
    DIRECT_LUT5_PHASE,
    DirectBitGradientTracker,
    DirectBitOccupancyTracker,
    DirectLUT5ComplexResNet,
    capture_hard_lut_tables,
    direct_lut5_bit_names,
    direct_lut5_diagnostics,
    direct_lut5_hardware_spec,
    direct_lut5_margin_loss,
    iter_direct_lut5_convs,
    set_direct_lut5_trainable,
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
            "Phase2.5: Phase1-anchored hard 5-to-2 LUT training with "
            "end-to-end activation-bit gradients"
        )
    )
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument(
        "--encoder-mode",
        required=True,
        choices=DIRECT_LUT5_ENCODERS,
    )
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

    parser.add_argument("--c8-beta", default=2.0, type=float)
    parser.add_argument("--lut-sets", default=1, type=int)
    parser.add_argument(
        "--lut-allocation",
        default="layer",
        choices=["layer", "channel"],
    )
    parser.add_argument("--lut-sets-per-channel", default=1, type=int)
    parser.add_argument("--initial-logit-margin", default=0.25, type=float)
    parser.add_argument("--lut-freeze-epochs", default=60, type=int)
    parser.add_argument("--lut-lr", default=0.002, type=float)
    parser.add_argument(
        "--lut-schedule",
        default="follow-base",
        choices=["constant", "follow-base"],
    )
    parser.add_argument("--lut-margin", default=0.15, type=float)
    parser.add_argument("--lut-margin-weight", default=0.0001, type=float)

    parser.add_argument("--distill-weight", default=0.5, type=float)
    parser.add_argument("--distill-temperature", default=2.0, type=float)
    parser.add_argument(
        "--bn-calibration-batches",
        default=100,
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
    if not 0 <= args.lut_freeze_epochs < args.num_epochs:
        raise ValueError("lut_freeze_epochs must be in [0, num_epochs)")
    if args.lr <= 0.0 or args.lut_lr <= 0.0:
        raise ValueError("lr and lut_lr must be positive")
    if args.c8_beta <= 0.0:
        raise ValueError("c8_beta must be positive")
    if args.initial_logit_margin <= 0.0:
        raise ValueError("initial_logit_margin must be positive")
    if args.lut_margin < 0.0 or args.lut_margin_weight < 0.0:
        raise ValueError("LUT margin settings must be non-negative")
    if args.distill_weight < 0.0 or args.distill_temperature <= 0.0:
        raise ValueError("Distillation settings are invalid")
    if args.bn_calibration_batches < -1:
        raise ValueError("bn_calibration_batches must be -1 or non-negative")
    if args.occupancy_interval < 1:
        raise ValueError("occupancy_interval must be positive")
    if args.lut_sets < 1 or args.lut_sets_per_channel < 1:
        raise ValueError("LUT set counts must be positive")


def build_student(args, num_classes):
    return DirectLUT5ComplexResNet(
        num_blocks=args.num_blocks,
        start_filters=args.start_filter,
        num_classes=num_classes,
        spectral_pool_scheme=args.spectral_pool_scheme,
        spectral_pool_gamma=args.spectral_pool_gamma,
        is_sar_input=False,
        encoder_mode=args.encoder_mode,
        c8_beta=args.c8_beta,
        lut_sets=args.lut_sets,
        lut_allocation=args.lut_allocation,
        lut_sets_per_channel=args.lut_sets_per_channel,
        initial_logit_margin=args.initial_logit_margin,
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
    lut_parameters = []
    lut_ids = set()
    for name, parameter in model.named_parameters():
        if name.endswith((".lut_r", ".lut_i")):
            lut_parameters.append(parameter)
            lut_ids.add(id(parameter))

    bn_parameters = collect_batchnorm_params(model)
    bn_ids = {id(parameter) for parameter in bn_parameters}
    decay_parameters = []
    no_decay_parameters = []
    for name, parameter in model.named_parameters():
        if id(parameter) in lut_ids or id(parameter) in bn_ids:
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
    for name, parameters, learning_rate, weight_decay in (
        ("decay", decay_parameters, args.lr, args.weight_decay),
        ("no_decay", no_decay_parameters, args.lr, 0.0),
        ("bn", bn_parameters, args.lr, 0.0),
        ("lut", lut_parameters, args.lut_lr, 0.0),
    ):
        if parameters:
            groups.append(
                {
                    "params": parameters,
                    "lr": learning_rate,
                    "weight_decay": weight_decay,
                    "group_name": name,
                }
            )
    if not lut_parameters:
        raise RuntimeError("Direct LUT5 optimizer found no LUT parameters")
    if args.optimizer == "sgd":
        return torch.optim.SGD(groups, momentum=args.momentum)
    return torch.optim.AdamW(groups)


def update_learning_rates(optimizer, epoch, args, lut_trainable):
    base_lr = get_lr_for_epoch(epoch, args)
    lut_ratio = args.lut_lr / args.lr
    for group in optimizer.param_groups:
        if group["group_name"] == "lut":
            if not lut_trainable:
                group["lr"] = 0.0
            elif args.lut_schedule == "constant":
                group["lr"] = args.lut_lr
            else:
                group["lr"] = base_lr * lut_ratio
        else:
            group["lr"] = base_lr
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


def train_epoch(
    model,
    teacher,
    loader,
    optimizer,
    device,
    args,
    lut_trainable,
):
    model.train()
    teacher.eval()
    totals = {
        "loss": 0.0,
        "ce": 0.0,
        "distill": 0.0,
        "margin": 0.0,
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
            output,
            teacher_output,
            args.distill_temperature,
        )
        if lut_trainable and args.lut_margin_weight > 0.0:
            margin = direct_lut5_margin_loss(model, args.lut_margin)
        else:
            margin = output.new_zeros(())
        loss = (
            ce
            + args.distill_weight * distill
            + args.lut_margin_weight * margin
        )
        if not torch.isfinite(loss):
            raise RuntimeError("Direct LUT5 training produced non-finite loss")
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
        totals["margin"] += float(margin.item()) * batch
        totals["correct"] += int((output.argmax(dim=1) == target).sum().item())
        totals["count"] += batch

    count = float(totals["count"])
    return {
        "train_loss": totals["loss"] / count,
        "train_ce": totals["ce"] / count,
        "train_distill": totals["distill"] / count,
        "train_lut_margin": totals["margin"] / count,
        "train_acc": totals["correct"] / count,
    }


def evaluate_all(
    model,
    val_loader,
    test_loader,
    device,
    track_occupancy=False,
):
    val_loss, val_acc = (
        evaluate(model, val_loader, device)
        if val_loader is not None
        else (0.0, 0.0)
    )
    tracker = DirectBitOccupancyTracker(model) if track_occupancy else None
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
    lut_trainable,
):
    return {
        "phase": DIRECT_LUT5_PHASE,
        "flow_phase": DIRECT_LUT5_PHASE,
        "flow_name": "hard Direct LUT5 from Phase1",
        "encoder_mode": args.encoder_mode,
        "model": {
            key: value.detach().cpu().clone()
            for key, value in model.state_dict().items()
        },
        "optimizer": optimizer.state_dict(),
        "epoch": int(epoch),
        "args": vars(args),
        "metrics": metrics,
        "lut_diagnostics": diagnostics,
        "bit_occupancy": occupancy,
        "bit_gradients": bit_gradients,
        "lut_trainable": bool(lut_trainable),
        "hard_lut_tables": capture_hard_lut_tables(model),
        "hardware_spec": direct_lut5_hardware_spec(args.encoder_mode),
        "hardware_deployable": bool(diagnostics["fully_hard"]),
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
        "phase": DIRECT_LUT5_PHASE,
        "flow_name": "hard Direct LUT5 from Phase1",
        "encoder_mode": args.encoder_mode,
        "best_checkpoint": str(best_checkpoint),
        "best_metrics": best_metrics,
        "last_metrics": last_metrics,
        "lut_diagnostics": diagnostics,
        "bit_occupancy": occupancy,
        "bit_gradients": bit_gradients,
        "hardware_spec": direct_lut5_hardware_spec(args.encoder_mode),
    }
    path = Path(workdir) / "direct_lut5_metrics.json"
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)

    phase_metrics = {
        "phase2p5": {
            **best_metrics,
            "encoder_mode": args.encoder_mode,
            "best_checkpoint": str(best_checkpoint),
            "hardware_deployable": True,
        }
    }
    with (Path(workdir) / "phase_metrics.json").open(
        "w", encoding="utf-8"
    ) as handle:
        json.dump(phase_metrics, handle, indent=2, sort_keys=True)


def log_diagnostics(logger, epoch, args, diagnostics, gradients, occupancy):
    logger.info(
        "[Direct LUT5 %s] Epoch %d: sign_diff=%d/%d (%.6f%%), near_zero=%d/%d (%.6f%%), extra_bit_slice_hard_diff=%d/%d (%.6f%%), soft_slice_diff=%.6f, abs_logit=(%.5f,%.5f,%.5f), hard=%s",
        args.encoder_mode,
        epoch,
        diagnostics["sign_diff"],
        diagnostics["entries"],
        100.0 * diagnostics["sign_diff_ratio"],
        diagnostics["near_zero"],
        diagnostics["entries"],
        100.0 * diagnostics["near_zero_ratio"],
        diagnostics["extra_bit_slice_hard_diff"],
        diagnostics["extra_bit_slice_pairs"],
        100.0 * diagnostics["extra_bit_slice_hard_diff_ratio"],
        diagnostics["extra_bit_slice_soft_abs_diff_mean"],
        diagnostics["abs_logit_min"],
        diagnostics["abs_logit_mean"],
        diagnostics["abs_logit_max"],
        diagnostics["fully_hard"],
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
            "[Direct LUT5 %s Bit Gradients] Epoch %d: tensors=%d, %s",
            args.encoder_mode,
            epoch,
            gradients["captured_tensors"],
            gradient_text,
        )
    if occupancy is not None:
        logger.info(
            "[Direct LUT5 %s Occupancy] Epoch %d: counts=%s, active_codes=%d, extra_bit_one_ratio=%.6f, normalized_entropy=%.6f",
            args.encoder_mode,
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
        "Direct LUT5 flow: encoder=%s, Phase1 checkpoint=%s, hard_forward_from_epoch0=True, LUT_freeze=%d, base_lr=%.6f, lut_lr=%.6f (%s), distill_weight=%.3f",
        args.encoder_mode,
        args.checkpoint,
        args.lut_freeze_epochs,
        args.lr,
        args.lut_lr,
        args.lut_schedule,
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
    load_phase_checkpoint(
        model,
        args.checkpoint,
        expected_phase=1,
    )
    load_phase_checkpoint(
        teacher,
        args.checkpoint,
        expected_phase=1,
    )

    device = torch.device(
        "cpu" if args.cpu or not torch.cuda.is_available() else "cuda"
    )
    model = model.to(device)
    teacher = teacher.to(device)
    teacher.eval()
    for parameter in teacher.parameters():
        parameter.requires_grad_(False)

    optimizer = build_optimizer(model, args)
    set_direct_lut5_trainable(model, False)
    bn_count, calibration_batches, calibration_samples = recalibrate_batch_norm(
        model,
        train_loader,
        device,
        args.bn_calibration_batches,
    )
    train_logger.info(
        "Recalibrated %d BN modules under hard LUT5 forward with %d batches (%d samples)",
        bn_count,
        calibration_batches,
        calibration_samples,
    )
    train_logger.info(
        "Device=%s, trainable_without_frozen_LUT=%d, LUT_modules=%d, bit_names=%s",
        device,
        sum(
            parameter.numel()
            for parameter in model.parameters()
            if parameter.requires_grad
        ),
        sum(1 for _ in iter_direct_lut5_convs(model)),
        direct_lut5_bit_names(args.encoder_mode),
    )

    initial_metrics, occupancy = evaluate_all(
        model,
        val_loader,
        test_loader,
        device,
        track_occupancy=True,
    )
    diagnostics = direct_lut5_diagnostics(model)
    initial_epoch_metrics = {
        "phase": DIRECT_LUT5_PHASE,
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
    best_filename = "Bestmodel_direct_lut5_{}.pt".format(args.encoder_mode)
    last_filename = "Lastmodel_direct_lut5_{}.pt".format(args.encoder_mode)
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
            False,
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
            False,
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
        "Initial hard LUT5 anchor after BN calibration: val_loss=%.6f val_acc=%.4f test_loss=%.6f test_acc=%.4f",
        initial_metrics["val_loss"],
        initial_metrics["val_acc"],
        initial_metrics["test_loss"],
        initial_metrics["test_acc"],
    )
    log_diagnostics(
        train_logger,
        0,
        args,
        diagnostics,
        bit_gradients,
        occupancy,
    )

    gradient_tracker = DirectBitGradientTracker(model)
    bit_names = direct_lut5_bit_names(args.encoder_mode)
    last_occupancy = occupancy
    lut_state = False

    for epoch in range(args.num_epochs):
        epoch_start = time.time()
        lut_trainable = epoch >= args.lut_freeze_epochs
        if lut_trainable != lut_state:
            set_direct_lut5_trainable(model, lut_trainable)
            lut_state = lut_trainable
            train_logger.info(
                "Direct LUT5 tables became trainable at epoch %d; hard forward remains enabled",
                epoch + 1,
            )
        group_lrs = update_learning_rates(
            optimizer,
            epoch,
            args,
            lut_trainable,
        )
        gradient_tracker.reset()
        train_metrics = train_epoch(
            model,
            teacher,
            train_loader,
            optimizer,
            device,
            args,
            lut_trainable,
        )
        bit_gradients = gradient_tracker.finish(bit_names)
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
        diagnostics = direct_lut5_diagnostics(model)
        epoch_metrics = {
            "phase": DIRECT_LUT5_PHASE,
            "epoch": epoch + 1,
            **train_metrics,
            **eval_metrics,
            "lut_trainable": lut_trainable,
            "learning_rates": group_lrs,
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
                    lut_trainable,
                ),
                str(workdir),
                best_filename,
            )
            train_logger.info(
                "Saved best hard Direct LUT5 model at epoch %d: %s_acc=%.4f",
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
                lut_trainable,
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
            "Direct LUT5 epoch %d details: ce=%.6f distill=%.6f margin=%.6f lrs=%s lut_trainable=%s best_%s=%.4f time=%.2fs",
            epoch + 1,
            train_metrics["train_ce"],
            train_metrics["train_distill"],
            train_metrics["train_lut_margin"],
            group_lrs,
            lut_trainable,
            metric_name,
            best_acc,
            time.time() - epoch_start,
        )
        log_diagnostics(
            train_logger,
            epoch + 1,
            args,
            diagnostics,
            bit_gradients,
            current_occupancy,
        )

    gradient_tracker.close()
    train_logger.info(
        "Completed Direct LUT5 %s: best_%s_acc=%.4f at epoch=%d, best=%s, last=%s",
        args.encoder_mode,
        metric_name,
        best_acc,
        best_metrics["epoch"],
        best_checkpoint,
        last_checkpoint,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

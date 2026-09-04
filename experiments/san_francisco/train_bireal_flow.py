#!/usr/bin/env python3
"""Train one stage of the San Francisco FP-BiReal-LUT flow."""

import argparse
import json
import logging
import math
from pathlib import Path
import sys

import torch
from torch import nn

from complexPyTorch.complexLayers import PairLUT4ComplexConv2d
from datasets.san_francisco import build_san_francisco_datasets
from experiments.flevoland.train import run_epoch, set_seed
from experiments.san_francisco.checkpoints import (
    load_bireal_from_fp,
    load_lut_from_bireal,
    load_lut6_lut4_residual,
    load_lut6_residual_from_lut4,
)
from experiments.san_francisco.models import (
    STAGE_TO_PHASE,
    build_san_francisco_bireal_model,
)
from experiments.san_francisco.train import (
    NUM_CLASSES,
    NUM_INPUT_CHANNELS,
    build_loaders,
    class_distribution,
    save_json,
    save_prediction_map,
    smoke_test_step,
)


LOGGER = logging.getLogger("san_francisco.bireal_flow")


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--stage", choices=tuple(STAGE_TO_PHASE), required=True
    )
    parser.add_argument("--data-root", default="data/san_francisco")
    parser.add_argument("--stk-file", default="san_francisco900x1024.stk")
    parser.add_argument("--labels-file", default="SF-AIRSAR-label2d.png")
    parser.add_argument("--patch-size", type=int, default=11)
    parser.add_argument("--split-mode", choices=("spatial", "random"), default="spatial")
    parser.add_argument("--spatial-block-size", type=int, default=64)
    parser.add_argument("--train-fraction", type=float, default=0.1)
    parser.add_argument("--val-fraction", type=float, default=0.1)
    parser.add_argument("--start-filters", type=int, default=16)
    parser.add_argument("--num-blocks", type=int, default=3)
    parser.add_argument(
        "--post-bn-mode", choices=("covariance", "naive", "none"), default="covariance"
    )
    parser.add_argument(
        "--shortcut-mode", choices=("fp", "option_a"), default="fp"
    )
    parser.add_argument(
        "--pair-lut-parameterization",
        choices=("independent", "categorical"),
        default="categorical",
    )
    parser.add_argument("--checkpoint")
    parser.add_argument("--train-from-scratch", action="store_true")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--lut-lr", type=float, default=5e-3)
    parser.add_argument("--base-lut-lr", type=float, default=0.0)
    parser.add_argument("--residual-alpha-start", type=float, default=0.0)
    parser.add_argument("--residual-alpha-end", type=float, default=1.0)
    parser.add_argument("--residual-ramp-epochs", type=int, default=120)
    parser.add_argument("--base-alpha-start", type=float, default=0.0)
    parser.add_argument("--base-alpha-end", type=float, default=0.0)
    parser.add_argument("--base-ramp-epochs", type=int, default=1)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument(
        "--schedule", choices=("constant", "cosine", "bireal"), default="cosine"
    )
    parser.add_argument("--min-lr-factor", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--split-seed", type=int)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--workdir", required=True)
    parser.add_argument("--smoke-test", action="store_true")
    parser.add_argument("--save-prediction-map", action="store_true")
    return parser.parse_args(argv)


def learning_rate(epoch, args):
    if args.schedule == "constant":
        return args.lr
    if args.schedule == "cosine":
        progress = epoch / float(max(args.epochs - 1, 1))
        minimum = args.lr * args.min_lr_factor
        return minimum + 0.5 * (args.lr - minimum) * (
            1.0 + math.cos(math.pi * progress)
        )
    warmup = min(5, args.epochs)
    if epoch < warmup:
        return args.lr * (epoch + 1) / float(warmup)
    if epoch < int(args.epochs * 0.5):
        return args.lr
    if epoch < int(args.epochs * 0.75):
        return args.lr * 0.1
    if epoch < int(args.epochs * 0.875):
        return args.lr * 0.01
    return args.lr * 0.001


def build_optimizer(model, args):
    if args.lr <= 0:
        raise ValueError("--lr must be positive")
    lut_parameters = []
    base_corrections = []
    for module in model.modules():
        if not isinstance(module, PairLUT4ComplexConv2d):
            continue
        lut_parameters.append(module.weight)
        if module.parameterization == "categorical_residual":
            base_corrections.append(module.dominance_base_correction)
    special_ids = {
        id(parameter) for parameter in lut_parameters + base_corrections
    }
    base_parameters = [
        parameter
        for parameter in model.parameters()
        if parameter.requires_grad and id(parameter) not in special_ids
    ]
    groups = [{"name": "network", "params": base_parameters, "lr_scale": 1.0}]
    if lut_parameters:
        groups.append(
            {
                "name": "lut6_residual" if base_corrections else "lut4",
                "params": lut_parameters,
                "lr_scale": args.lut_lr / args.lr,
                "weight_decay": 0.0,
            }
        )
    if base_corrections:
        groups.append(
            {
                "name": "lut4_base_correction",
                "params": base_corrections,
                "lr_scale": args.base_lut_lr / args.lr,
                "weight_decay": 0.0,
            }
        )
    return torch.optim.Adam(groups, lr=args.lr, weight_decay=args.weight_decay)


def ramp_value(start, end, epoch, ramp_epochs):
    if ramp_epochs <= 1:
        return end
    progress = min(max(epoch, 0) / float(ramp_epochs - 1), 1.0)
    return start + progress * (end - start)


def configure_residual_alphas(model, epoch, args):
    modules = [
        module
        for module in model.modules()
        if isinstance(module, PairLUT4ComplexConv2d)
        and module.parameterization == "categorical_residual"
    ]
    if not modules:
        return None, None
    residual_alpha = ramp_value(
        args.residual_alpha_start,
        args.residual_alpha_end,
        epoch,
        args.residual_ramp_epochs,
    )
    base_alpha = ramp_value(
        args.base_alpha_start,
        args.base_alpha_end,
        epoch,
        args.base_ramp_epochs,
    )
    for module in modules:
        module.set_dominance_residual_alpha(residual_alpha)
        module.set_dominance_base_alpha(base_alpha)
    return residual_alpha, base_alpha


def residual_selection_is_eligible(residual_alpha, base_alpha, args):
    if residual_alpha is None:
        return True
    return (
        abs(residual_alpha - args.residual_alpha_end) <= 1e-8
        and abs(base_alpha - args.base_alpha_end) <= 1e-8
    )


def set_learning_rate(optimizer, value):
    for group in optimizer.param_groups:
        group["lr"] = value * group.get("lr_scale", 1.0)


def initialize_stage(model, args, device):
    if args.stage == "bireal_fp":
        if args.checkpoint:
            raise ValueError("bireal_fp starts from scratch and does not accept --checkpoint")
        return
    if args.train_from_scratch:
        LOGGER.warning("Training %s from scratch", args.stage)
        return
    if not args.checkpoint:
        raise ValueError(
            "{} requires --checkpoint unless --train-from-scratch is set".format(
                args.stage
            )
        )
    if args.stage == "bireal":
        load_bireal_from_fp(model, args.checkpoint, device=device)
        LOGGER.info("Initialized BiReal model from FP checkpoint %s", args.checkpoint)
    elif args.stage == "bireal_lut":
        _, compiled = load_lut_from_bireal(model, args.checkpoint, device=device)
        LOGGER.info(
            "Initialized LUT4 model from %s; compiled %d PairLUT4 operators",
            args.checkpoint,
            compiled,
        )
    elif args.stage == "lut6_residual":
        _, expanded = load_lut6_residual_from_lut4(
            model, args.checkpoint, device=device
        )
        LOGGER.info(
            "Initialized LUT6 residual from %s; expanded %d LUT4 operators",
            args.checkpoint,
            expanded,
        )
    else:
        load_lut6_lut4_residual(model, args.checkpoint, device=device)
        LOGGER.info(
            "Initialized LUT4 base-correction stage from %s", args.checkpoint
        )


def checkpoint_payload(model, optimizer, args, epoch, val_metrics):
    return {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "stage": args.stage,
        "phase": STAGE_TO_PHASE[args.stage],
        "epoch": epoch,
        "val_oa": val_metrics["oa"],
        "val_aa": val_metrics["aa"],
        "args": vars(args),
        "class_values": list(range(1, NUM_CLASSES + 1)),
    }


def main(argv=None):
    args = parse_args(argv)
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s %(levelname)s] %(message)s",
    )
    set_seed(args.seed)
    workdir = Path(args.workdir)
    workdir.mkdir(parents=True, exist_ok=True)
    split_seed = args.seed if args.split_seed is None else args.split_seed
    bundle = build_san_francisco_datasets(
        args.data_root,
        stk_file=args.stk_file,
        labels_file=args.labels_file,
        patch_size=args.patch_size,
        split_mode=args.split_mode,
        train_fraction=args.train_fraction,
        val_fraction=args.val_fraction,
        seed=split_seed,
        spatial_block_size=args.spatial_block_size,
        stats_path=workdir / "normalization.json",
    )
    device = torch.device(
        args.device if args.device != "cuda" or torch.cuda.is_available() else "cpu"
    )
    train_loader, val_loader, test_loader = build_loaders(bundle, args, device)
    model = build_san_francisco_bireal_model(
        args.stage,
        in_channels=NUM_INPUT_CHANNELS,
        num_classes=NUM_CLASSES,
        start_filters=args.start_filters,
        num_blocks=args.num_blocks,
        post_bn_mode=args.post_bn_mode,
        pair_lut_parameterization=args.pair_lut_parameterization,
        shortcut_mode=args.shortcut_mode,
    ).to(device)
    initialize_stage(model, args, device)
    initial_residual_alpha, initial_base_alpha = configure_residual_alphas(
        model, 0, args
    )
    criterion = nn.CrossEntropyLoss()

    lut_count = sum(
        isinstance(module, PairLUT4ComplexConv2d) for module in model.modules()
    )
    LOGGER.info(
        "Stage=%s phase=%d device=%s parameters=%d PairLUT4=%d shortcut=%s",
        args.stage,
        STAGE_TO_PHASE[args.stage],
        device,
        sum(parameter.numel() for parameter in model.parameters()),
        lut_count,
        args.shortcut_mode,
    )
    if initial_residual_alpha is not None:
        LOGGER.info(
            "Initial residual alpha=%.4f base alpha=%.4f LUT LR=%.6g base LUT LR=%.6g",
            initial_residual_alpha,
            initial_base_alpha,
            args.lut_lr,
            args.base_lut_lr,
        )
    LOGGER.info("Training seed=%d split seed=%d", args.seed, split_seed)
    LOGGER.info(
        "Split sizes: train=%d val=%d test=%d classes=%s",
        len(bundle.train),
        len(bundle.val),
        len(bundle.test),
        class_distribution(bundle.train),
    )
    if args.smoke_test:
        smoke_test_step(model, train_loader, criterion, device)
        return 0

    optimizer = build_optimizer(model, args)
    history = []
    best_val_oa = -1.0
    checkpoint_path = workdir / "best.pt"
    for epoch in range(args.epochs):
        current_lr = learning_rate(epoch, args)
        set_learning_rate(optimizer, current_lr)
        residual_alpha, base_alpha = configure_residual_alphas(model, epoch, args)
        train_metrics = run_epoch(model, train_loader, criterion, device, optimizer)
        val_metrics = run_epoch(model, val_loader, criterion, device)
        selection_eligible = residual_selection_is_eligible(
            residual_alpha, base_alpha, args
        )
        record = {
            "epoch": epoch + 1,
            "lr": current_lr,
            "residual_alpha": residual_alpha,
            "base_alpha": base_alpha,
            "selection_eligible": selection_eligible,
            "train": train_metrics,
            "val": val_metrics,
        }
        history.append(record)
        LOGGER.info(
            "Epoch %03d lr=%.6g residual_alpha=%s base_alpha=%s "
            "train_loss=%.6f train_OA=%.4f val_loss=%.6f val_OA=%.4f val_AA=%.4f",
            epoch + 1,
            current_lr,
            "n/a" if residual_alpha is None else "{:.4f}".format(residual_alpha),
            "n/a" if base_alpha is None else "{:.4f}".format(base_alpha),
            train_metrics["loss"],
            train_metrics["oa"],
            val_metrics["loss"],
            val_metrics["oa"],
            val_metrics["aa"],
        )
        if selection_eligible and val_metrics["oa"] > best_val_oa:
            best_val_oa = val_metrics["oa"]
            torch.save(
                checkpoint_payload(model, optimizer, args, epoch + 1, val_metrics),
                checkpoint_path,
            )
        save_json(workdir / "history.json", history)

    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model"], strict=True)
    test_metrics = run_epoch(model, test_loader, criterion, device)
    save_json(workdir / "test_metrics.json", test_metrics)
    if args.save_prediction_map:
        save_prediction_map(
            model, bundle.test, test_loader, device, workdir / "prediction_map.npy"
        )
    LOGGER.info(
        "Best epoch=%d val_OA=%.4f final test_OA=%.4f test_AA=%.4f",
        checkpoint["epoch"],
        checkpoint["val_oa"],
        test_metrics["oa"],
        test_metrics["aa"],
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())

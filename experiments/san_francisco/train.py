#!/usr/bin/env python3
"""Train an FP32 complex CNN baseline on San Francisco AIRSAR C3 data."""

import argparse
import json
import logging
from pathlib import Path
import sys

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from datasets.san_francisco import build_san_francisco_datasets
from experiments.flevoland.models import FlevolandComplexCNN
from experiments.flevoland.train import run_epoch, set_seed


LOGGER = logging.getLogger("san_francisco")
NUM_INPUT_CHANNELS = 6
NUM_CLASSES = 5


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", default="data/san_francisco")
    parser.add_argument("--stk-file", default="san_francisco900x1024.stk")
    parser.add_argument("--labels-file", default="SF-AIRSAR-label2d.png")
    parser.add_argument("--patch-size", type=int, default=11)
    parser.add_argument("--split-mode", choices=("spatial", "random"), default="spatial")
    parser.add_argument("--spatial-block-size", type=int, default=64)
    parser.add_argument("--train-fraction", type=float, default=0.1)
    parser.add_argument("--val-fraction", type=float, default=0.1)
    parser.add_argument("--conv-type", choices=("fp", "bireal"), default="fp")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--split-seed", type=int)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--workdir", default="runs/san_francisco/fp_baseline")
    parser.add_argument("--smoke-test", action="store_true")
    parser.add_argument("--save-prediction-map", action="store_true")
    return parser.parse_args(argv)


def class_distribution(dataset):
    raw_labels = dataset.labels[
        dataset.coordinates[:, 0], dataset.coordinates[:, 1]
    ]
    return np.bincount(raw_labels, minlength=NUM_CLASSES + 1)[1:].tolist()


def build_loaders(bundle, args, device):
    common = {
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
        "pin_memory": device.type == "cuda",
    }
    if args.num_workers > 0:
        common["persistent_workers"] = True
    return (
        DataLoader(bundle.train, shuffle=True, **common),
        DataLoader(bundle.val, shuffle=False, **common),
        DataLoader(bundle.test, shuffle=False, **common),
    )


def smoke_test_step(model, loader, criterion, device):
    model.train()
    inputs, targets = next(iter(loader))
    inputs = inputs.to(device)
    targets = targets.to(device)
    LOGGER.info(
        "Smoke input shape=%s dtype=%s real=[%.4f, %.4f] imag=[%.4f, %.4f]",
        tuple(inputs.shape),
        inputs.dtype,
        inputs.real.min().item(),
        inputs.real.max().item(),
        inputs.imag.min().item(),
        inputs.imag.max().item(),
    )
    features = model.forward_features(inputs)
    logits = model(inputs)
    loss = criterion(logits, targets)
    loss.backward()
    finite_gradients = all(
        parameter.grad is None or torch.isfinite(parameter.grad).all()
        for parameter in model.parameters()
    )
    if not torch.is_complex(features):
        raise RuntimeError("Complex feature path was not preserved")
    if logits.shape != (inputs.size(0), NUM_CLASSES):
        raise RuntimeError("Unexpected logits shape: {}".format(tuple(logits.shape)))
    if not torch.isfinite(loss) or not finite_gradients:
        raise FloatingPointError("Smoke test produced a non-finite loss or gradient")
    LOGGER.info(
        "Smoke forward/backward passed: features=%s logits=%s loss=%.6f",
        tuple(features.shape),
        tuple(logits.shape),
        loss.item(),
    )


def save_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def save_prediction_map(model, dataset, loader, device, path):
    model.eval()
    prediction_map = np.zeros(dataset.labels.shape, dtype=np.uint8)
    offset = 0
    with torch.no_grad():
        for inputs, _ in loader:
            predictions = model(inputs.to(device)).argmax(dim=1).cpu().numpy() + 1
            coordinates = dataset.coordinates[offset:offset + len(predictions)]
            prediction_map[coordinates[:, 0], coordinates[:, 1]] = predictions
            offset += len(predictions)
    np.save(path, prediction_map)


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
    if args.device == "cuda" and device.type == "cpu":
        LOGGER.warning("CUDA is unavailable; falling back to CPU")
    train_loader, val_loader, test_loader = build_loaders(bundle, args, device)
    model = FlevolandComplexCNN(
        in_channels=NUM_INPUT_CHANNELS,
        num_classes=NUM_CLASSES,
        conv_type=args.conv_type,
    ).to(device)
    criterion = nn.CrossEntropyLoss()

    LOGGER.info(
        "San Francisco AIRSAR: channels=%d classes=%d split=%s device=%s",
        NUM_INPUT_CHANNELS,
        NUM_CLASSES,
        args.split_mode,
        device,
    )
    LOGGER.info(
        "Split sizes: train=%d val=%d test=%d",
        len(bundle.train),
        len(bundle.val),
        len(bundle.test),
    )
    LOGGER.info("Training seed=%d split seed=%d", args.seed, split_seed)
    LOGGER.info("Training class distribution: %s", class_distribution(bundle.train))

    if args.smoke_test:
        smoke_test_step(model, train_loader, criterion, device)
        return 0

    optimizer = torch.optim.Adam(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    best_val_oa = -1.0
    history = []
    checkpoint_path = workdir / "best.pt"
    for epoch in range(1, args.epochs + 1):
        train_metrics = run_epoch(model, train_loader, criterion, device, optimizer)
        val_metrics = run_epoch(model, val_loader, criterion, device)
        record = {"epoch": epoch, "train": train_metrics, "val": val_metrics}
        history.append(record)
        LOGGER.info(
            "Epoch %03d train_loss=%.6f train_OA=%.4f val_loss=%.6f "
            "val_OA=%.4f val_AA=%.4f",
            epoch,
            train_metrics["loss"],
            train_metrics["oa"],
            val_metrics["loss"],
            val_metrics["oa"],
            val_metrics["aa"],
        )
        if val_metrics["oa"] > best_val_oa:
            best_val_oa = val_metrics["oa"]
            torch.save(
                {
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "epoch": epoch,
                    "val_oa": best_val_oa,
                    "args": vars(args),
                    "class_values": list(range(1, NUM_CLASSES + 1)),
                },
                checkpoint_path,
            )
        save_json(workdir / "history.json", history)

    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model"])
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

#!/usr/bin/env python3
"""Train a small center-pixel classifier to diagnose Pol-InSAR representations."""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch import nn


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from datasets.pol_insar_island import (
    CLASS_NAMES,
    PolInSARScene,
    official_split,
    train_scales,
    transform_t3_patch,
)


def _sample(coordinates, maximum, rng):
    if maximum <= 0 or len(coordinates) <= maximum:
        return coordinates
    return coordinates[rng.choice(len(coordinates), size=maximum, replace=False)]


def _features(scene, coordinates, representation, scales):
    rows, cols = coordinates[:, 0], coordinates[:, 1]
    values = np.stack([channel[rows, cols] for channel in scene.channels]).astype(np.complex64, copy=False)
    values = transform_t3_patch(values, representation, scales)
    return np.concatenate((values.real, values.imag), axis=0).T.astype(np.float32, copy=False)


def _evaluate(model, features, labels, device, batch_size):
    correct = 0
    with torch.no_grad():
        for start in range(0, len(labels), batch_size):
            x = torch.from_numpy(features[start:start + batch_size]).to(device)
            correct += int((model(x).argmax(1).cpu().numpy() == labels[start:start + batch_size]).sum())
    return correct / len(labels)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", default="datasets_external/pol_insar_island")
    parser.add_argument("--representation", choices=("raw_rms", "trace", "log_coherence"), default="raw_rms")
    parser.add_argument("--patch-size", type=int, default=11)
    parser.add_argument("--val-fraction", type=float, default=0.15)
    parser.add_argument("--block-size", type=int, default=16)
    parser.add_argument("--split-seed", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-train", type=int, default=200000)
    parser.add_argument("--max-eval", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--hidden", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    torch.manual_seed(args.seed)
    scene = PolInSARScene.load(args.data_root)
    coordinates, split = official_split(scene, args.patch_size, args.val_fraction, args.block_size, args.split_seed)
    scales = train_scales(scene, coordinates["train"])
    selected = {
        "train": _sample(coordinates["train"], args.max_train, rng),
        "validation": _sample(coordinates["validation"], args.max_eval, rng),
        "test": _sample(coordinates["test"], args.max_eval, rng),
    }
    labels = {
        "train": np.asarray(scene.train_labels[selected["train"][:, 0], selected["train"][:, 1]] - 1, dtype=np.int64),
        "validation": np.asarray(scene.train_labels[selected["validation"][:, 0], selected["validation"][:, 1]] - 1, dtype=np.int64),
        "test": np.asarray(scene.test_labels[selected["test"][:, 0], selected["test"][:, 1]] - 1, dtype=np.int64),
    }
    features = {name: _features(scene, coords, args.representation, scales) for name, coords in selected.items()}
    mean, std = features["train"].mean(0), features["train"].std(0).clip(1e-6)
    features = {name: (value - mean) / std for name, value in features.items()}
    device = torch.device(args.device if args.device != "cuda" or torch.cuda.is_available() else "cpu")
    layers = [nn.Linear(12, args.hidden), nn.ReLU(), nn.Linear(args.hidden, len(CLASS_NAMES))] if args.hidden else [nn.Linear(12, len(CLASS_NAMES))]
    model = nn.Sequential(*layers).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss()
    history = []
    for epoch in range(args.epochs):
        model.train()
        order = rng.permutation(len(labels["train"]))
        for start in range(0, len(order), args.batch_size):
            index = order[start:start + args.batch_size]
            x = torch.from_numpy(features["train"][index]).to(device)
            y = torch.from_numpy(labels["train"][index]).to(device)
            optimizer.zero_grad(set_to_none=True)
            criterion(model(x), y).backward()
            optimizer.step()
        model.eval()
        history.append({"epoch": epoch + 1, **{name: _evaluate(model, features[name], labels[name], device, args.batch_size) for name in features}})
    report = {"args": vars(args), "split": split, "final": history[-1], "history": history}
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report["final"], indent=2))


if __name__ == "__main__":
    main()

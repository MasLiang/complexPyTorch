#!/usr/bin/env python3
"""Render a T3 diagonal pseudo-RGB image and train/validation/test overlays."""

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from datasets.pol_insar_island import PolInSARScene, official_split


def _scaled_log(channel):
    image = np.log(np.maximum(np.asarray(channel.real), 1e-6))
    low, high = np.quantile(image, (0.01, 0.99))
    return np.clip((image - low) / max(high - low, 1e-6), 0.0, 1.0)


def _mask(shape, coordinates):
    result = np.zeros(shape, dtype=bool)
    result[coordinates[:, 0], coordinates[:, 1]] = True
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", default="datasets_external/pol_insar_island")
    parser.add_argument("--patch-size", type=int, default=11)
    parser.add_argument("--val-fraction", type=float, default=0.15)
    parser.add_argument("--block-size", type=int, default=16)
    parser.add_argument("--split-seed", type=int, default=0)
    parser.add_argument("--stride", type=int, default=4)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    scene = PolInSARScene.load(args.data_root)
    coordinates, _ = official_split(scene, args.patch_size, args.val_fraction, args.block_size, args.split_seed)
    # T3 diagonals are real and form a physically meaningful pseudo-RGB view.
    rgb = np.stack((_scaled_log(scene.channels[2]), _scaled_log(scene.channels[1]), _scaled_log(scene.channels[0])), axis=-1)
    train = _mask(scene.shape, coordinates["train"])
    validation = _mask(scene.shape, coordinates["validation"])
    test = _mask(scene.shape, coordinates["test"])
    overlay = rgb.copy()
    for mask, color in ((train, (0.1, 0.5, 1.0)), (validation, (1.0, 0.85, 0.0)), (test, (1.0, 0.1, 0.1))):
        overlay[mask] = 0.45 * overlay[mask] + 0.55 * np.asarray(color)

    stride = args.stride
    figure, axes = plt.subplots(1, 2, figsize=(16, 8), constrained_layout=True)
    axes[0].imshow(rgb[::stride, ::stride])
    axes[0].set_title("T3 diagonal pseudo-RGB: [T33, T22, T11]")
    axes[1].imshow(overlay[::stride, ::stride])
    axes[1].set_title("Selected centers: train blue, validation yellow, test red")
    for axis in axes:
        axis.set_axis_off()
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=180)


if __name__ == "__main__":
    main()

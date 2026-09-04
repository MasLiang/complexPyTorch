#!/usr/bin/env python3
"""Decode and validate the San Francisco AIRSAR complex dataset."""

import argparse
from pathlib import Path
import sys

import numpy as np

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from datasets.san_francisco import (
    C3_CHANNELS,
    build_san_francisco_datasets,
    c3_to_hermitian,
)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", default="data/san_francisco")
    parser.add_argument("--patch-size", type=int, default=11)
    parser.add_argument("--split-mode", choices=("random", "spatial"), default="spatial")
    parser.add_argument("--train-fraction", type=float, default=0.1)
    parser.add_argument("--val-fraction", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--spatial-block-size", type=int, default=64)
    parser.add_argument("--stats-path")
    return parser.parse_args()


def describe(values):
    return "min={:.6g} max={:.6g} mean={:.6g} std={:.6g}".format(
        float(values.min()),
        float(values.max()),
        float(values.mean()),
        float(values.std()),
    )


def main():
    args = parse_args()
    stats_path = args.stats_path or str(Path(args.data_root) / "normalization.json")
    bundle = build_san_francisco_datasets(
        args.data_root,
        patch_size=args.patch_size,
        split_mode=args.split_mode,
        train_fraction=args.train_fraction,
        val_fraction=args.val_fraction,
        seed=args.seed,
        spatial_block_size=args.spatial_block_size,
        stats_path=stats_path,
    )

    image = bundle.image
    print("STK shape:", bundle.stokes.shape)
    print("STK header bytes:", bundle.stokes.header_bytes)
    print("C3 shape:", image.shape)
    print("C3 dtype:", image.dtype)
    for index, name in enumerate(C3_CHANNELS):
        print("{} real: {}".format(name, describe(image[index].real)))
        print("{} imag: {}".format(name, describe(image[index].imag)))
        if index >= 3:
            phase = np.angle(image[index])
            nonzero = np.abs(image[index]) > 0
            print(
                "{} phase: {} nonzero_imag_fraction={:.6f}".format(
                    name,
                    describe(phase[nonzero]),
                    float(np.count_nonzero(image[index].imag) / image[index].imag.size),
                )
            )

    hermitian = c3_to_hermitian(image)
    hermitian_error = np.max(np.abs(hermitian - np.conj(hermitian.swapaxes(0, 1))))
    print("Hermitian max error:", float(hermitian_error))
    print("Diagonal imaginary max:", float(np.max(np.abs(image[:3].imag))))

    values, counts = np.unique(bundle.labels, return_counts=True)
    print("Raw label distribution:", dict(zip(values.tolist(), counts.tolist())))
    for split_name in ("train", "val", "test"):
        coordinates = bundle.coordinates[split_name]
        split_labels = bundle.labels[coordinates[:, 0], coordinates[:, 1]] - 1
        distribution = np.bincount(split_labels, minlength=5)
        print(
            "{} length={} class_distribution={}".format(
                split_name, len(coordinates), distribution.tolist()
            )
        )

    sample, target = bundle.train[0]
    print("Sample shape:", tuple(sample.shape))
    print("Sample dtype:", sample.dtype)
    print("Sample real range:", (float(sample.real.min()), float(sample.real.max())))
    print("Sample imag range:", (float(sample.imag.min()), float(sample.imag.max())))
    print("Sample target:", int(target))
    print("Normalization stats:", stats_path)


if __name__ == "__main__":
    main()

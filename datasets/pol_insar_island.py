"""Pol-InSAR-Island FP1/L T3 loading with official-mask leakage protection."""

from dataclasses import dataclass
from pathlib import Path
import re

import numpy as np
import torch
from torch.utils.data import Dataset


T3_CHANNELS = ("T11", "T22", "T33", "T12", "T13", "T23")
REPRESENTATIONS = ("raw_rms", "trace", "log_coherence")
CLASS_NAMES = (
    "Tidal flat", "Water", "Coastal shrub", "Dense high vegetation",
    "White dune", "Peat bog", "Grey dune", "Couch grass",
    "Upper saltmarsh", "Lower saltmarsh", "Sand", "Settlement",
)


def _header_dimensions(path):
    text = Path(path).read_text(encoding="latin1", errors="replace")
    values = {}
    for key in ("samples", "lines"):
        match = re.search(r"^\s*{}\s*=\s*(\d+)".format(key), text, re.MULTILINE | re.IGNORECASE)
        if match is None:
            raise ValueError("Missing {} in ENVI header {}".format(key, path))
        values[key] = int(match.group(1))
    return values["lines"], values["samples"]


def _box_sum(mask, radius):
    """Number of true pixels in each in-bounds square neighborhood."""
    value = np.asarray(mask, dtype=np.int32)
    height, width = value.shape
    padded = np.pad(value, ((1, 0), (1, 0)), mode="constant")
    integral = padded.cumsum(axis=0).cumsum(axis=1)
    rows = np.arange(height)
    cols = np.arange(width)
    top = np.maximum(rows - radius, 0)
    bottom = np.minimum(rows + radius + 1, height)
    left = np.maximum(cols - radius, 0)
    right = np.minimum(cols + radius + 1, width)
    return (
        integral[bottom[:, None], right[None, :]]
        - integral[top[:, None], right[None, :]]
        - integral[bottom[:, None], left[None, :]]
        + integral[top[:, None], left[None, :]]
    )


@dataclass
class PolInSARScene:
    root: Path
    channels: tuple
    train_labels: np.ndarray
    test_labels: np.ndarray

    @property
    def shape(self):
        return self.train_labels.shape

    @classmethod
    def load(cls, root, t3_dir="audit/fp1_l_t3", labels_dir="audit/labels"):
        root = Path(root)
        first = root / t3_dir / "T11.bin"
        height, width = _header_dimensions(first.with_suffix(first.suffix + ".hdr"))
        arrays = []
        for name in T3_CHANNELS:
            if name in ("T11", "T22", "T33"):
                path = root / t3_dir / (name + ".bin")
                real = np.memmap(path, dtype="<f4", mode="r", shape=(height, width))
                arrays.append(real.astype(np.complex64))
            else:
                real = np.memmap(root / t3_dir / (name + "_real.bin"), dtype="<f4", mode="r", shape=(height, width))
                imag = np.memmap(root / t3_dir / (name + "_imag.bin"), dtype="<f4", mode="r", shape=(height, width))
                arrays.append((real + 1j * imag).astype(np.complex64))
        train_path = root / labels_dir / "label_train.bin"
        test_path = root / labels_dir / "label_test.bin"
        train_labels = np.memmap(train_path, dtype=np.uint8, mode="r", shape=(height, width))
        test_labels = np.memmap(test_path, dtype=np.uint8, mode="r", shape=(height, width))
        if np.any((train_labels > 0) & (test_labels > 0)):
            raise ValueError("Official Pol-InSAR train/test masks overlap")
        return cls(root, tuple(arrays), train_labels, test_labels)

    def patch(self, row, col, radius):
        height, width = self.shape
        rows = np.arange(row - radius, row + radius + 1)
        cols = np.arange(col - radius, col + radius + 1)
        rows = np.where(rows < 0, -rows, rows)
        rows = np.where(rows >= height, 2 * height - rows - 2, rows)
        cols = np.where(cols < 0, -cols, cols)
        cols = np.where(cols >= width, 2 * width - cols - 2, cols)
        return np.stack([channel[np.ix_(rows, cols)] for channel in self.channels]).astype(np.complex64, copy=False)


def _spatial_validation_assignment(labels, block_size, fraction, seed):
    height, width = labels.shape
    rows = (height + block_size - 1) // block_size
    cols = (width + block_size - 1) // block_size
    dominant = np.zeros((rows, cols), dtype=np.uint8)
    for row in range(rows):
        for col in range(cols):
            tile = labels[row * block_size:min((row + 1) * block_size, height), col * block_size:min((col + 1) * block_size, width)]
            counts = np.bincount(tile.ravel(), minlength=len(CLASS_NAMES) + 1)
            if counts[1:].sum():
                dominant[row, col] = counts[1:].argmax() + 1
    rng = np.random.default_rng(seed)
    validation = np.zeros_like(dominant, dtype=bool)
    for class_id in range(1, len(CLASS_NAMES) + 1):
        indices = np.argwhere(dominant == class_id)
        rng.shuffle(indices)
        selected = int(round(len(indices) * fraction))
        if len(indices) > 1:
            selected = max(1, min(selected, len(indices) - 1))
        validation[tuple(indices[:selected].T)] = True
    pixel_assignment = np.repeat(np.repeat(validation, block_size, axis=0), block_size, axis=1)[:height, :width]
    return pixel_assignment


def official_split(scene, patch_size=11, val_fraction=0.15, block_size=64, seed=0):
    """Split official train labels spatially and remove cross-split patches."""
    if patch_size % 2 != 1:
        raise ValueError("patch_size must be odd")
    if block_size <= patch_size:
        raise ValueError("block_size must exceed patch_size")
    radius = patch_size // 2
    train_mask = np.asarray(scene.train_labels) > 0
    test_mask = np.asarray(scene.test_labels) > 0
    val_assignment = _spatial_validation_assignment(scene.train_labels, block_size, val_fraction, seed)
    train_assignment = ~val_assignment
    neighborhood_area = (2 * radius + 1) ** 2
    train_safe = _box_sum(train_assignment, radius) == neighborhood_area
    val_safe = _box_sum(val_assignment, radius) == neighborhood_area
    no_test_context = _box_sum(test_mask, radius) == 0
    no_train_context = _box_sum(train_mask, radius) == 0
    masks = {
        "train": train_mask & train_assignment & train_safe & no_test_context,
        "validation": train_mask & val_assignment & val_safe & no_test_context,
        "test": test_mask & no_train_context,
    }
    coordinates = {name: np.argwhere(mask).astype(np.int32) for name, mask in masks.items()}
    for name, value in coordinates.items():
        if not len(value):
            raise ValueError("Pol-InSAR {} split is empty after guard-band filtering".format(name))
    report = {
        "patch_size": patch_size,
        "guard_band_radius": radius,
        "validation_block_size": block_size,
        "validation_fraction_of_official_train": val_fraction,
        "split_seed": seed,
        "official_train_labeled": int(train_mask.sum()),
        "official_test_labeled": int(test_mask.sum()),
        "ignored_unlabeled": int((~train_mask & ~test_mask).sum()),
        "discarded_train_or_validation_guard": int(train_mask.sum() - len(coordinates["train"]) - len(coordinates["validation"])),
        "discarded_test_guard": int(test_mask.sum() - len(coordinates["test"])),
        "counts": {
            name: np.bincount((scene.train_labels if name != "test" else scene.test_labels)[coords[:, 0], coords[:, 1]], minlength=len(CLASS_NAMES) + 1)[1:].tolist()
            for name, coords in coordinates.items()
        },
    }
    return coordinates, report


def train_scales(scene, coordinates, eps=1e-8):
    rows, cols = coordinates[:, 0], coordinates[:, 1]
    powers = []
    for channel in scene.channels:
        values = channel[rows, cols]
        powers.append(float(np.mean(np.abs(values) ** 2)))
    return np.sqrt(np.maximum(np.asarray(powers, dtype=np.float32), eps))


class PolInSARPatches(Dataset):
    def __init__(self, scene, coordinates, labels, scales, patch_size=11, representation="raw_rms"):
        if representation not in REPRESENTATIONS:
            raise ValueError("Unknown Pol-InSAR representation: {}".format(representation))
        self.scene = scene
        self.coordinates = np.asarray(coordinates, dtype=np.int32)
        self.labels = labels
        self.scales = np.asarray(scales, dtype=np.float32)
        self.radius = patch_size // 2
        self.representation = representation

    def __len__(self):
        return len(self.coordinates)

    def __getitem__(self, index):
        row, col = self.coordinates[index]
        patch = self.scene.patch(int(row), int(col), self.radius)
        patch = transform_t3_patch(patch, self.representation, self.scales)
        label = int(self.labels[row, col]) - 1
        return torch.from_numpy(np.ascontiguousarray(patch)), label


def transform_t3_patch(patch, representation, scales=None, eps=1e-6):
    """Return a phase-preserving T3 representation for one `[6,H,W]` patch."""
    if representation == "raw_rms":
        if scales is None:
            raise ValueError("raw_rms requires train RMS scales")
        scale_shape = (len(scales),) + (1,) * (patch.ndim - 1)
        return patch / np.asarray(scales, dtype=np.float32).reshape(scale_shape)
    diagonal = np.maximum(patch[:3].real, eps)
    if representation == "trace":
        return patch / np.maximum(diagonal.sum(axis=0, keepdims=True), eps)
    if representation == "log_coherence":
        output = np.empty_like(patch)
        output[:3] = np.log(diagonal).astype(np.complex64)
        output[3] = patch[3] / np.sqrt(diagonal[0] * diagonal[1])
        output[4] = patch[4] / np.sqrt(diagonal[0] * diagonal[2])
        output[5] = patch[5] / np.sqrt(diagonal[1] * diagonal[2])
        return output
    raise ValueError("Unknown Pol-InSAR representation: {}".format(representation))


def build_pol_insar_t3_datasets(root, patch_size=11, val_fraction=0.15, block_size=64, seed=0, representation="raw_rms"):
    if representation not in REPRESENTATIONS:
        raise ValueError("Unknown Pol-InSAR representation: {}".format(representation))
    scene = PolInSARScene.load(root)
    coordinates, report = official_split(scene, patch_size, val_fraction, block_size, seed)
    scales = train_scales(scene, coordinates["train"])
    datasets = {
        "train": PolInSARPatches(scene, coordinates["train"], scene.train_labels, scales, patch_size, representation),
        "validation": PolInSARPatches(scene, coordinates["validation"], scene.train_labels, scales, patch_size, representation),
        "test": PolInSARPatches(scene, coordinates["test"], scene.test_labels, scales, patch_size, representation),
    }
    report.update({
        "input": {"frequency": "L", "flight_path": "FP1", "channels": list(T3_CHANNELS), "shape": [6, *scene.shape]},
        "representation": {
            "name": representation,
            "equation": {
                "raw_rms": "x_c / sqrt(E_train[|x_c|^2] + eps)",
                "trace": "T_ij / (T_11 + T_22 + T_33 + eps)",
                "log_coherence": "[log(T_11), log(T_22), log(T_33), T_12/sqrt(T_11*T_22), T_13/sqrt(T_11*T_33), T_23/sqrt(T_22*T_33)]",
            }[representation],
        },
        "normalization": {"mode": "train_rms", "scales": scales.tolist()} if representation == "raw_rms" else {"mode": "intrinsic_pixelwise"},
        "class_names": list(CLASS_NAMES),
    })
    return datasets, report

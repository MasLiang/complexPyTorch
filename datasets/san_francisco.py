"""San Francisco AIRSAR STK-MLC decoding and patch classification data."""

from dataclasses import dataclass
import json
import math
from pathlib import Path
import re

import numpy as np
import torch
from torch.utils.data import Dataset


DEFAULT_HEIGHT = 900
DEFAULT_WIDTH = 1024
BYTES_PER_PIXEL = 10
C3_CHANNELS = ("C11", "C22", "C33", "C12", "C13", "C23")


@dataclass(frozen=True)
class AIRSARStokes:
    """Nine independent entries of the symmetric 4x4 Stokes matrix."""

    m11: np.ndarray
    m12: np.ndarray
    m13: np.ndarray
    m14: np.ndarray
    m22: np.ndarray
    m23: np.ndarray
    m24: np.ndarray
    m33: np.ndarray
    m34: np.ndarray
    m44: np.ndarray
    header_bytes: int = 0
    general_scale_factor: float = 1.0

    @property
    def shape(self):
        return self.m11.shape


def _infer_dimensions(path, height, width):
    if height is not None and width is not None:
        return int(height), int(width)
    match = re.search(r"(\d+)[xX](\d+)", Path(path).name)
    if match:
        inferred_height, inferred_width = map(int, match.groups())
        return height or inferred_height, width or inferred_width
    return height or DEFAULT_HEIGHT, width or DEFAULT_WIDTH


def _signed_square(value):
    value = value.astype(np.float32)
    return np.sign(value) * np.square(value / np.float32(127.0))


def decode_stk(path, height=None, width=None):
    """Decode an old-processor AIRSAR STK-MLC file into Stokes entries.

    This follows PolSARpro's ``airsar_convert.c`` path for processor versions
    prior to 5.0. The ten compressed bytes are pixel-interleaved signed chars.
    The San Francisco sample has no embedded header; a prefixed header is
    tolerated when its length is exactly the bytes preceding the image payload.
    """

    path = Path(path)
    height, width = _infer_dimensions(path, height, width)
    payload_size = height * width * BYTES_PER_PIXEL
    file_size = path.stat().st_size
    if file_size < payload_size:
        raise ValueError(
            "STK file is too small: expected at least {} bytes for {}x{}, got {}".format(
                payload_size, height, width, file_size
            )
        )
    header_bytes = file_size - payload_size
    with path.open("rb") as handle:
        if header_bytes:
            handle.seek(header_bytes)
        raw = np.fromfile(handle, dtype=np.int8, count=payload_size)
    if raw.size != payload_size:
        raise ValueError("Could not read the complete STK image payload")

    compressed = raw.reshape(height, width, BYTES_PER_PIXEL)
    byte = [compressed[..., index].astype(np.float32) for index in range(10)]

    # The old STK mantissa is a signed char in the reference importer.
    m11 = (np.float32(1.5) + byte[1] / np.float32(254.0)) * np.exp2(byte[0])
    m12 = m11 * (byte[2] / np.float32(127.0))
    m13 = m11 * _signed_square(byte[3])
    m14 = m11 * _signed_square(byte[4])
    m23 = m11 * _signed_square(byte[5])
    m24 = m11 * _signed_square(byte[6])
    m33 = m11 * (byte[7] / np.float32(127.0))
    m34 = m11 * (byte[8] / np.float32(127.0))
    m44 = m11 * (byte[9] / np.float32(127.0))
    m22 = m11 - m33 - m44

    entries = (m11, m12, m13, m14, m22, m23, m24, m33, m34, m44)
    if not all(np.isfinite(entry).all() for entry in entries):
        raise ValueError("Decoded STK data contains NaN or Inf")
    return AIRSARStokes(
        *(entry.astype(np.float32, copy=False) for entry in entries),
        header_bytes=header_bytes,
    )


def stokes_to_c3(stokes):
    """Construct the standard lexicographic 3x3 covariance matrix channels.

    The basis is ``[S_hh, sqrt(2) S_hv, S_vv]``. Only the upper Hermitian
    triangle is returned in channel order C11, C22, C33, C12, C13, C23.
    """

    root_two = np.float32(math.sqrt(2.0))
    c11 = 2.0 * stokes.m11 + 2.0 * stokes.m12 - stokes.m33 - stokes.m44
    c22 = 2.0 * (stokes.m33 + stokes.m44)
    c33 = 2.0 * stokes.m11 - 2.0 * stokes.m12 - stokes.m33 - stokes.m44
    c12 = root_two * (
        (stokes.m13 + stokes.m23) - 1j * (stokes.m14 + stokes.m24)
    )
    c13 = (stokes.m33 - stokes.m44) - 2j * stokes.m34
    c23 = root_two * (
        (stokes.m13 - stokes.m23) + 1j * (stokes.m24 - stokes.m14)
    )
    zeros = np.zeros(stokes.shape, dtype=np.float32)
    channels = (
        c11 + 1j * zeros,
        c22 + 1j * zeros,
        c33 + 1j * zeros,
        c12,
        c13,
        c23,
    )
    image = np.stack(channels).astype(np.complex64, copy=False)
    if not np.isfinite(image.real).all() or not np.isfinite(image.imag).all():
        raise ValueError("C3 construction produced NaN or Inf")
    return image


def c3_to_hermitian(image):
    """Expand six upper-triangle channels to a 3x3 Hermitian matrix."""

    if image.shape[0] != 6:
        raise ValueError("Expected six C3 channels, got {}".format(image.shape))
    matrix = np.empty((3, 3) + image.shape[1:], dtype=np.complex64)
    matrix[0, 0], matrix[1, 1], matrix[2, 2] = image[:3]
    matrix[0, 1], matrix[0, 2], matrix[1, 2] = image[3:]
    matrix[1, 0] = np.conj(matrix[0, 1])
    matrix[2, 0] = np.conj(matrix[0, 2])
    matrix[2, 1] = np.conj(matrix[1, 2])
    return matrix


def load_label_map(path, expected_shape=None):
    try:
        from PIL import Image
    except ImportError as exc:
        raise RuntimeError("Pillow is required to load the AIRSAR label PNG") from exc
    labels = np.asarray(Image.open(path), dtype=np.uint8)
    if labels.ndim != 2:
        raise ValueError("Expected a grayscale label map, got {}".format(labels.shape))
    if expected_shape is not None and labels.shape != tuple(expected_shape):
        raise ValueError(
            "Label shape {} does not match image shape {}".format(
                labels.shape, tuple(expected_shape)
            )
        )
    unexpected = np.setdiff1d(np.unique(labels), np.arange(6, dtype=np.uint8))
    if unexpected.size:
        raise ValueError("Unexpected label values: {}".format(unexpected.tolist()))
    return labels


def load_san_francisco_scene(stk_path, labels_path, height=None, width=None):
    stokes = decode_stk(stk_path, height=height, width=width)
    image = stokes_to_c3(stokes)
    labels = load_label_map(labels_path, expected_shape=image.shape[1:])
    return image, labels, stokes


@dataclass(frozen=True)
class ComplexNormalizationStats:
    real_mean: np.ndarray
    real_std: np.ndarray
    imag_mean: np.ndarray
    imag_std: np.ndarray

    @classmethod
    def fit(cls, image, coordinates, eps=1e-6):
        rows, cols = coordinates[:, 0], coordinates[:, 1]
        samples = image[:, rows, cols]
        real_std = samples.real.std(axis=1)
        imag_std = samples.imag.std(axis=1)
        return cls(
            samples.real.mean(axis=1).astype(np.float32),
            np.maximum(real_std, eps).astype(np.float32),
            samples.imag.mean(axis=1).astype(np.float32),
            np.maximum(imag_std, eps).astype(np.float32),
        )

    def apply(self, image):
        shape = (-1, 1, 1)
        real = (image.real - self.real_mean.reshape(shape)) / self.real_std.reshape(shape)
        imag = (image.imag - self.imag_mean.reshape(shape)) / self.imag_std.reshape(shape)
        return (real + 1j * imag).astype(np.complex64)

    def save(self, path):
        payload = {
            key: getattr(self, key).tolist()
            for key in ("real_mean", "real_std", "imag_mean", "imag_std")
        }
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(json.dumps(payload, indent=2) + "\n")

    @classmethod
    def load(cls, path):
        payload = json.loads(Path(path).read_text())
        return cls(
            *(np.asarray(payload[key], dtype=np.float32) for key in (
                "real_mean", "real_std", "imag_mean", "imag_std"
            ))
        )


def _stratified_random_split(labels, train_fraction, val_fraction, seed):
    rng = np.random.default_rng(seed)
    splits = {"train": [], "val": [], "test": []}
    for raw_class in range(1, 6):
        coordinates = np.argwhere(labels == raw_class)
        rng.shuffle(coordinates)
        train_end = max(1, int(len(coordinates) * train_fraction))
        val_end = train_end + int(len(coordinates) * val_fraction)
        splits["train"].append(coordinates[:train_end])
        splits["val"].append(coordinates[train_end:val_end])
        splits["test"].append(coordinates[val_end:])
    return {
        name: np.concatenate(parts).astype(np.int64, copy=False)
        for name, parts in splits.items()
    }


def _spatial_block_split(
    labels,
    train_fraction,
    val_fraction,
    seed,
    patch_radius,
    block_size,
):
    """Assign spatial blocks and remove patches crossing split boundaries."""

    if block_size <= 2 * patch_radius:
        raise ValueError("spatial block size must be larger than the patch diameter")
    height, width = labels.shape
    block_rows = math.ceil(height / block_size)
    block_cols = math.ceil(width / block_size)
    rng = np.random.default_rng(seed)
    assignment = np.full((block_rows, block_cols), 2, dtype=np.int8)
    dominant_class = np.zeros((block_rows, block_cols), dtype=np.uint8)
    for block_row in range(block_rows):
        for block_col in range(block_cols):
            tile = labels[
                block_row * block_size:min((block_row + 1) * block_size, height),
                block_col * block_size:min((block_col + 1) * block_size, width),
            ]
            counts = np.bincount(tile.ravel(), minlength=6)
            if counts[1:].sum():
                dominant_class[block_row, block_col] = np.argmax(counts[1:]) + 1

    # Stratify whole spatial blocks by their dominant labeled class. Pixels
    # whose patches touch differently assigned blocks are removed below.
    for raw_class in range(1, 6):
        blocks = np.argwhere(dominant_class == raw_class)
        rng.shuffle(blocks)
        if len(blocks) < 3:
            continue
        train_count = max(1, round(len(blocks) * train_fraction))
        val_count = max(1, round(len(blocks) * val_fraction))
        for block_row, block_col in blocks[:train_count]:
            assignment[block_row, block_col] = 0
        for block_row, block_col in blocks[train_count:train_count + val_count]:
            assignment[block_row, block_col] = 1

    result = {"train": [], "val": [], "test": []}
    names = ("train", "val", "test")
    for row, col in np.argwhere(labels > 0):
        row_min = max(0, row - patch_radius) // block_size
        row_max = min(height - 1, row + patch_radius) // block_size
        col_min = max(0, col - patch_radius) // block_size
        col_max = min(width - 1, col + patch_radius) // block_size
        local = assignment[row_min:row_max + 1, col_min:col_max + 1]
        split_id = int(assignment[row // block_size, col // block_size])
        if np.all(local == split_id):
            result[names[split_id]].append((row, col))
    return {
        name: np.asarray(coordinates, dtype=np.int64).reshape(-1, 2)
        for name, coordinates in result.items()
    }


def split_labeled_coordinates(
    labels,
    mode="spatial",
    train_fraction=0.1,
    val_fraction=0.1,
    seed=0,
    patch_size=11,
    spatial_block_size=64,
):
    if train_fraction <= 0 or val_fraction < 0:
        raise ValueError("train_fraction must be positive and val_fraction non-negative")
    if train_fraction + val_fraction >= 1:
        raise ValueError("train_fraction + val_fraction must be less than one")
    if mode == "random":
        return _stratified_random_split(labels, train_fraction, val_fraction, seed)
    if mode == "spatial":
        return _spatial_block_split(
            labels,
            train_fraction,
            val_fraction,
            seed,
            patch_size // 2,
            spatial_block_size,
        )
    raise ValueError("Unknown split mode: {}".format(mode))


class SanFranciscoAIRSARDataset(Dataset):
    """Complex C3 patches centered on labeled AIRSAR pixels."""

    def __init__(
        self,
        image,
        labels,
        coordinates,
        normalization=None,
        patch_size=11,
        image_is_padded=False,
    ):
        if patch_size <= 0 or patch_size % 2 == 0:
            raise ValueError("patch_size must be a positive odd integer")
        self.patch_size = patch_size
        self.radius = patch_size // 2
        self.coordinates = np.asarray(coordinates, dtype=np.int64)
        self.labels = labels
        if image_is_padded:
            self.image = image
        else:
            normalized = normalization.apply(image) if normalization is not None else image
            self.image = np.pad(
                normalized,
                ((0, 0), (self.radius, self.radius), (self.radius, self.radius)),
                mode="reflect",
            )

    def __len__(self):
        return len(self.coordinates)

    def __getitem__(self, index):
        row, col = self.coordinates[index]
        patch = self.image[:, row:row + self.patch_size, col:col + self.patch_size]
        sample = torch.from_numpy(np.ascontiguousarray(patch)).to(torch.complex64)
        target = torch.tensor(int(self.labels[row, col]) - 1, dtype=torch.long)
        return sample, target


@dataclass(frozen=True)
class SanFranciscoDataBundle:
    train: SanFranciscoAIRSARDataset
    val: SanFranciscoAIRSARDataset
    test: SanFranciscoAIRSARDataset
    image: np.ndarray
    labels: np.ndarray
    normalization: ComplexNormalizationStats
    stokes: AIRSARStokes
    coordinates: dict


def build_san_francisco_datasets(
    data_root,
    stk_file="san_francisco900x1024.stk",
    labels_file="SF-AIRSAR-label2d.png",
    patch_size=11,
    split_mode="spatial",
    train_fraction=0.1,
    val_fraction=0.1,
    seed=0,
    spatial_block_size=64,
    stats_path=None,
):
    data_root = Path(data_root)
    stk_path = data_root / stk_file
    labels_path = data_root / labels_file
    image, labels, stokes = load_san_francisco_scene(stk_path, labels_path)
    coordinates = split_labeled_coordinates(
        labels,
        mode=split_mode,
        train_fraction=train_fraction,
        val_fraction=val_fraction,
        seed=seed,
        patch_size=patch_size,
        spatial_block_size=spatial_block_size,
    )
    if any(len(coordinates[name]) == 0 for name in ("train", "val", "test")):
        raise ValueError("At least one dataset split is empty")
    normalization = ComplexNormalizationStats.fit(image, coordinates["train"])
    if stats_path is not None:
        normalization.save(stats_path)
    normalized = normalization.apply(image)
    radius = patch_size // 2
    padded = np.pad(
        normalized,
        ((0, 0), (radius, radius), (radius, radius)),
        mode="reflect",
    )
    datasets = {
        name: SanFranciscoAIRSARDataset(
            padded,
            labels,
            coordinates[name],
            patch_size=patch_size,
            image_is_padded=True,
        )
        for name in ("train", "val", "test")
    }
    return SanFranciscoDataBundle(
        datasets["train"],
        datasets["val"],
        datasets["test"],
        image,
        labels,
        normalization,
        stokes,
        coordinates,
    )

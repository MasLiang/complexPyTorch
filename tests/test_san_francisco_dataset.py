from pathlib import Path

import numpy as np
import torch

from datasets.san_francisco import (
    ComplexNormalizationStats,
    SanFranciscoAIRSARDataset,
    c3_to_hermitian,
    decode_stk,
    split_labeled_coordinates,
    stokes_to_c3,
)


def _write_stk(path: Path, height=4, width=5):
    compressed = np.zeros((height, width, 10), dtype=np.int8)
    compressed[..., 0] = -2
    compressed[..., 1] = 20
    compressed[..., 2] = 15
    compressed[..., 3] = 25
    compressed[..., 4] = -18
    compressed[..., 5] = 12
    compressed[..., 6] = -9
    compressed[..., 7] = 30
    compressed[..., 8] = -7
    compressed[..., 9] = 20
    compressed.tofile(path)
    return compressed


def test_old_stk_decoder_matches_reference_equations(tmp_path):
    compressed = _write_stk(tmp_path / "tiny4x5.stk")
    stokes = decode_stk(tmp_path / "tiny4x5.stk")
    b = compressed[0, 0].astype(np.float32)
    m11 = (1.5 + b[1] / 254.0) * np.exp2(b[0])
    assert stokes.shape == (4, 5)
    np.testing.assert_allclose(stokes.m11, m11)
    np.testing.assert_allclose(stokes.m12, m11 * b[2] / 127.0)
    np.testing.assert_allclose(
        stokes.m14, m11 * np.sign(b[4]) * np.square(b[4] / 127.0)
    )
    np.testing.assert_allclose(stokes.m22, stokes.m11 - stokes.m33 - stokes.m44)


def test_c3_is_complex_and_exactly_hermitian(tmp_path):
    _write_stk(tmp_path / "tiny4x5.stk")
    image = stokes_to_c3(decode_stk(tmp_path / "tiny4x5.stk"))
    matrix = c3_to_hermitian(image)
    assert image.shape == (6, 4, 5)
    assert image.dtype == np.complex64
    assert np.count_nonzero(image[3:].imag) > 0
    assert np.count_nonzero(image[:3].imag) == 0
    np.testing.assert_allclose(matrix, np.conj(matrix.swapaxes(0, 1)))


def test_patch_dataset_normalizes_and_remaps_labels():
    rng = np.random.default_rng(4)
    image = (
        rng.normal(size=(6, 24, 24)) + 1j * rng.normal(size=(6, 24, 24))
    ).astype(np.complex64)
    image[:3] = image[:3].real
    labels = np.zeros((24, 24), dtype=np.uint8)
    labels[2:12, 2:12] = 1
    labels[12:22, 12:22] = 2
    coordinates = np.argwhere(labels > 0)
    stats = ComplexNormalizationStats.fit(image, coordinates)
    dataset = SanFranciscoAIRSARDataset(
        image, labels, coordinates, stats, patch_size=11
    )
    sample, target = dataset[0]
    assert sample.shape == (6, 11, 11)
    assert sample.dtype == torch.complex64
    assert target.dtype == torch.long
    assert int(target) == 0


def test_random_and_spatial_splits_are_deterministic_and_disjoint():
    labels = np.zeros((160, 160), dtype=np.uint8)
    for class_index in range(1, 6):
        start = (class_index - 1) * 30
        labels[start:start + 40, :] = class_index
    for mode in ("random", "spatial"):
        first = split_labeled_coordinates(
            labels,
            mode=mode,
            train_fraction=0.2,
            val_fraction=0.2,
            seed=9,
            patch_size=11,
            spatial_block_size=32,
        )
        second = split_labeled_coordinates(
            labels,
            mode=mode,
            train_fraction=0.2,
            val_fraction=0.2,
            seed=9,
            patch_size=11,
            spatial_block_size=32,
        )
        assert all(np.array_equal(first[name], second[name]) for name in first)
        coordinate_sets = {
            name: set(map(tuple, coordinates.tolist())) for name, coordinates in first.items()
        }
        assert coordinate_sets["train"].isdisjoint(coordinate_sets["val"])
        assert coordinate_sets["train"].isdisjoint(coordinate_sets["test"])
        assert coordinate_sets["val"].isdisjoint(coordinate_sets["test"])

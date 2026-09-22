#!/usr/bin/env python3
"""Inspect raw complex-data releases without materializing full datasets.

The tool inventories files and samples common scientific containers (HDF5,
NumPy, pickle, ENVI rasters, GeoTIFF, and CSV).  It is deliberately read-only:
raw archives are never extracted or modified by this audit.
"""

import argparse
import csv
import hashlib
import json
import math
import pickle
import re
import struct
import tarfile
import zipfile
from collections import Counter
from pathlib import Path

import numpy as np


def file_digest(path, chunk_size=1024 * 1024):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                return digest.hexdigest()
            digest.update(chunk)


def finite_array_stats(array, sample_limit):
    value = np.asarray(array)
    if value.size > sample_limit:
        flat = value.reshape(-1)
        indices = np.linspace(0, flat.size - 1, sample_limit, dtype=np.int64)
        value = flat[indices]
    if not np.issubdtype(value.dtype, np.number) and not np.iscomplexobj(value):
        return {"dtype": str(value.dtype), "sampled_values": int(value.size)}
    is_complex = np.iscomplexobj(value)
    real = np.real(value).astype(np.float64, copy=False)
    imag = np.imag(value).astype(np.float64, copy=False) if is_complex else None
    finite = np.isfinite(real) if imag is None else np.isfinite(real) & np.isfinite(imag)
    valid_real = real[finite]
    result = {
        "dtype": str(value.dtype),
        "sampled_values": int(value.size),
        "complex": bool(is_complex),
        "nan_or_inf": int(value.size - finite.sum()),
        "real_range": None if not valid_real.size else [float(valid_real.min()), float(valid_real.max())],
    }
    if not is_complex:
        return result
    valid_imag = imag[finite]
    magnitude = np.abs(value[finite])
    phase = np.angle(value[finite])
    result.update({
        "imag_range": None if not valid_imag.size else [float(valid_imag.min()), float(valid_imag.max())],
        "imag_nonzero_fraction": float(np.count_nonzero(valid_imag) / max(valid_imag.size, 1)),
        "mean_magnitude": None if not magnitude.size else float(magnitude.mean()),
        "magnitude_std": None if not magnitude.size else float(magnitude.std()),
        "phase_histogram": np.histogram(phase, bins=12, range=(-math.pi, math.pi))[0].tolist(),
        "real_imag_correlation": None if valid_real.size < 2 or valid_real.std() == 0 or valid_imag.std() == 0 else float(np.corrcoef(valid_real, valid_imag)[0, 1]),
    })
    return result


def iq_layout_stats(array, sample_limit):
    """Recognize common real float encodings of an I/Q complex signal."""
    value = np.asarray(array)
    axis = None
    layout = None
    if value.ndim >= 2 and value.shape[-1] == 2:
        axis, layout = -1, "last_axis_iq"
    elif value.ndim >= 3 and value.shape[1] == 2:
        axis, layout = 1, "channel_axis_iq"
    if axis is None or np.iscomplexobj(value) or not np.issubdtype(value.dtype, np.floating):
        return {}
    real = np.take(value, 0, axis=axis)
    imag = np.take(value, 1, axis=axis)
    return {
        "iq_encoding": layout,
        "canonical_complex_shape": list(real.shape),
        "iq_complex_stats": finite_array_stats(
            real.astype(np.complex64) + 1j * imag.astype(np.complex64),
            sample_limit,
        ),
    }


def inspect_numpy(path, sample_limit):
    if path.suffix == ".npy":
        value = np.load(path, mmap_mode="r")
        return {"arrays": {path.name: {"shape": list(value.shape), **finite_array_stats(value, sample_limit), **iq_layout_stats(value, sample_limit)}}}
    archive = np.load(path, mmap_mode="r", allow_pickle=False)
    return {"arrays": {name: {"shape": list(archive[name].shape), **finite_array_stats(archive[name], sample_limit), **iq_layout_stats(archive[name], sample_limit)} for name in archive.files}}


def inspect_hdf5(path, sample_limit):
    import h5py

    result = {"datasets": {}}
    with h5py.File(path, "r") as handle:
        def visitor(name, node):
            if isinstance(node, h5py.Dataset):
                slices = tuple(slice(0, min(size, 16)) for size in node.shape)
                sample = node[slices] if node.shape else node[()]
                result["datasets"][name] = {"shape": list(node.shape), **finite_array_stats(sample, sample_limit), **iq_layout_stats(sample, sample_limit)}
        handle.visititems(visitor)
    return result


def inspect_pickle(path, sample_limit):
    with path.open("rb") as handle:
        value = pickle.load(handle, encoding="latin1")
    if isinstance(value, dict):
        arrays = {}
        labels = Counter()
        for key, item in value.items():
            if isinstance(item, np.ndarray):
                arrays[repr(key)] = {"shape": list(item.shape), **finite_array_stats(item, sample_limit), **iq_layout_stats(item, sample_limit)}
            if isinstance(key, tuple) and len(key) >= 2:
                labels[repr(key[0])] += len(item)
        return {"pickle_type": "dict", "keys": len(value), "arrays": arrays, "class_balance_from_keys": dict(sorted(labels.items()))}
    return {"pickle_type": type(value).__name__}


def parse_envi_header(path):
    content = path.read_text(encoding="latin1", errors="replace")
    values = {}
    for key in ("samples", "lines", "bands", "data type", "byte order", "interleave"):
        match = re.search(r"^\s*{}\s*=\s*([^\n]+)".format(re.escape(key)), content, flags=re.MULTILINE | re.IGNORECASE)
        if match:
            values[key.replace(" ", "_")] = match.group(1).strip().strip("{}")
    return values


def inspect_legacy_float_tiff(path, sample_limit):
    """Read simple uncompressed MATLAB-written float TIFFs without rasterio."""
    type_sizes = {1: 1, 3: 2, 4: 4}
    with path.open("rb") as handle:
        byte_order = handle.read(2)
        if byte_order != b"II" or handle.read(2) != b"*\x00":
            return {"error": "unsupported TIFF byte order or signature"}
        ifd_offset = struct.unpack("<I", handle.read(4))[0]
        handle.seek(ifd_offset)
        entries = struct.unpack("<H", handle.read(2))[0]
        tags = {}
        for _ in range(entries):
            tag, field_type, count, value = struct.unpack("<HHII", handle.read(12))
            tags[tag] = (field_type, count, value)

        def read_tag(tag):
            field_type, count, value = tags[tag]
            size = type_sizes[field_type] * count
            payload = struct.pack("<I", value)[:size] if size <= 4 else None
            if payload is None:
                handle.seek(value)
                payload = handle.read(size)
            formats = {1: "B", 3: "H", 4: "I"}
            return struct.unpack("<{}{}".format(count, formats[field_type]), payload)

        width, height = read_tag(256)[0], read_tag(257)[0]
        samples = read_tag(277)[0]
        bits = read_tag(258)
        sample_format = read_tag(339) if 339 in tags else (1,) * samples
        offset, byte_count = read_tag(273)[0], read_tag(279)[0]
        if not (set(bits) == {32} and set(sample_format) == {3}):
            return {"error": "unsupported legacy TIFF sample layout"}
        expected = width * height * samples * 4
        if byte_count < expected:
            return {"error": "legacy TIFF strip is shorter than declared shape"}
        handle.seek(offset)
        sample = np.frombuffer(handle.read(expected), dtype="<f4").reshape(height, width, samples)

    result = {
        "reader": "legacy_matlab_tiff",
        "shape": list(sample.shape),
        "bits_per_sample": list(bits),
        "sample_format": list(sample_format),
        **finite_array_stats(sample, sample_limit),
    }
    if samples == 4:
        complex_planes = np.stack((sample[..., 0] + 1j * sample[..., 1], sample[..., 2] + 1j * sample[..., 3]))
        result["four_plane_complex_encoding"] = {
            "canonical_complex_shape": list(complex_planes.shape),
            "pairing": ["band0+j*band1", "band2+j*band3"],
            **finite_array_stats(complex_planes, sample_limit),
        }
    return result


def inspect_geotiff(path, sample_limit):
    try:
        import rasterio
    except ImportError:
        try:
            from PIL import Image
        except ImportError:
            return inspect_legacy_float_tiff(path, sample_limit)
        try:
            with Image.open(path) as image:
                sample = np.asarray(image)
                return {
                    "reader": "Pillow",
                    "shape": list(sample.shape),
                    "mode": image.mode,
                    **finite_array_stats(sample, sample_limit),
                }
        except Exception:
            return inspect_legacy_float_tiff(path, sample_limit)
    with rasterio.open(path) as source:
        sample = source.read(window=((0, min(source.height, 64)), (0, min(source.width, 64))))
        return {
            "reader": "rasterio",
            "shape": [source.count, source.height, source.width],
            "dtypes": list(source.dtypes),
            **finite_array_stats(sample, sample_limit),
        }


def inspect_archive(path):
    if zipfile.is_zipfile(path):
        with zipfile.ZipFile(path) as archive:
            members = archive.infolist()
            return {"archive_members": len(members), "archive_uncompressed_bytes": sum(item.file_size for item in members), "first_members": [item.filename for item in members[:20]]}
    if tarfile.is_tarfile(path):
        with tarfile.open(path) as archive:
            members = archive.getmembers()
            return {"archive_members": len(members), "archive_uncompressed_bytes": sum(item.size for item in members), "first_members": [item.name for item in members[:20]]}
    return {}


def inspect_csv(path):
    with path.open(newline="", encoding="utf-8", errors="replace") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
    counters = {}
    for field in ("label", "class", "modulation", "is_vessel", "is_fishing", "partition", "scene_id"):
        if rows and field in rows[0]:
            counters[field] = dict(Counter(row[field] for row in rows).most_common())
    return {"rows": len(rows), "columns": [] if not rows else list(rows[0]), "distributions": counters}


def inspect_file(path, sample_limit, checksum_limit):
    result = {"path": str(path), "bytes": path.stat().st_size}
    if path.stat().st_size <= checksum_limit:
        result["sha256"] = file_digest(path)
    suffix = path.suffix.lower()
    try:
        if suffix in (".npy", ".npz"):
            result.update(inspect_numpy(path, sample_limit))
        elif suffix in (".h5", ".hdf5"):
            result.update(inspect_hdf5(path, sample_limit))
        elif suffix in (".pkl", ".pickle"):
            result.update(inspect_pickle(path, sample_limit))
        elif suffix == ".hdr":
            result["envi_header"] = parse_envi_header(path)
        elif suffix in (".tif", ".tiff"):
            result["geotiff"] = inspect_geotiff(path, sample_limit)
        elif suffix == ".csv":
            result["labels"] = inspect_csv(path)
        elif suffix in (".zip", ".gz", ".bz2", ".tar"):
            result.update(inspect_archive(path))
    except Exception as exc:
        result["inspection_error"] = "{}: {}".format(type(exc).__name__, exc)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--sample-limit", type=int, default=100_000)
    parser.add_argument("--checksum-limit-mb", type=int, default=1024)
    parser.add_argument("--max-files", type=int, default=500)
    args = parser.parse_args(argv)
    files = sorted(path for path in args.root.rglob("*") if path.is_file())[:args.max_files]
    report = {
        "dataset": args.dataset,
        "root": str(args.root),
        "file_count_audited": len(files),
        "files": [inspect_file(path, args.sample_limit, args.checksum_limit_mb * 1024 * 1024) for path in files],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(args.output)


if __name__ == "__main__":
    main()

"""OpenSARShip native SLC chips with scene- and MMSI-disjoint splits."""

from collections import Counter, defaultdict
from dataclasses import dataclass
import io
import json
from pathlib import Path
import struct
import zipfile
import xml.etree.ElementTree as ET

import numpy as np
import torch
from torch.utils.data import Dataset


DEFAULT_CLASSES = ("Cargo", "Tanker", "Other Type")


def _read_float_tiff(payload):
    """Decode the audited uncompressed little-endian four-plane float TIFF."""
    handle = io.BytesIO(payload)
    if handle.read(2) != b"II" or handle.read(2) != b"*\x00":
        raise ValueError("OpenSARShip patch is not a little-endian TIFF")
    ifd_offset = struct.unpack("<I", handle.read(4))[0]
    handle.seek(ifd_offset)
    entries = struct.unpack("<H", handle.read(2))[0]
    tags = {}
    for _ in range(entries):
        tag, field_type, count, value = struct.unpack("<HHII", handle.read(12))
        tags[tag] = (field_type, count, value)
    sizes = {1: 1, 3: 2, 4: 4}
    formats = {1: "B", 3: "H", 4: "I"}

    def tag_value(tag):
        field_type, count, value = tags[tag]
        size = sizes[field_type] * count
        raw = struct.pack("<I", value)[:size] if size <= 4 else None
        if raw is None:
            handle.seek(value)
            raw = handle.read(size)
        return struct.unpack("<{}{}".format(count, formats[field_type]), raw)

    width, height = tag_value(256)[0], tag_value(257)[0]
    samples = tag_value(277)[0]
    bits = tag_value(258)
    sample_format = tag_value(339) if 339 in tags else (1,) * samples
    offset, byte_count = tag_value(273)[0], tag_value(279)[0]
    if samples != 4 or set(bits) != {32} or set(sample_format) != {3}:
        raise ValueError("Unexpected OpenSARShip TIFF layout")
    expected = width * height * samples * 4
    if byte_count < expected:
        raise ValueError("Truncated OpenSARShip TIFF payload")
    handle.seek(offset)
    values = np.frombuffer(handle.read(expected), dtype="<f4").reshape(height, width, samples)
    if not np.isfinite(values).all():
        raise ValueError("OpenSARShip TIFF contains NaN or Inf")
    return np.stack((values[..., 0] + 1j * values[..., 1], values[..., 2] + 1j * values[..., 3])).astype(np.complex64)


def _center_crop_or_pad(image, size):
    channels, height, width = image.shape
    output = np.zeros((channels, size, size), dtype=np.complex64)
    source_top = max((height - size) // 2, 0)
    source_left = max((width - size) // 2, 0)
    target_top = max((size - height) // 2, 0)
    target_left = max((size - width) // 2, 0)
    copy_height = min(height, size)
    copy_width = min(width, size)
    output[:, target_top:target_top + copy_height, target_left:target_left + copy_width] = image[:, source_top:source_top + copy_height, source_left:source_left + copy_width]
    return output


def _label_from_member(member):
    stem = Path(member).stem
    if "_x" not in stem:
        raise ValueError("Cannot infer OpenSARShip class from {}".format(member))
    return stem.rsplit("_x", 1)[0]


def build_manifest(root, classes=DEFAULT_CLASSES, force=False):
    root = Path(root)
    processed = root / "processed"
    processed.mkdir(parents=True, exist_ok=True)
    manifest_path = processed / "opensarship_slc_manifest.json"
    class_names = tuple(classes)
    if manifest_path.is_file() and not force:
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        if tuple(payload.get("classes", ())) == class_names:
            return payload
    archive_path = root / "raw" / "OpenSARShip_original_zip.zip"
    if not archive_path.is_file():
        raise FileNotFoundError("OpenSARShip archive not found: {}".format(archive_path))
    records = []
    with zipfile.ZipFile(archive_path) as outer:
        for scene_index, scene in enumerate(outer.infolist()):
            if "SLC" not in scene.filename or not scene.filename.endswith(".zip"):
                continue
            payload = outer.read(scene)
            with zipfile.ZipFile(io.BytesIO(payload)) as inner:
                patches = [item.filename for item in inner.infolist() if "/Patch/" in item.filename and item.filename.endswith(".tif")]
                if not patches:
                    continue
                xml_name = next((item.filename for item in inner.infolist() if item.filename.endswith("/Ship.xml")), None)
                if xml_name is None:
                    raise ValueError("SLC scene has patches but no Ship.xml: {}".format(scene.filename))
                ships = ET.fromstring(inner.read(xml_name)).findall("ship")
                if len(patches) != len(ships):
                    raise ValueError("Patch/XML count mismatch in {}: {} vs {}".format(scene.filename, len(patches), len(ships)))
                for patch_index, (member, ship) in enumerate(zip(patches, ships)):
                    label = _label_from_member(member)
                    if label not in class_names:
                        continue
                    mmsi = (ship.findtext("./AISShipInformation/MMSI") or "").strip()
                    if not mmsi or mmsi == "0":
                        continue
                    records.append({
                        "scene": scene.filename,
                        "member": member,
                        "mmsi": mmsi,
                        "label": label,
                        "scene_index": scene_index,
                        "patch_index": patch_index,
                        "cache": "cache/{:03d}_{:04d}.npy".format(scene_index, patch_index),
                    })
    if not records:
        raise ValueError("No OpenSARShip SLC records remain after class/identity filtering")
    payload = {"classes": list(class_names), "records": records, "archive": str(archive_path)}
    manifest_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return payload


class _UnionFind:
    def __init__(self, size):
        self.parent = list(range(size))

    def find(self, value):
        while self.parent[value] != value:
            self.parent[value] = self.parent[self.parent[value]]
            value = self.parent[value]
        return value

    def union(self, left, right):
        left, right = self.find(left), self.find(right)
        if left != right:
            self.parent[right] = left


def _components(records):
    groups = _UnionFind(len(records))
    first_scene, first_mmsi = {}, {}
    for index, record in enumerate(records):
        for lookup, key in ((first_scene, record["scene"]), (first_mmsi, record["mmsi"])):
            if key in lookup:
                groups.union(index, lookup[key])
            else:
                lookup[key] = index
    result = defaultdict(list)
    for index in range(len(records)):
        result[groups.find(index)].append(index)
    return list(result.values())


def _allocate_groups(records, groups, classes, seed, fractions=(0.70, 0.15, 0.15)):
    """Assign indivisible groups while keeping class/sample targets close."""
    class_to_index = {name: index for index, name in enumerate(classes)}
    vectors = []
    for group in groups:
        vector = np.zeros(len(classes), dtype=np.float64)
        for index in group:
            vector[class_to_index[records[index]["label"]]] += 1
        vectors.append(vector)
    total = np.sum(vectors, axis=0)
    target = np.asarray(fractions, dtype=np.float64)[:, None] * total[None, :]
    best = None
    rng = np.random.default_rng(seed)
    for _ in range(256):
        order = np.arange(len(groups))
        rng.shuffle(order)
        order = sorted(order, key=lambda index: (len(groups[index]), rng.random()), reverse=True)
        counts = np.zeros_like(target)
        allocation = [[] for _ in fractions]
        for group_index in order:
            vector = vectors[group_index]
            candidates = []
            for split in range(len(fractions)):
                proposal = counts.copy()
                proposal[split] += vector
                scale = np.maximum(target, 1.0)
                score = float(np.square((proposal - target) / scale).sum())
                score += 0.05 * float(np.square((proposal.sum(axis=1) - target.sum(axis=1)) / np.maximum(target.sum(axis=1), 1.0)).sum())
                candidates.append(score)
            split = int(np.argmin(candidates))
            counts[split] += vector
            allocation[split].extend(groups[group_index])
        score = float(np.square((counts - target) / np.maximum(target, 1.0)).sum())
        if min(len(item) for item in allocation) == 0:
            score += 1e6
        if best is None or score < best[0]:
            best = score, allocation
    names = ("train", "validation", "test")
    return {name: [records[index] for index in indices] for name, indices in zip(names, best[1])}


def _split_scene_disjoint_mmsi_filtered(records, classes, seed):
    """Split whole scenes then drop vessels seen in more than one split.

    The scene/MMSI graph in the audited release has one giant connected
    component, so treating connected components as indivisible leaves no test
    set.  This stricter alternative keeps scenes disjoint and removes every
    cross-split MMSI before training, yielding three identity-disjoint splits.
    """
    scene_groups = list(defaultdict(list, {
        scene: [index for index, record in enumerate(records) if record["scene"] == scene]
        for scene in sorted({record["scene"] for record in records})
    }).values())
    provisional = _allocate_groups(records, scene_groups, classes, seed)
    split_for_mmsi = defaultdict(set)
    for split, values in provisional.items():
        for record in values:
            split_for_mmsi[record["mmsi"]].add(split)
    splits = {
        split: [record for record in values if len(split_for_mmsi[record["mmsi"]]) == 1]
        for split, values in provisional.items()
    }
    if any(not values for values in splits.values()):
        raise ValueError("OpenSARShip strict scene/MMSI filtering left an empty split")
    return splits, int(sum(len(values) for values in provisional.values()) - sum(len(values) for values in splits.values()))

def _load_scene_payload(archive_path, scene):
    with zipfile.ZipFile(archive_path) as outer:
        return outer.read(scene)


def prepare_cache(root, records, chip_size):
    root = Path(root)
    archive_path = root / "raw" / "OpenSARShip_original_zip.zip"
    pending = defaultdict(list)
    for record in records:
        cache = root / "processed" / record["cache"]
        if not cache.is_file():
            pending[record["scene"]].append(record)
    for scene, scene_records in pending.items():
        payload = _load_scene_payload(archive_path, scene)
        with zipfile.ZipFile(io.BytesIO(payload)) as inner:
            for record in scene_records:
                image = _center_crop_or_pad(_read_float_tiff(inner.read(record["member"])), chip_size)
                cache = root / "processed" / record["cache"]
                cache.parent.mkdir(parents=True, exist_ok=True)
                np.save(cache, image)


def train_scales(root, records, eps=1e-8):
    root = Path(root)
    power_sum = np.zeros(2, dtype=np.float64)
    count = 0
    for record in records:
        value = np.load(root / "processed" / record["cache"], mmap_mode="r")
        power_sum += np.square(np.abs(value).reshape(2, -1).astype(np.float64)).sum(axis=1)
        count += value.shape[1] * value.shape[2]
    if not count:
        raise ValueError("OpenSARShip training split is empty")
    return np.sqrt(np.maximum(power_sum / count, eps)).astype(np.float32)


class OpenSARShipSLCDataset(Dataset):
    def __init__(self, root, records, classes, scales):
        self.root = Path(root)
        self.records = list(records)
        self.classes = tuple(classes)
        self.labels = {name: index for index, name in enumerate(self.classes)}
        self.scales = np.asarray(scales, dtype=np.float32)

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        record = self.records[index]
        image = np.load(self.root / "processed" / record["cache"])
        image = image / self.scales[:, None, None]
        return torch.from_numpy(np.ascontiguousarray(image.astype(np.complex64))), self.labels[record["label"]]


def _split_report(splits, classes):
    report = {}
    mmsi_sets, scene_sets = {}, {}
    for name, records in splits.items():
        counts = Counter(record["label"] for record in records)
        mmsi_sets[name] = {record["mmsi"] for record in records}
        scene_sets[name] = {record["scene"] for record in records}
        report[name] = {
            "chips": len(records),
            "unique_mmsi": len(mmsi_sets[name]),
            "scenes": len(scene_sets[name]),
            "class_counts": {label: int(counts[label]) for label in classes},
        }
    names = tuple(splits)
    report["leakage_check"] = {
        "mmsi_overlap": {"{}__{}".format(left, right): len(mmsi_sets[left] & mmsi_sets[right]) for pos, left in enumerate(names) for right in names[pos + 1:]},
        "scene_overlap": {"{}__{}".format(left, right): len(scene_sets[left] & scene_sets[right]) for pos, left in enumerate(names) for right in names[pos + 1:]},
    }
    if any(report["leakage_check"][key][pair] for key in ("mmsi_overlap", "scene_overlap") for pair in report["leakage_check"][key]):
        raise RuntimeError("OpenSARShip grouped split leaks MMSI or scene identity")
    return report


def build_opensarship_slc_datasets(root, chip_size=128, classes=DEFAULT_CLASSES, split_seed=0, force_manifest=False):
    root = Path(root)
    manifest = build_manifest(root, classes, force_manifest)
    records = manifest["records"]
    splits, dropped_cross_split_mmsi = _split_scene_disjoint_mmsi_filtered(records, tuple(classes), split_seed)
    retained = [record for values in splits.values() for record in values]
    prepare_cache(root, retained, chip_size)
    scales = train_scales(root, splits["train"])
    datasets = {name: OpenSARShipSLCDataset(root, values, classes, scales) for name, values in splits.items()}
    report = _split_report(splits, classes)
    report.update({
        "input": {"channels": ["VH", "VV"], "tensor": "complex64 [2,H,W]", "plane_mapping": ["band0+j*band1", "band2+j*band3"], "chip_size": chip_size, "crop_pad": "center crop if larger, zero pad if smaller; no interpolation"},
        "normalization": {"mode": "train_rms", "equation": "x_p / sqrt(E_train[|x_p|^2] + eps)", "scales": scales.tolist()},
        "classes": list(classes),
        "filtered_records": len(records),
        "dropped_cross_split_mmsi": dropped_cross_split_mmsi,
        "split_strategy": "scene-disjoint assignment followed by removal of every MMSI occurring in multiple splits",
        "split_seed": split_seed,
    })
    return datasets, report

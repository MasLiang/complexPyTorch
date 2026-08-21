#!/usr/bin/env python3
"""Measure dominance-bit and LUT6 address occupancy on real dataset samples."""

import argparse
import json
import logging
import math
import sys
from pathlib import Path
from types import SimpleNamespace

import torch
from torch.utils.data import DataLoader, Subset

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import training
from complexPyTorch.complexLayers import PairLUT4ComplexConv2d


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--datadir", type=Path, default=Path("data"))
    parser.add_argument("--split", choices=["train", "test"], default="test")
    parser.add_argument("--max-samples", type=int, default=10000)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def checkpoint_state(payload):
    for key in ("model", "state_dict"):
        if isinstance(payload.get(key), dict):
            return payload[key]
    raise RuntimeError("Checkpoint contains no model state")


def model_args(payload, args):
    defaults = vars(training.parse_args([]))
    defaults.update(payload.get("args", {}))
    defaults.update({
        "datadir": str(args.datadir),
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
        "ddp": False,
        "cpu": args.device == "cpu",
        "no_validation": True,
    })
    return SimpleNamespace(**defaults)


class OccupancyTracker:
    def __init__(self, model):
        self.modules = {}
        self.counts = {}
        self.handles = []
        for name, module in model.named_modules():
            if (
                isinstance(module, PairLUT4ComplexConv2d)
                and module.activation_encoding == "dominance"
                and module.lut_inputs == 6
            ):
                self.modules[name] = module
                self.counts[name] = torch.zeros(
                    module.group_num, 64, dtype=torch.int64
                )
                self.handles.append(
                    module.register_forward_pre_hook(
                        self._hook(name), with_kwargs=True
                    )
                )
        if not self.modules:
            raise RuntimeError("No dominance PairLUT6 modules found")

    def _hook(self, name):
        def capture(module, positional, keyword):
            inp = positional[0]
            source = keyword.get("dominance_source")
            if source is None:
                raise RuntimeError(
                    "{} did not receive dominance_source".format(name)
                )
            with torch.no_grad():
                patches = module._group_patches(
                    inp, dominance_source=source
                ).round().long()
                shifts = torch.tensor(
                    [32, 16, 8, 4, 2, 1],
                    device=patches.device,
                    dtype=torch.long,
                )
                addresses = (patches * shifts).sum(dim=-1)
                groups = torch.arange(
                    module.group_num,
                    device=addresses.device,
                    dtype=torch.long,
                ).view(1, 1, -1)
                indexed = addresses + groups * 64
                batch_counts = torch.bincount(
                    indexed.reshape(-1),
                    minlength=module.group_num * 64,
                ).reshape(module.group_num, 64)
                self.counts[name] += batch_counts.cpu()
        return capture

    def close(self):
        for handle in self.handles:
            handle.remove()


def normalized_entropy(counts):
    probabilities = counts.double()
    probabilities /= probabilities.sum(dim=-1, keepdim=True).clamp_min(1.0)
    terms = torch.where(
        probabilities > 0,
        -probabilities * probabilities.log(),
        torch.zeros_like(probabilities),
    )
    return terms.sum(dim=-1) / math.log(64.0)


def bit_statistics(counts):
    addresses = torch.arange(64)
    total = counts.sum().item()
    p0_one = counts[:, (addresses & 8) != 0].sum().item()
    p1_one = counts[:, (addresses & 1) != 0].sum().item()
    combinations = []
    for p0 in (0, 1):
        for p1 in (0, 1):
            mask = (((addresses >> 3) & 1) == p0) & (
                (addresses & 1) == p1
            )
            combinations.append(counts[:, mask].sum().item() / total)
    return {
        "p0_zero": 1.0 - p0_one / total,
        "p0_one": p0_one / total,
        "p1_zero": 1.0 - p1_one / total,
        "p1_one": p1_one / total,
        "p0p1_00": combinations[0],
        "p0p1_01": combinations[1],
        "p0p1_10": combinations[2],
        "p0p1_11": combinations[3],
    }


def sensitivity_statistics(module, counts):
    with torch.no_grad():
        categories = module._categorical_logits().argmax(dim=-1).cpu()
    addresses = torch.arange(64)
    p0_changed = categories != categories.index_select(2, addresses ^ 8)
    p1_changed = categories != categories.index_select(2, addresses ^ 1)
    changed = p0_changed | p1_changed
    weights = counts.unsqueeze(0).expand(module.out_channels, -1, -1)
    weighted_total = weights.sum().item()
    changed_total = changed.sum().item()
    return {
        "table_category_sensitive_fraction": changed.float().mean().item(),
        "hit_weighted_category_sensitive_fraction": (
            (changed * weights).sum().item() / weighted_total
        ),
        "sensitive_entries_unvisited_fraction": (
            (changed & (weights == 0)).sum().item() / max(changed_total, 1)
        ),
        "sensitive_entries_lt10_hits_fraction": (
            (changed & (weights < 10)).sum().item() / max(changed_total, 1)
        ),
        "sensitive_entries_lt100_hits_fraction": (
            (changed & (weights < 100)).sum().item() / max(changed_total, 1)
        ),
        "p0_table_sensitive_fraction": p0_changed.float().mean().item(),
        "p1_table_sensitive_fraction": p1_changed.float().mean().item(),
    }


def layer_report(name, module, counts):
    total = int(counts.sum())
    per_group_coverage = (counts > 0).float().mean(dim=1)
    entropy = normalized_entropy(counts)
    global_counts = counts.sum(dim=0)
    global_frequency = global_counts.double() / global_counts.sum()
    top = torch.topk(global_counts, k=8)
    bottom = torch.topk(global_counts, k=8, largest=False)
    report = {
        "name": name,
        "groups": module.group_num,
        "total_lut_lookups": total,
        **bit_statistics(counts),
        "group_address_coverage": {
            "mean": per_group_coverage.mean().item(),
            "median": per_group_coverage.median().item(),
            "min": per_group_coverage.min().item(),
            "groups_with_all_64": (per_group_coverage == 1).float().mean().item(),
        },
        "group_address_hit_fraction": {
            "zero": (counts == 0).float().mean().item(),
            "lt10": (counts < 10).float().mean().item(),
            "lt100": (counts < 100).float().mean().item(),
        },
        "normalized_address_entropy": {
            "mean": entropy.mean().item(),
            "median": entropy.median().item(),
            "min": entropy.min().item(),
        },
        "global_address_frequency": [
            float(value) for value in global_frequency
        ],
        "most_frequent_addresses": [
            {"address": int(address), "hits": int(hits)}
            for hits, address in zip(top.values, top.indices)
        ],
        "least_frequent_addresses": [
            {"address": int(address), "hits": int(hits)}
            for hits, address in zip(bottom.values, bottom.indices)
        ],
    }
    report.update(sensitivity_statistics(module, counts))
    return report


def aggregate(reports):
    lookups = sum(report["total_lut_lookups"] for report in reports)
    def weighted(key):
        return sum(
            report[key] * report["total_lut_lookups"]
            for report in reports
        ) / lookups
    return {
        "layers": len(reports),
        "total_lut_lookups": lookups,
        "p0_one": weighted("p0_one"),
        "p1_one": weighted("p1_one"),
        "p0p1_00": weighted("p0p1_00"),
        "p0p1_01": weighted("p0p1_01"),
        "p0p1_10": weighted("p0p1_10"),
        "p0p1_11": weighted("p0p1_11"),
        "group_address_zero_fraction": sum(
            report["group_address_hit_fraction"]["zero"] * report["groups"]
            for report in reports
        ) / sum(report["groups"] for report in reports),
        "hit_weighted_category_sensitive_fraction": weighted(
            "hit_weighted_category_sensitive_fraction"
        ),
    }


def main():
    args = parse_args()
    if args.max_samples < 1:
        raise ValueError("--max-samples must be positive")
    payload = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    saved_args = model_args(payload, args)
    logger = logging.getLogger("occupancy")
    logger.addHandler(logging.NullHandler())
    train_data, _, test_data, num_classes, _, _ = training.build_datasets(
        saved_args, logger, False, True
    )
    dataset = train_data if args.split == "train" else test_data
    sample_count = min(args.max_samples, len(dataset))
    dataset = Subset(dataset, range(sample_count))
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=args.device != "cpu",
    )

    device = torch.device(args.device)
    model = training.build_model(saved_args, num_classes).to(device)
    model.load_state_dict(checkpoint_state(payload), strict=True)
    model.eval()
    tracker = OccupancyTracker(model)
    with torch.inference_mode():
        for data, _ in loader:
            model(data.to(device, non_blocking=True))
    tracker.close()

    reports = [
        layer_report(name, tracker.modules[name], tracker.counts[name])
        for name in tracker.modules
    ]
    result = {
        "checkpoint": str(args.checkpoint),
        "split": args.split,
        "samples": sample_count,
        "aggregate": aggregate(reports),
        "layers": reports,
    }
    rendered = json.dumps(result, indent=2, sort_keys=True)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)


if __name__ == "__main__":
    main()

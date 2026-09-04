#!/usr/bin/env python3
"""Measure dominance entropy and per-group LUT address utilization."""

import argparse
import csv
import json
import math
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from complexPyTorch.complexLayers import PairLUT4ComplexConv2d
from datasets.san_francisco import build_san_francisco_datasets
from experiments.san_francisco.checkpoints import checkpoint_state, load_checkpoint
from experiments.san_francisco.models import build_san_francisco_bireal_model


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data-root", default="data/san_francisco")
    parser.add_argument("--split", choices=("train", "val", "test"), default="train")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--max-batches", type=int, default=100)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output-dir")
    return parser.parse_args(argv)


def binary_entropy(one_count, total):
    if total == 0:
        return 0.0
    probability = one_count / total
    if probability <= 0.0 or probability >= 1.0:
        return 0.0
    return -probability * math.log2(probability) - (1.0 - probability) * math.log2(1.0 - probability)


def utilization_metrics(counts):
    counts = counts.to(torch.float64)
    totals = counts.sum(dim=1)
    valid = totals > 0
    probabilities = counts[valid] / totals[valid, None]
    entropies = -(probabilities * probabilities.clamp_min(1e-30).log2()).sum(dim=1)
    used = (counts[valid] > 0).sum(dim=1).to(torch.float64)
    state_count = counts.size(1)
    global_counts = counts.sum(dim=0)
    global_probabilities = global_counts / global_counts.sum().clamp_min(1.0)
    global_entropy = -(
        global_probabilities * global_probabilities.clamp_min(1e-30).log2()
    ).sum()
    return {
        "states": state_count,
        "groups": int(valid.sum().item()),
        "mean_used_states": used.mean().item(),
        "min_used_states": int(used.min().item()),
        "max_used_states": int(used.max().item()),
        "mean_utilization": (used / state_count).mean().item(),
        "unused_entry_fraction": 1.0 - (used.sum() / (used.numel() * state_count)).item(),
        "mean_normalized_entropy": (entropies / math.log2(state_count)).mean().item(),
        "mean_effective_states": entropies.exp2().mean().item(),
        "global_used_states": int((global_counts > 0).sum().item()),
        "global_normalized_entropy": (global_entropy / math.log2(state_count)).item(),
    }


class LUTUsageCollector:
    def __init__(self, model):
        self.layers = {}
        self.handles = []
        for name, module in model.named_modules():
            if not isinstance(module, PairLUT4ComplexConv2d):
                continue
            self.layers[name] = {
                "module": module,
                "lut4_counts": torch.zeros(module.group_num, 16, dtype=torch.int64),
                "lut6_counts": (
                    torch.zeros(module.group_num, 64, dtype=torch.int64)
                    if module.lut_inputs == 6
                    else None
                ),
                "dominance_counts": torch.zeros(2, 2, dtype=torch.int64),
                "dominance_joint_counts": torch.zeros(4, dtype=torch.int64),
            }
            self.handles.append(
                module.register_forward_pre_hook(
                    self._make_hook(name), with_kwargs=True
                )
            )
        if not self.layers:
            raise RuntimeError("Checkpoint model contains no PairLUT operators")

    @staticmethod
    def _grouped_counts(addresses, group_num, state_count):
        group_ids = torch.arange(group_num, device=addresses.device).view(1, 1, -1)
        encoded = addresses + group_ids * state_count
        return torch.bincount(
            encoded.reshape(-1), minlength=group_num * state_count
        ).reshape(group_num, state_count).cpu()

    def _make_hook(self, name):
        def hook(module, args, kwargs):
            inp = args[0]
            dominance_source = kwargs.get("dominance_source")
            if dominance_source is None and len(args) > 1:
                dominance_source = args[1]
            patches = module._group_patches(inp, dominance_source=dominance_source)
            bits = (patches >= 0.5).to(torch.int64)
            record = self.layers[name]
            if module.lut_inputs == 4:
                lut4_address = (
                    bits[..., 0] * 8
                    + bits[..., 1] * 4
                    + bits[..., 2] * 2
                    + bits[..., 3]
                )
            else:
                lut6_address = (
                    bits[..., 0] * 32
                    + bits[..., 1] * 16
                    + bits[..., 2] * 8
                    + bits[..., 3] * 4
                    + bits[..., 4] * 2
                    + bits[..., 5]
                )
                record["lut6_counts"] += self._grouped_counts(
                    lut6_address, module.group_num, 64
                )
                lut4_address = (
                    bits[..., 0] * 8
                    + bits[..., 1] * 4
                    + bits[..., 3] * 2
                    + bits[..., 4]
                )
                for position, bit_index in enumerate((2, 5)):
                    record["dominance_counts"][position] += torch.bincount(
                        bits[..., bit_index].reshape(-1), minlength=2
                    ).cpu()
                joint = bits[..., 2] * 2 + bits[..., 5]
                record["dominance_joint_counts"] += torch.bincount(
                    joint.reshape(-1), minlength=4
                ).cpu()
            record["lut4_counts"] += self._grouped_counts(
                lut4_address, module.group_num, 16
            )

        return hook

    def close(self):
        for handle in self.handles:
            handle.remove()

    def report(self):
        report = []
        for name, record in self.layers.items():
            dominance_counts = record["dominance_counts"]
            dominance_total = int(dominance_counts.sum().item())
            layer = {
                "layer": name,
                "lut_inputs": record["module"].lut_inputs,
                "lut4": utilization_metrics(record["lut4_counts"]),
            }
            if record["lut6_counts"] is not None:
                combined_ones = int(dominance_counts[:, 1].sum().item())
                combined_total = dominance_total
                layer["dominance"] = {
                    "p_one": combined_ones / combined_total if combined_total else 0.0,
                    "entropy_bits": binary_entropy(combined_ones, combined_total),
                    "p0_one": (
                        dominance_counts[0, 1].item() / dominance_counts[0].sum().item()
                    ),
                    "p1_one": (
                        dominance_counts[1, 1].item() / dominance_counts[1].sum().item()
                    ),
                    "joint_counts": record["dominance_joint_counts"].tolist(),
                }
                layer["lut6"] = utilization_metrics(record["lut6_counts"])
            report.append(layer)
        return report


def checkpoint_model(checkpoint, device):
    metadata = checkpoint.get("args", {}) if isinstance(checkpoint, dict) else {}
    stage = checkpoint.get("stage") if isinstance(checkpoint, dict) else None
    if stage not in ("bireal_lut", "lut6_residual", "lut6_lut4_residual"):
        raise ValueError("Expected a LUT-stage checkpoint, got {!r}".format(stage))
    model = build_san_francisco_bireal_model(
        stage,
        in_channels=6,
        num_classes=5,
        start_filters=int(metadata.get("start_filters", 16)),
        num_blocks=int(metadata.get("num_blocks", 3)),
        post_bn_mode=metadata.get("post_bn_mode", "covariance"),
        pair_lut_parameterization=metadata.get(
            "pair_lut_parameterization", "categorical"
        ),
        shortcut_mode=metadata.get("shortcut_mode", "fp"),
    ).to(device)
    model.load_state_dict(checkpoint_state(checkpoint), strict=True)
    return model.eval(), stage, metadata


def write_csv(path, layers):
    fieldnames = [
        "layer", "lut_inputs", "dominance_p_one", "dominance_entropy_bits",
        "lut4_mean_used", "lut4_mean_utilization", "lut4_unused_fraction",
        "lut4_normalized_entropy", "lut6_mean_used", "lut6_mean_utilization",
        "lut6_unused_fraction", "lut6_normalized_entropy",
    ]
    with Path(path).open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for layer in layers:
            dominance = layer.get("dominance", {})
            lut6 = layer.get("lut6", {})
            writer.writerow({
                "layer": layer["layer"],
                "lut_inputs": layer["lut_inputs"],
                "dominance_p_one": dominance.get("p_one"),
                "dominance_entropy_bits": dominance.get("entropy_bits"),
                "lut4_mean_used": layer["lut4"]["mean_used_states"],
                "lut4_mean_utilization": layer["lut4"]["mean_utilization"],
                "lut4_unused_fraction": layer["lut4"]["unused_entry_fraction"],
                "lut4_normalized_entropy": layer["lut4"]["mean_normalized_entropy"],
                "lut6_mean_used": lut6.get("mean_used_states"),
                "lut6_mean_utilization": lut6.get("mean_utilization"),
                "lut6_unused_fraction": lut6.get("unused_entry_fraction"),
                "lut6_normalized_entropy": lut6.get("mean_normalized_entropy"),
            })


def main(argv=None):
    args = parse_args(argv)
    checkpoint_path = Path(args.checkpoint)
    output_dir = Path(args.output_dir) if args.output_dir else checkpoint_path.parent / "lut_usage"
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(
        args.device if args.device != "cuda" or torch.cuda.is_available() else "cpu"
    )
    checkpoint = load_checkpoint(checkpoint_path, device=device)
    model, stage, metadata = checkpoint_model(checkpoint, device)
    bundle = build_san_francisco_datasets(
        args.data_root,
        stk_file=metadata.get("stk_file", "san_francisco900x1024.stk"),
        labels_file=metadata.get("labels_file", "SF-AIRSAR-label2d.png"),
        patch_size=int(metadata.get("patch_size", 11)),
        split_mode=metadata.get("split_mode", "spatial"),
        spatial_block_size=int(metadata.get("spatial_block_size", 64)),
        train_fraction=float(metadata.get("train_fraction", 0.1)),
        val_fraction=float(metadata.get("val_fraction", 0.1)),
        seed=int(metadata.get("seed", 0)),
    )
    dataset = getattr(bundle, args.split)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )
    collector = LUTUsageCollector(model)
    samples = 0
    batches = 0
    try:
        with torch.no_grad():
            for batch_index, (inputs, _) in enumerate(loader):
                if args.max_batches > 0 and batch_index >= args.max_batches:
                    break
                model(inputs.to(device, non_blocking=True))
                samples += inputs.size(0)
                batches += 1
    finally:
        collector.close()
    layers = collector.report()
    report = {
        "checkpoint": str(checkpoint_path),
        "stage": stage,
        "split": args.split,
        "samples": samples,
        "batches": batches,
        "layers": layers,
    }
    (output_dir / "lut_usage.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    write_csv(output_dir / "lut_usage.csv", layers)
    print("layer                              H(p)   p(1)   LUT4 used  LUT6 used  LUT6 unused")
    for layer in layers:
        dominance = layer.get("dominance", {})
        lut6 = layer.get("lut6", {})
        print(
            "{:<34} {:>5}  {:>5}  {:>8.2f}  {:>8}  {:>10}".format(
                layer["layer"],
                "-" if not dominance else "{:.3f}".format(dominance["entropy_bits"]),
                "-" if not dominance else "{:.3f}".format(dominance["p_one"]),
                layer["lut4"]["mean_used_states"],
                "-" if not lut6 else "{:.2f}".format(lut6["mean_used_states"]),
                "-" if not lut6 else "{:.1%}".format(lut6["unused_entry_fraction"]),
            )
        )
    print("Saved {} and {}".format(output_dir / "lut_usage.json", output_dir / "lut_usage.csv"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Measure pair-LUT input-state occupancy and entropy from a checkpoint."""

import argparse
import json
import logging
import sys
from argparse import Namespace
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

import training
from complexPyTorch.complexLayers import PairLUTNeuronConv2d


def _checkpoint_model_args(payload):
    values = payload.get("args")
    if not isinstance(values, dict):
        raise RuntimeError("Checkpoint has no serialized training arguments")
    defaults = vars(training.parse_args([]))
    defaults.update(values)
    return Namespace(**defaults)


def _test_dataset(data_dir):
    normalize = transforms.Normalize(
        (0.4914, 0.4822, 0.4465),
        (0.2023, 0.1994, 0.2010),
    )
    return datasets.CIFAR10(
        root=data_dir,
        train=False,
        download=False,
        transform=transforms.Compose([transforms.ToTensor(), normalize]),
    )


class StateCollector:
    def __init__(self, model):
        self.counts = {}
        self.handles = []
        for name, module in model.named_modules():
            if isinstance(module, PairLUTNeuronConv2d):
                self.handles.append(
                    module.register_forward_pre_hook(self._hook(name))
                )

    def _hook(self, name):
        def hook(module, inputs):
            inp = inputs[0]
            real = (inp.real >= 0.0).to(torch.float32)
            imag = (inp.imag >= 0.0).to(torch.float32)
            dummy = torch.zeros(
                inp.size(0),
                2,
                inp.size(2),
                inp.size(3),
                device=inp.device,
                dtype=torch.float32,
            )
            bits = torch.cat([real, imag, dummy], dim=1)
            patches = F.unfold(
                bits,
                kernel_size=module.kernel_size,
                padding=module.padding,
                stride=module.stride,
            )
            locations = patches.size(-1)
            values = patches[:, module.unfold_indices.reshape(-1), :].view(
                inp.size(0), module.pair_num, module.LUT_K, locations
            )
            weights = torch.tensor(
                [8, 4, 2, 1], device=inp.device, dtype=torch.int64
            ).view(1, 1, 4, 1)
            states = (values.to(torch.int64) * weights).sum(dim=2)
            offsets = (
                torch.arange(module.pair_num, device=inp.device)
                .view(1, -1, 1)
                * 16
            )
            counts = torch.bincount(
                (states + offsets).reshape(-1),
                minlength=module.pair_num * 16,
            ).view(module.pair_num, 16)
            counts = counts.cpu()
            if name not in self.counts:
                self.counts[name] = counts
            else:
                self.counts[name] += counts

        return hook

    def close(self):
        for handle in self.handles:
            handle.remove()


def _summarize_counts(counts):
    totals = counts.sum(dim=1, keepdim=True).clamp_min(1)
    probabilities = counts.to(torch.float64) / totals
    entropy = -torch.where(
        probabilities > 0.0,
        probabilities * torch.log2(probabilities),
        torch.zeros_like(probabilities),
    ).sum(dim=1)
    occupied = (counts > 0).to(torch.float64).mean(dim=1)
    state_ids = torch.arange(16, dtype=torch.int64)
    bit_ones = []
    aggregate_counts = counts.sum(dim=0).to(torch.float64)
    aggregate_total = aggregate_counts.sum().clamp_min(1.0)
    for shift in (3, 2, 1, 0):
        selected = ((state_ids >> shift) & 1).to(torch.bool)
        bit_ones.append(float(aggregate_counts[selected].sum() / aggregate_total))
    return {
        "pairs": int(counts.size(0)),
        "samples_per_pair": int(counts.sum(dim=1).min()),
        "mean_entropy_bits": float(entropy.mean()),
        "minimum_entropy_bits": float(entropy.min()),
        "maximum_entropy_bits": float(entropy.max()),
        "mean_effective_states": float(torch.pow(2.0, entropy).mean()),
        "mean_occupied_fraction": float(occupied.mean()),
        "bit_ones_fraction": bit_ones,
    }


def summarize(checkpoint_path, data_dir, batches, batch_size, device):
    checkpoint_path = Path(checkpoint_path)
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    args = _checkpoint_model_args(payload)
    model = training.build_model(args, num_classes=10)
    model.load_state_dict(payload["model"], strict=True)
    model.to(device).eval()

    loader = DataLoader(
        _test_dataset(data_dir),
        batch_size=batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=device.type == "cuda",
    )
    collector = StateCollector(model)
    try:
        with torch.no_grad():
            for index, (images, _) in enumerate(loader):
                if index >= batches:
                    break
                model(images.to(device, non_blocking=True))
    finally:
        collector.close()

    layers = {
        name: _summarize_counts(counts)
        for name, counts in collector.counts.items()
    }
    if not layers:
        raise RuntimeError("No pair-LUT layer states were captured")
    total_pairs = sum(layer["pairs"] for layer in layers.values())
    aggregate = {
        "layers": len(layers),
        "pairs": total_pairs,
        "mean_entropy_bits": sum(
            layer["mean_entropy_bits"] * layer["pairs"]
            for layer in layers.values()
        ) / total_pairs,
        "mean_effective_states": sum(
            layer["mean_effective_states"] * layer["pairs"]
            for layer in layers.values()
        ) / total_pairs,
        "mean_occupied_fraction": sum(
            layer["mean_occupied_fraction"] * layer["pairs"]
            for layer in layers.values()
        ) / total_pairs,
    }
    return {
        "checkpoint": str(checkpoint_path),
        "batches": batches,
        "batch_size": batch_size,
        "aggregate": aggregate,
        "layers": layers,
    }


def to_markdown(result):
    aggregate = result["aggregate"]
    lines = [
        "# Pair-LUT Input-State Analysis",
        "",
        "- Checkpoint: `{}`".format(result["checkpoint"]),
        "- Samples: `{}` batches x `{}`".format(
            result["batches"], result["batch_size"]
        ),
        "- Mean entropy: `{:.4f}/4` bits; effective states: `{:.2f}/16`".format(
            aggregate["mean_entropy_bits"],
            aggregate["mean_effective_states"],
        ),
        "- Observed state fraction: `{:.4f}`".format(
            aggregate["mean_occupied_fraction"]
        ),
        "",
        "## Layers",
    ]
    for name, layer in result["layers"].items():
        lines.append(
            "- `{}`: entropy `{:.4f}` (min `{:.4f}`), effective states "
            "`{:.2f}`, occupied `{:.4f}`, bit ones `{}`".format(
                name,
                layer["mean_entropy_bits"],
                layer["minimum_entropy_bits"],
                layer["mean_effective_states"],
                layer["mean_occupied_fraction"],
                ", ".join(
                    "{:.3f}".format(value)
                    for value in layer["bit_ones_fraction"]
                ),
            )
        )
    return "\n".join(lines) + "\n"


def parse_args(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint")
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--batches", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--device", default=None)
    parser.add_argument("--format", choices=("markdown", "json"), default="markdown")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    logging.disable(logging.CRITICAL)
    device = torch.device(
        args.device
        if args.device is not None
        else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    result = summarize(
        args.checkpoint,
        args.data_dir,
        args.batches,
        args.batch_size,
        device,
    )
    if args.format == "json":
        print(json.dumps(result, indent=2, sort_keys=True))
    else:
        print(to_markdown(result), end="")


if __name__ == "__main__":
    main()

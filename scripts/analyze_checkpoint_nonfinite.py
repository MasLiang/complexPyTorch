#!/usr/bin/env python3
"""Report non-finite tensors in a PyTorch training checkpoint."""

import argparse
import json
from pathlib import Path

import torch


def tensor_summary(name, tensor):
    detached = tensor.detach()
    if detached.is_complex():
        finite = torch.isfinite(detached.real) & torch.isfinite(detached.imag)
        magnitude = detached.abs()
    elif detached.is_floating_point():
        finite = torch.isfinite(detached)
        magnitude = detached.abs()
    else:
        return None
    total = detached.numel()
    finite_count = int(finite.sum().item())
    result = {
        "name": name,
        "shape": list(detached.shape),
        "dtype": str(detached.dtype),
        "nonfinite": total - finite_count,
        "total": total,
    }
    finite_magnitude = magnitude[finite]
    if finite_magnitude.numel():
        result["finite_max_abs"] = float(finite_magnitude.max().item())
    return result


def walk_tensors(value, prefix=""):
    if torch.is_tensor(value):
        yield prefix or "<root>", value
    elif isinstance(value, dict):
        for key, child in value.items():
            child_prefix = "{}.{}".format(prefix, key) if prefix else str(key)
            yield from walk_tensors(child, child_prefix)
    elif isinstance(value, (list, tuple)):
        for index, child in enumerate(value):
            yield from walk_tensors(child, "{}[{}]".format(prefix, index))


def summarize_section(value):
    tensors = []
    nonfinite = []
    for name, tensor in walk_tensors(value):
        summary = tensor_summary(name, tensor)
        if summary is None:
            continue
        tensors.append(summary)
        if summary["nonfinite"]:
            nonfinite.append(summary)
    return {
        "tensor_count": len(tensors),
        "nonfinite_tensor_count": len(nonfinite),
        "nonfinite_element_count": sum(item["nonfinite"] for item in nonfinite),
        "nonfinite_tensors": nonfinite,
    }


def find_section(checkpoint, candidates):
    if not isinstance(checkpoint, dict):
        return None, None
    for name in candidates:
        if name in checkpoint:
            return name, checkpoint[name]
    return None, None


def analyze(path):
    checkpoint = torch.load(path, map_location="cpu")
    model_name, model = find_section(
        checkpoint, ("model_state_dict", "state_dict", "model")
    )
    optimizer_name, optimizer = find_section(
        checkpoint, ("optimizer_state_dict", "optimizer")
    )
    result = {"checkpoint": str(path)}
    if isinstance(checkpoint, dict):
        for name in ("epoch", "phase", "best_acc", "best_test_acc"):
            value = checkpoint.get(name)
            if isinstance(value, (int, float, str, bool)):
                result[name] = value
    if model is None:
        model_name, model = "checkpoint", checkpoint
    result[model_name] = summarize_section(model)
    if optimizer is not None:
        result[optimizer_name] = summarize_section(optimizer)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--output")
    args = parser.parse_args()
    output = json.dumps(analyze(args.checkpoint), indent=2, sort_keys=True) + "\n"
    if args.output:
        path = Path(args.output)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(output, encoding="utf-8")
    else:
        print(output, end="")


if __name__ == "__main__":
    main()

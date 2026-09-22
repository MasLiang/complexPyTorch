#!/usr/bin/env python3
"""Dataset-agnostic LUT4/LUT6 address, phase-bit, and flip analysis."""

import argparse
import sys
import json
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))
from types import SimpleNamespace

import torch
from torch.utils.data import DataLoader

from complexPyTorch.complexLayers import PairLUT4ComplexConv2d
from experiments.flow.checkpoints import checkpoint_state, load_checkpoint
from experiments.san_francisco.analyze_lut_usage import LUTUsageCollector, write_csv


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", required=True, help="Unified-flow seed root containing protocol.json")
    parser.add_argument("--stage", choices=("lut4", "lut6_residual"), required=True)
    parser.add_argument("--checkpoint", help="Default: <run-root>/<stage>/best.pt")
    parser.add_argument("--split", choices=("train", "validation", "test"), default="train")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--max-batches", type=int, default=100, help="0 means complete split")
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output-dir")
    return parser.parse_args(argv)


def _protocol(run_root):
    from experiments.run_flow import PROTOCOLS

    payload = json.loads((Path(run_root) / "protocol.json").read_text(encoding="utf-8"))
    args = SimpleNamespace(**payload["args"])
    return PROTOCOLS[payload["dataset"]](), args, payload


def _base_table(module):
    logits = module.dominance_base
    choices = logits.argmax(dim=-1)
    codebook = module.output_codebook.to(device=logits.device, dtype=logits.dtype)
    table = codebook[choices].permute(3, 0, 1, 2).reshape(2 * module.out_channels, module.group_num, 16)
    return module._expand_lut4_for_lut6(table)


def hard_flip_rates(model):
    result = {}
    for name, module in model.named_modules():
        if not isinstance(module, PairLUT4ComplexConv2d) or module.lut_inputs != 6:
            continue
        final = module.materialize_table().detach()
        base = _base_table(module).detach()
        result[name] = {
            "hard_output_flip_rate": float((final != base).float().mean().item()),
            "hard_output_flip_count": int((final != base).sum().item()),
            "hard_output_entries": int(final.numel()),
        }
    return result


def main(argv=None):
    args = parse_args(argv)
    root = Path(args.run_root)
    output = Path(args.output_dir) if args.output_dir else root / args.stage / "lut_usage"
    output.mkdir(parents=True, exist_ok=True)
    protocol, stored_args, metadata = _protocol(root)
    stage = next(spec for spec in protocol.stages(stored_args) if spec.name == args.stage)
    device = torch.device(args.device if args.device != "cuda" or torch.cuda.is_available() else "cpu")
    checkpoint_path = Path(args.checkpoint) if args.checkpoint else root / args.stage / "best.pt"
    checkpoint = load_checkpoint(checkpoint_path, device=device)
    model = protocol.build_model(stage, stored_args, int(metadata.get("num_classes", len(metadata.get("class_names", metadata.get("classes", []))))))
    if not metadata.get("num_classes", metadata.get("class_names", metadata.get("classes"))):
        raise RuntimeError("Could not infer class count from protocol metadata")
    model = model.to(device).eval()
    model.load_state_dict(checkpoint_state(checkpoint), strict=True)
    bundle = protocol.build_bundle(stored_args)
    dataset = getattr(bundle, args.split)
    collector = LUTUsageCollector(model)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=device.type == "cuda")
    samples = batches = 0
    try:
        with torch.no_grad():
            for index, (inputs, _) in enumerate(loader):
                if args.max_batches and index >= args.max_batches:
                    break
                model(inputs.to(device, non_blocking=True))
                samples += inputs.size(0)
                batches += 1
    finally:
        collector.close()
    layers = collector.report()
    flips = hard_flip_rates(model)
    for layer in layers:
        if layer["layer"] in flips:
            layer.update(flips[layer["layer"]])
    report = {"dataset": protocol.name, "checkpoint": str(checkpoint_path), "stage": args.stage, "split": args.split, "samples": samples, "batches": batches, "layers": layers}
    (output / "lut_usage.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    write_csv(output / "lut_usage.csv", layers)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

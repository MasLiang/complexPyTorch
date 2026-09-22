#!/usr/bin/env python3
"""Verify that exported hard LUT truth tables exactly reproduce LUT-stage inference."""

import argparse
import sys
import json
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))
from types import MethodType, SimpleNamespace

import torch
from torch.utils.data import DataLoader

from complexPyTorch.complexLayers import PairLUT4ComplexConv2d
from experiments.flow.checkpoints import checkpoint_state, load_checkpoint


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", required=True)
    parser.add_argument("--stage", default="lut6_residual", choices=("lut4", "lut6_residual"))
    parser.add_argument("--checkpoint")
    parser.add_argument("--split", default="test", choices=("train", "validation", "test"))
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--max-batches", type=int, default=0, help="0 validates the complete split")
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args(argv)


def _load_protocol(root):
    from experiments.run_flow import PROTOCOLS

    payload = json.loads((root / "protocol.json").read_text(encoding="utf-8"))
    return PROTOCOLS[payload["dataset"]](), SimpleNamespace(**payload["args"]), payload


def _export_tables(model):
    return {name: module.materialize_table().detach().cpu() for name, module in model.named_modules() if isinstance(module, PairLUT4ComplexConv2d)}


def _install_tables(model, tables):
    for name, module in model.named_modules():
        if not isinstance(module, PairLUT4ComplexConv2d):
            continue
        table = tables[name]
        def materialize(self, table=table):
            return table.to(device=self.weight.device, dtype=self.weight.dtype)
        module.materialize_table = MethodType(materialize, module)


def main(argv=None):
    args = parse_args(argv)
    root = Path(args.run_root)
    protocol, stored_args, metadata = _load_protocol(root)
    spec = next(item for item in protocol.stages(stored_args) if item.name == args.stage)
    classes = int(metadata.get("num_classes", len(metadata.get("class_names", metadata.get("classes", [])))))
    if not classes:
        raise RuntimeError("Protocol metadata has no class names")
    device = torch.device(args.device if args.device != "cuda" or torch.cuda.is_available() else "cpu")
    checkpoint_path = Path(args.checkpoint) if args.checkpoint else root / args.stage / "best.pt"
    checkpoint = load_checkpoint(checkpoint_path, device=device)
    source = protocol.build_model(spec, stored_args, classes).to(device).eval()
    source.load_state_dict(checkpoint_state(checkpoint), strict=True)
    tables = _export_tables(source)
    torch.save({"stage": args.stage, "tables": tables}, root / args.stage / "hard_lut_tables.pt")
    exported = protocol.build_model(spec, stored_args, classes).to(device).eval()
    exported.load_state_dict(checkpoint_state(checkpoint), strict=True)
    _install_tables(exported, tables)
    bundle = protocol.build_bundle(stored_args)
    loader = DataLoader(getattr(bundle, args.split), batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=device.type == "cuda")
    source_outputs, hard_outputs = {}, {}
    source_hooks = []
    hard_hooks = []
    for name, module in source.named_modules():
        if isinstance(module, PairLUT4ComplexConv2d):
            source_hooks.append(module.register_forward_hook(lambda _, __, output, name=name: source_outputs.__setitem__(name, output.detach())))
    for name, module in exported.named_modules():
        if isinstance(module, PairLUT4ComplexConv2d):
            hard_hooks.append(module.register_forward_hook(lambda _, __, output, name=name: hard_outputs.__setitem__(name, output.detach())))
    max_logit_difference = 0.0
    max_lut_output_difference = 0.0
    prediction_mismatches = 0
    lut_output_mismatches = 0
    samples = batches = 0
    try:
        with torch.no_grad():
            for index, (inputs, _) in enumerate(loader):
                if args.max_batches and index >= args.max_batches:
                    break
                inputs = inputs.to(device, non_blocking=True)
                source_outputs.clear()
                hard_outputs.clear()
                reference = source(inputs)
                hard = exported(inputs)
                max_logit_difference = max(max_logit_difference, float((reference - hard).abs().max().item()))
                prediction_mismatches += int((reference.argmax(dim=1) != hard.argmax(dim=1)).sum().item())
                if source_outputs.keys() != hard_outputs.keys():
                    raise RuntimeError("LUT layer sets differ after hard export")
                for name in source_outputs:
                    difference = (source_outputs[name] - hard_outputs[name]).abs()
                    max_lut_output_difference = max(max_lut_output_difference, float(difference.max().item()))
                    lut_output_mismatches += int((difference != 0).sum().item())
                samples += inputs.size(0)
                batches += 1
    finally:
        for handle in source_hooks + hard_hooks:
            handle.remove()
    report = {"dataset": protocol.name, "stage": args.stage, "checkpoint": str(checkpoint_path), "split": args.split, "samples": samples, "batches": batches, "max_lut_output_difference": max_lut_output_difference, "lut_output_mismatches": lut_output_mismatches, "max_logit_difference": max_logit_difference, "prediction_mismatches": prediction_mismatches, "expected_exact": True}
    (root / args.stage / "hard_lut_validation.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    if max_lut_output_difference != 0.0 or lut_output_mismatches != 0 or max_logit_difference != 0.0 or prediction_mismatches != 0:
        raise RuntimeError("Hard truth-table inference differs from training-model inference: {}".format(report))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Compare standard and hardware Phase2.7 forwards on a complete test set."""

import argparse
import json
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader


ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from complexPyTorch.dualLut6Flow import (  # noqa: E402
    DUAL_LUT6_PHASE,
    DualLUT6ComplexResNet,
    iter_two_bit_activations,
    set_dual_lut6_implementation,
)
from training import build_datasets, set_seed  # noqa: E402


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--datadir", type=Path)
    parser.add_argument("--batch-size", default=128, type=int)
    parser.add_argument("--num-workers", default=4, type=int)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--activation-diagnostic-batches", default=4, type=int)
    parser.add_argument("--output", type=Path)
    return parser.parse_args(argv)


def strip_wrappers(key):
    changed = True
    while changed:
        changed = False
        for prefix in ("_orig_mod.", "module."):
            if key.startswith(prefix):
                key = key[len(prefix):]
                changed = True
    return key


def checkpoint_phase(checkpoint):
    for container in (
        checkpoint,
        checkpoint.get("metrics", {}),
        checkpoint.get("args", {}),
    ):
        if isinstance(container, dict) and container.get("phase") is not None:
            return float(container["phase"])
    return None


def clean_state_dict(state):
    return {strip_wrappers(key): value for key, value in state.items()}


def infer_num_classes(state):
    for key, value in state.items():
        if strip_wrappers(key) == "fc.weight":
            return int(value.shape[0])
    raise RuntimeError("Checkpoint has no fc.weight")


def build_model(checkpoint):
    stored_args = checkpoint.get("args", {})
    state = checkpoint.get("model", checkpoint.get("state_dict", checkpoint))
    model = DualLUT6ComplexResNet(
        num_blocks=int(stored_args.get("num_blocks", 3)),
        start_filters=int(stored_args.get("start_filter", 11)),
        num_classes=infer_num_classes(state),
        spectral_pool_scheme=stored_args.get(
            "spectral_pool_scheme", "none"
        ),
        spectral_pool_gamma=float(
            stored_args.get("spectral_pool_gamma", 0.5)
        ),
        is_sar_input=False,
        threshold_init=float(stored_args.get("threshold_init", 0.675)),
        high_ratio=float(stored_args.get("high_ratio", 3.0)),
        magnitude_beta=float(stored_args.get("magnitude_beta", 4.0)),
    )
    missing, unexpected = model.load_state_dict(
        clean_state_dict(state),
        strict=False,
    )
    if missing or unexpected:
        raise RuntimeError(
            "Checkpoint load mismatch: missing={}, unexpected={}".format(
                missing,
                unexpected,
            )
        )
    return model


def build_test_loader(checkpoint, args):
    stored_args = checkpoint.get("args", {})
    datadir = (
        args.datadir
        if args.datadir is not None
        else Path(stored_args.get("datadir", ROOT_DIR / "data"))
    )
    dataset_args = SimpleNamespace(
        dataset=stored_args.get("dataset", "cifar10"),
        datadir=str(datadir),
        no_validation=bool(stored_args.get("no_validation", False)),
        batch_size=args.batch_size,
        seed=int(stored_args.get("seed", 0xE4223644E98B8E64)),
    )
    _, _, test_set, _, _, _ = build_datasets(
        dataset_args,
        train_logger=None,
        ddp_enabled=False,
        is_main=True,
    )
    return DataLoader(
        test_set,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=args.device.startswith("cuda"),
        persistent_workers=args.num_workers > 0,
    )


class ActivationCodeComparator:
    def __init__(self, model, max_batches):
        self.max_batches = max(int(max_batches), 0)
        self.batch_index = 0
        self.mode = None
        self.reference = {}
        self.stats = {}
        self.handles = []
        for name, module in model.named_modules():
            if any(module is item for item in iter_two_bit_activations(model)):
                self.stats[name] = {"different": 0, "total": 0}
                self.handles.append(
                    module.register_forward_hook(self._make_hook(name))
                )

    def _make_hook(self, name):
        def hook(module, inputs, output):
            del output
            if self.batch_index >= self.max_batches:
                return
            real_bits, imag_bits = module.hard_code_bits(inputs[0])
            packed = (
                (real_bits[..., 0] << 3)
                | (real_bits[..., 1] << 2)
                | (imag_bits[..., 0] << 1)
                | imag_bits[..., 1]
            ).to(torch.uint8).detach().cpu()
            if self.mode == "standard":
                self.reference[name] = packed
            elif self.mode == "hardware":
                expected = self.reference.pop(name)
                self.stats[name]["different"] += int(
                    (packed != expected).sum().item()
                )
                self.stats[name]["total"] += packed.numel()
            else:
                raise RuntimeError("Activation comparator mode is unset")

        return hook

    def begin_standard(self, batch_index):
        self.batch_index = batch_index
        self.mode = "standard"
        self.reference.clear()

    def begin_hardware(self, batch_index):
        self.batch_index = batch_index
        self.mode = "hardware"

    def end_batch(self):
        if self.reference:
            raise RuntimeError(
                "Unmatched activation captures: {}".format(
                    sorted(self.reference)
                )
            )

    def finish(self):
        for handle in self.handles:
            handle.remove()
        result = {}
        for name, values in self.stats.items():
            total = values["total"]
            result[name] = {
                **values,
                "different_fraction": (
                    float(values["different"]) / total if total else 0.0
                ),
            }
        return result


@torch.no_grad()
def verify(model, loader, device, activation_diagnostic_batches):
    model.eval()
    comparator = ActivationCodeComparator(
        model,
        activation_diagnostic_batches,
    )
    result = {
        "samples": 0,
        "standard_correct": 0,
        "hardware_correct": 0,
        "prediction_mismatches": 0,
        "standard_only_correct": 0,
        "hardware_only_correct": 0,
        "standard_loss_sum": 0.0,
        "hardware_loss_sum": 0.0,
        "absolute_logit_difference_sum": 0.0,
        "logit_elements": 0,
        "max_abs_logit_difference": 0.0,
    }
    start = time.time()
    for batch_index, (data, target) in enumerate(loader):
        data = data.to(device)
        target = target.to(device)

        comparator.begin_standard(batch_index)
        set_dual_lut6_implementation(model, "standard")
        standard = model(data)

        comparator.begin_hardware(batch_index)
        set_dual_lut6_implementation(model, "hardware")
        hardware = model(data)
        comparator.end_batch()

        standard_prediction = standard.argmax(dim=1)
        hardware_prediction = hardware.argmax(dim=1)
        standard_correct = standard_prediction == target
        hardware_correct = hardware_prediction == target
        difference = (standard - hardware).abs()

        result["samples"] += data.size(0)
        result["standard_correct"] += int(standard_correct.sum().item())
        result["hardware_correct"] += int(hardware_correct.sum().item())
        result["prediction_mismatches"] += int(
            (standard_prediction != hardware_prediction).sum().item()
        )
        result["standard_only_correct"] += int(
            (standard_correct & ~hardware_correct).sum().item()
        )
        result["hardware_only_correct"] += int(
            (~standard_correct & hardware_correct).sum().item()
        )
        result["standard_loss_sum"] += float(
            F.cross_entropy(standard, target, reduction="sum").item()
        )
        result["hardware_loss_sum"] += float(
            F.cross_entropy(hardware, target, reduction="sum").item()
        )
        result["absolute_logit_difference_sum"] += float(
            difference.sum().item()
        )
        result["logit_elements"] += difference.numel()
        result["max_abs_logit_difference"] = max(
            result["max_abs_logit_difference"],
            float(difference.max().item()),
        )

    set_dual_lut6_implementation(model, "standard")
    samples = result["samples"]
    result.update(
        {
            "standard_accuracy": result["standard_correct"] / samples,
            "hardware_accuracy": result["hardware_correct"] / samples,
            "prediction_mismatch_fraction": (
                result["prediction_mismatches"] / samples
            ),
            "standard_loss": result["standard_loss_sum"] / samples,
            "hardware_loss": result["hardware_loss_sum"] / samples,
            "mean_abs_logit_difference": (
                result["absolute_logit_difference_sum"]
                / result["logit_elements"]
            ),
            "elapsed_seconds": time.time() - start,
            "activation_code_differences": comparator.finish(),
            "activation_diagnostic_batches": min(
                activation_diagnostic_batches,
                len(loader),
            ),
        }
    )
    for key in (
        "standard_correct",
        "hardware_correct",
        "standard_loss_sum",
        "hardware_loss_sum",
        "absolute_logit_difference_sum",
        "logit_elements",
    ):
        result.pop(key)
    return result


def main(argv=None):
    args = parse_args(argv)
    checkpoint_path = args.checkpoint.resolve()
    if not checkpoint_path.is_file():
        raise FileNotFoundError(
            "Checkpoint not found: {}".format(checkpoint_path)
        )
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    phase = checkpoint_phase(checkpoint)
    if phase != DUAL_LUT6_PHASE:
        raise ValueError(
            "Expected Phase {}, found Phase {}".format(
                DUAL_LUT6_PHASE,
                phase,
            )
        )
    if not checkpoint.get("hardware_deployable", False):
        raise ValueError("Checkpoint was saved before rho reached one")

    stored_args = checkpoint.get("args", {})
    set_seed(int(stored_args.get("seed", 0xE4223644E98B8E64)))
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    model = build_model(checkpoint).to(device)
    loader = build_test_loader(checkpoint, args)
    result = verify(
        model,
        loader,
        device,
        args.activation_diagnostic_batches,
    )
    result.update(
        {
            "checkpoint": str(checkpoint_path),
            "checkpoint_epoch": checkpoint.get("epoch"),
            "device": str(device),
        }
    )
    output = (
        args.output.resolve()
        if args.output is not None
        else checkpoint_path.parents[1]
        / "dual_lut6_hardware_verification.json"
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2, sort_keys=True)
    print("Wrote {}".format(output))
    print(
        "standard_acc={:.4f} hardware_acc={:.4f} mismatches={}/{} "
        "max_diff={:.6e} mean_diff={:.6e}".format(
            result["standard_accuracy"],
            result["hardware_accuracy"],
            result["prediction_mismatches"],
            result["samples"],
            result["max_abs_logit_difference"],
            result["mean_abs_logit_difference"],
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

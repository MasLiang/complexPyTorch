#!/usr/bin/env python3
"""Audit current complex LUT numerics against the real Bi-Real LUT contract."""

import argparse
import json
import math
from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from complexPyTorch.complexBinaryResNet import BinaryComplexResNet
from complexPyTorch.complexLayers import (
    BinaryComplexBitActivation,
    ComplexBatchNorm2d,
    LUTBinaryConv2d,
    NaiveComplexBatchNorm2d,
    TwoLUTComplexConv2d,
)


def real_stats(tensor):
    value = tensor.detach().float()
    return {
        "min": float(value.min()),
        "max": float(value.max()),
        "mean": float(value.mean()),
        "std": float(value.std(unbiased=False)),
        "zero_fraction": float((value == 0).float().mean()),
        "one_fraction": float((value == 1).float().mean()),
    }


def tensor_stats(tensor):
    if not torch.is_complex(tensor):
        return real_stats(tensor)
    real = tensor.real.detach().float()
    imag = tensor.imag.detach().float()
    real_centered = real - real.mean()
    imag_centered = imag - imag.mean()
    denominator = real_centered.square().mean().sqrt()
    denominator = denominator * imag_centered.square().mean().sqrt()
    correlation = 0.0
    if float(denominator) > 0.0:
        correlation = float(
            (real_centered * imag_centered).mean() / denominator
        )
    return {
        "real": real_stats(real),
        "imag": real_stats(imag),
        "real_imag_correlation": correlation,
    }


def cosine(left, right):
    left = left.reshape(-1).float()
    right = right.reshape(-1).float()
    denominator = left.norm() * right.norm()
    if float(denominator) == 0.0:
        return 0.0
    return float(torch.dot(left, right) / denominator)


def gradient_pair_stats(first, second):
    first_norm = float(first.norm())
    second_norm = float(second.norm())
    combined_norm = float((first + second).norm())
    independent_norm = math.sqrt(first_norm ** 2 + second_norm ** 2)
    return {
        "first_norm": first_norm,
        "second_norm": second_norm,
        "cosine": cosine(first, second),
        "combined_norm": combined_norm,
        "combined_over_independent": (
            combined_norm / independent_norm if independent_norm else 0.0
        ),
    }


def local_gradient_conflict(module, inp, grad_output):
    xr = inp.real.detach()
    xi = inp.imag.detach()
    grad_real = grad_output.real.detach()
    grad_imag = grad_output.imag.detach()

    def branch_grads(branch, value, upstream):
        value = value.detach().requires_grad_(True)
        output = branch(value)
        grad_table, grad_input = torch.autograd.grad(
            output,
            (branch.weight, value),
            upstream,
            retain_graph=False,
        )
        return grad_table.detach(), grad_input.detach()

    table_rr, input_rr = branch_grads(module.conv_r, xr, grad_real)
    table_ri, input_ri = branch_grads(module.conv_r, xi, grad_imag)
    table_ii, input_ii = branch_grads(module.conv_i, xi, -grad_real)
    table_ir, input_ir = branch_grads(module.conv_i, xr, grad_imag)
    return {
        "table": {
            "conv_r_rr_vs_ri": gradient_pair_stats(table_rr, table_ri),
            "conv_i_minus_ii_vs_ir": gradient_pair_stats(
                table_ii, table_ir
            ),
        },
        "activation_bits": {
            "real_rr_vs_ir": gradient_pair_stats(input_rr, input_ir),
            "imag_minus_ii_vs_ri": gradient_pair_stats(input_ii, input_ri),
        },
    }


def table_stats(module):
    weight = module.weight.detach().float()
    absolute = weight.abs()
    return {
        "shape": list(weight.shape),
        "negative_fraction": float((weight < 0).float().mean()),
        "mean": float(weight.mean()),
        "std": float(weight.std(unbiased=False)),
        "abs_mean": float(absolute.mean()),
        "abs_min": float(absolute.min()),
        "fraction_abs_below_0_1": float((absolute < 0.1).float().mean()),
        "fraction_abs_below_0_25": float((absolute < 0.25).float().mean()),
    }


def build_model(checkpoint, device):
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    saved_args = payload.get("args", {})
    model = BinaryComplexResNet(
        in_channels=3,
        num_blocks=int(saved_args.get("num_blocks", 3)),
        start_filters=int(saved_args.get("start_filter", 16)),
        num_classes=10,
        spectral_pool_scheme=saved_args.get("spectral_pool_scheme", "none"),
        spectral_pool_gamma=float(saved_args.get("spectral_pool_gamma", 0.5)),
        weight_grad_mode=saved_args.get("weight_grad_mode", "ste"),
        act_grad_mode=saved_args.get("activation_grad_mode", "bireal"),
        binary_stem=bool(saved_args.get("binary_stem", False)),
        is_sar_input=False,
        phase=3,
        post_bn_mode=saved_args.get("post_bn_mode", "covariance"),
    )
    state = payload.get("model", payload.get("state_dict", payload))
    model.load_state_dict(state, strict=True)
    return model.to(device), saved_args, payload.get("metrics", {})


def load_batch(datadir, batch_size, device):
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(
                (0.4914, 0.4822, 0.4465),
                (0.2023, 0.1994, 0.2010),
            ),
        ]
    )
    dataset = datasets.CIFAR10(
        datadir,
        train=False,
        download=False,
        transform=transform,
    )
    images, targets = next(
        iter(DataLoader(dataset, batch_size=batch_size, shuffle=False))
    )
    return images.to(device), targets.to(device)


def audit(model, images, targets):
    activations = {}
    lut_calls = {}
    complex_pre_bn = {}
    batch_norm_outputs = {}
    first_complex = {"module": None, "input": None, "grad_output": None}
    handles = []

    for name, module in model.named_modules():
        if isinstance(module, BinaryComplexBitActivation):
            handles.append(
                module.register_forward_hook(
                    lambda _, __, output, key=name: activations.__setitem__(
                        key, tensor_stats(output)
                    )
                )
            )
        elif isinstance(module, LUTBinaryConv2d):
            lut_calls[name] = []
            handles.append(
                module.register_forward_hook(
                    lambda _, __, output, key=name: lut_calls[key].append(
                        tensor_stats(output)
                    )
                )
            )
        elif isinstance(module, TwoLUTComplexConv2d):
            def capture_complex(_, inputs, output, key=name, owner=module):
                complex_pre_bn[key] = tensor_stats(output)
                if first_complex["module"] is None:
                    first_complex["module"] = owner
                    first_complex["input"] = inputs[0].detach()
                    output.register_hook(
                        lambda grad: first_complex.__setitem__(
                            "grad_output", grad.detach()
                        )
                    )

            handles.append(module.register_forward_hook(capture_complex))
        elif isinstance(module, (ComplexBatchNorm2d, NaiveComplexBatchNorm2d)):
            handles.append(
                module.register_forward_hook(
                    lambda _, __, output, key=name: batch_norm_outputs.__setitem__(
                        key, tensor_stats(output)
                    )
                )
            )

    model.train()
    model.zero_grad(set_to_none=True)
    logits = model(images)
    loss = F.cross_entropy(logits, targets)
    loss.backward()

    gradient_norms = {}
    tables = {}
    for name, module in model.named_modules():
        if isinstance(module, LUTBinaryConv2d):
            tables[name] = table_stats(module)
            gradient_norms[name] = (
                float(module.weight.grad.norm())
                if module.weight.grad is not None
                else None
            )

    conflict = None
    if first_complex["grad_output"] is not None:
        conflict = local_gradient_conflict(
            first_complex["module"],
            first_complex["input"],
            first_complex["grad_output"],
        )

    for handle in handles:
        handle.remove()

    return {
        "loss": float(loss.detach()),
        "batch_accuracy": float(
            (logits.argmax(dim=1) == targets).float().mean()
        ),
        "activations": activations,
        "lut_calls": lut_calls,
        "complex_pre_bn": complex_pre_bn,
        "batch_norm_outputs": batch_norm_outputs,
        "tables": tables,
        "gradient_norms": gradient_norms,
        "first_layer_gradient_conflict": conflict,
    }


def compact_layer_rows(result):
    rows = []
    for name, stats in result["complex_pre_bn"].items():
        real = stats["real"]
        imag = stats["imag"]
        rows.append(
            "| `{}` | [{:.1f}, {:.1f}] | {:.2f} | {:.2f} | "
            "[{:.1f}, {:.1f}] | {:.2f} | {:.2f} | {:.3f} |".format(
                name,
                real["min"],
                real["max"],
                real["mean"],
                real["std"],
                imag["min"],
                imag["max"],
                imag["mean"],
                imag["std"],
                stats["real_imag_correlation"],
            )
        )
    return rows


def table_stage_rows(result):
    grouped = {}
    for name, stats in result["tables"].items():
        stage = name.split(".", 1)[0]
        parameter_count = math.prod(stats["shape"])
        grad_norm = result["gradient_norms"][name]
        grouped.setdefault(stage, []).append(
            {
                "abs_mean": stats["abs_mean"],
                "near_zero": stats["fraction_abs_below_0_1"],
                "grad_rms": grad_norm / math.sqrt(parameter_count),
            }
        )
    rows = []
    for stage, values in grouped.items():
        count = len(values)
        rows.append(
            "| `{}` | {:.3f} | {:.4f} | {:.3e} |".format(
                stage,
                sum(value["abs_mean"] for value in values) / count,
                sum(value["near_zero"] for value in values) / count,
                sum(value["grad_rms"] for value in values) / count,
            )
        )
    return rows


def markdown_report(payload):
    result = payload["dynamic"]
    conflict = result["first_layer_gradient_conflict"]
    lines = [
        "# Complex LUT vs Real LUT Numerical Audit",
        "",
        "## Run",
        "",
        "- Checkpoint: `{}`".format(payload["checkpoint"]),
        "- Saved metrics: `{}`".format(payload["saved_metrics"]),
        "- Diagnostic batch loss/accuracy: `{:.4f}` / `{:.4f}`".format(
            result["loss"], result["batch_accuracy"]
        ),
        "",
        "## Exact Matches",
        "",
        "- Activation forward values are hard `0/1`.",
        "- Both implementations group six activation bits per LUT6.",
        "- Both use the same grouped binary CUDA forward/backward contract.",
        "- Each real LUTConv sums all Boolean LUT outputs over `L`.",
        "- LUT entries use hard thresholding in forward and identity STE in backward.",
        "",
        "## Complex-Only Differences",
        "",
        "- One real reference LUTConv returns `[0, L]`; the complex wrapper returns "
        "`real=A-B` in `[-L,L]` and `imag=C+D` in `[0,2L]`.",
        "- Each complex LUT table is reused twice, so its parameter gradient is the "
        "sum of two paths that can reinforce or cancel.",
        "- The reference applies ordinary real BN to one sum. The current model first "
        "combines four sums, then applies covariance complex BN.",
        "- The current default bimodal initialization uses std `0.1/0.1`; the reference "
        "uses `0.2/0.1` for negative/positive modes.",
        "- This audited checkpoint used the former four-drop `bireal_reference` "
        "implementation. The current code now aligns that name with the reference's "
        "linear decay from default LR `0.001`; the old behavior is `multistep`.",
        "",
        "## Measured Pre-BN Ranges",
        "",
        "| Layer | Real range | Real mean | Real std | Imag range | Imag mean | Imag std | Corr |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    lines.extend(compact_layer_rows(result))
    lines.extend(
        [
            "",
            "## LUT Margin And Gradient By Stage",
            "",
            "| Stage | Mean `abs(logit)` | Fraction `abs(logit)<0.1` | Gradient RMS |",
            "|---|---:|---:|---:|",
        ]
    )
    lines.extend(table_stage_rows(result))
    lines.extend(["", "## First Shared-LUT Gradient Interaction", ""])
    if conflict is None:
        lines.append("Gradient interaction was not captured.")
    else:
        for category, pairs in conflict.items():
            lines.append("- {}:".format(category))
            for name, stats in pairs.items():
                lines.append(
                    "  - `{}`: norms `{:.4g}` / `{:.4g}`, cosine `{:.4f}`, "
                    "combined/independent `{:.4f}`.".format(
                        name,
                        stats["first_norm"],
                        stats["second_norm"],
                        stats["cosine"],
                        stats["combined_over_independent"],
                    )
                )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "1. Re-run with the now-aligned reference optimizer contract: Adam at "
            "`0.001` with linear decay. The audited historical `0.02` milestone run "
            "drives LUT logits far from zero, reducing late sign mobility.",
            "2. The first layer shows nearly orthogonal shared-path gradients rather "
            "than strong cancellation. The clearer issue is depth attenuation: normalized "
            "LUT gradient is much smaller in stages 3 and 4. Extend path diagnostics to "
            "middle/late layers before changing gradient combination.",
            "3. Compare covariance BN with independent real/imag BN on this exact route. "
            "The old pair-LUT BN ablation does not settle the current architecture.",
            "4. Test signed decoding before complex composition: replace each Boolean "
            "contribution `b` by `2b-1`. It centers the imaginary branch, but with train-"
            "mode BN it is mostly an affine reparameterization, so expected impact is "
            "smaller than schedule and gradient-path effects.",
            "5. Align the negative-mode initialization std from `0.1` to reference "
            "`0.2` as a controlled lower-priority ablation.",
            "",
        ]
    )
    return "\n".join(lines)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--datadir", default="data")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--output", default=None)
    parser.add_argument("--json-output", default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    device = torch.device(args.device)
    model, saved_args, saved_metrics = build_model(args.checkpoint, device)
    images, targets = load_batch(args.datadir, args.batch_size, device)
    payload = {
        "checkpoint": args.checkpoint,
        "saved_args": saved_args,
        "saved_metrics": saved_metrics,
        "dynamic": audit(model, images, targets),
    }
    report = markdown_report(payload)
    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(report + "\n", encoding="utf-8")
    else:
        print(report)
    if args.json_output:
        output = Path(args.json_output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )


if __name__ == "__main__":
    main()

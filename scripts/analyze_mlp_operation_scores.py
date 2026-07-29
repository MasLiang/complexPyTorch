#!/usr/bin/env python3
"""Inspect 32-state local-operation scores in MLP-flow checkpoints."""

import argparse
import json
from pathlib import Path

import torch


def parse_args():
    parser = argparse.ArgumentParser(
        description="Analyze the 32-state scores learned by MLP flow operators"
    )
    parser.add_argument("checkpoints", nargs="+")
    parser.add_argument("--output-prefix", required=True)
    return parser.parse_args()


def strip_wrappers(key):
    changed = True
    while changed:
        changed = False
        for prefix in ("_orig_mod.", "module."):
            if key.startswith(prefix):
                key = key[len(prefix):]
                changed = True
    return key


def physical_states(dtype=torch.float32):
    indices = torch.arange(32, dtype=torch.long)
    shifts = torch.arange(4, -1, -1, dtype=torch.long)
    states = ((indices[:, None] >> shifts[None, :]) & 1).to(dtype)
    return states * 2.0 - 1.0


def exact_complex_scores(states):
    xr, xi, _, wr, wi = states.unbind(dim=-1)
    return torch.stack(
        (xr * wr - xi * wi, xr * wi + xi * wr),
        dim=-1,
    )


def state_bases(dtype):
    states = physical_states(dtype)
    xr, xi, dominance, wr, wi = states.unbind(dim=-1)
    activation = torch.stack(
        (
            torch.ones_like(xr),
            xr,
            xi,
            dominance,
            xr * xi,
            xr * dominance,
            xi * dominance,
            xr * xi * dominance,
        ),
        dim=-1,
    )
    weight = torch.stack(
        (torch.ones_like(wr), wr, wi, wr * wi),
        dim=-1,
    )
    return states, activation, weight


def enumerate_polynomial_scores(coefficients):
    states, activation, weight = state_bases(coefficients.dtype)
    scores = torch.einsum(
        "sopq,np,nq->sno",
        coefficients,
        activation,
        weight,
    )
    return states, scores


def enumerate_binary_mlp_scores(hidden_weight, output_weight, threshold, scale):
    states = physical_states(hidden_weight.dtype)
    hidden_sign = hidden_weight.sign()
    hidden_pre = (
        torch.einsum("shd,nd->snh", hidden_sign, states)
        + threshold.unsqueeze(1)
    )
    hidden_bits = (hidden_pre >= 0.0).to(hidden_weight.dtype)
    output_sign = output_weight.sign()
    scores = float(scale.reshape(-1)[0].item()) * torch.einsum(
        "soh,snh->sno",
        output_sign,
        hidden_bits,
    )
    return states, scores


def scalar_stats(values):
    values = values.detach().to(torch.float64).reshape(-1)
    if not values.numel():
        return {"count": 0}
    quantiles = torch.quantile(
        values,
        torch.tensor(
            [0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0],
            dtype=values.dtype,
        ),
    )
    return {
        "count": int(values.numel()),
        "mean": float(values.mean().item()),
        "q0": float(quantiles[0].item()),
        "q10": float(quantiles[1].item()),
        "q25": float(quantiles[2].item()),
        "q50": float(quantiles[3].item()),
        "q75": float(quantiles[4].item()),
        "q90": float(quantiles[5].item()),
        "q100": float(quantiles[6].item()),
    }


def analyze_scores(scores, states, representation):
    scores = scores.detach().cpu()
    initial_scores = exact_complex_scores(states).unsqueeze(0).expand_as(scores)
    tie_mask = initial_scores.abs() < 1e-8
    non_tie_mask = ~tie_mask
    hard = scores >= 0.0
    initial_hard = initial_scores >= 0.0
    changed = hard != initial_hard

    d_low = torch.nonzero(states[:, 2] < 0.0, as_tuple=False).reshape(-1)
    d_high = d_low | 0b00100
    hard_d_diff = hard[:, d_low, :] != hard[:, d_high, :]
    score_d_diff = scores[:, d_high, :] - scores[:, d_low, :]

    tie_abs = scores[tie_mask].abs()
    non_tie_abs = scores[non_tie_mask].abs()
    nonzero_tie_abs = tie_abs[tie_abs > 1e-12]
    amplification = (
        2.0 / nonzero_tie_abs
        if nonzero_tie_abs.numel()
        else torch.empty(0, dtype=scores.dtype)
    )
    total = int(scores.numel())
    tie_total = int(tie_mask.sum().item())
    non_tie_total = int(non_tie_mask.sum().item())

    metrics = {
        "representation": representation,
        "sets": int(scores.shape[0]),
        "entries": total,
        "initial_ties": tie_total,
        "initial_tie_ratio": tie_total / float(total),
        "changed": int(changed.sum().item()),
        "changed_ratio": float(changed.float().mean().item()),
        "changed_ties": int((changed & tie_mask).sum().item()),
        "changed_tie_ratio": float(
            (changed & tie_mask).sum().item() / float(tie_total)
        ),
        "changed_non_ties": int((changed & non_tie_mask).sum().item()),
        "changed_non_tie_ratio": float(
            (changed & non_tie_mask).sum().item() / float(non_tie_total)
        ),
        "d_sensitive": int(hard_d_diff.sum().item()),
        "d_pairs": int(hard_d_diff.numel()),
        "d_sensitive_ratio": float(hard_d_diff.float().mean().item()),
        "zero_score_ratio": float((scores.abs() < 1e-8).float().mean().item()),
        "all_abs_scores": scalar_stats(scores.abs()),
        "tie_abs_scores": scalar_stats(tie_abs),
        "non_tie_abs_scores": scalar_stats(non_tie_abs),
        "hard_amplification_on_ties": scalar_stats(amplification),
        "d_score_abs_difference": scalar_stats(score_d_diff.abs()),
    }
    raw = {
        "tie_abs": tie_abs,
        "non_tie_abs": non_tie_abs,
        "amplification": amplification,
    }
    return metrics, raw


def clean_checkpoint_state(checkpoint):
    source = checkpoint.get("model", checkpoint.get("state_dict", checkpoint))
    return {strip_wrappers(key): value for key, value in source.items()}


def extract_layers(checkpoint):
    state = clean_checkpoint_state(checkpoint)
    phase = float(checkpoint.get("flow_phase", checkpoint.get("phase", -1)))
    layers = []

    polynomial_keys = sorted(
        key for key in state if key.endswith("operation_coefficients")
    )
    for key in polynomial_keys:
        coefficients = state[key].detach().cpu()
        states, scores = enumerate_polynomial_scores(coefficients)
        metrics, raw = analyze_scores(scores, states, "floating_polynomial")
        layers.append((key, metrics, raw))

    hidden_suffix = "binary_mlp_hidden_weight"
    hidden_keys = sorted(key for key in state if key.endswith(hidden_suffix))
    for hidden_key in hidden_keys:
        prefix = hidden_key[: -len(hidden_suffix)]
        output_key = prefix + "binary_mlp_output_weight"
        threshold_key = prefix + "binary_mlp_hidden_threshold"
        scale_key = prefix + "binary_mlp_output_scale"
        required = (output_key, threshold_key, scale_key)
        missing = [key for key in required if key not in state]
        if missing:
            raise RuntimeError(
                "Incomplete binary MLP state for {}: missing {}".format(
                    hidden_key,
                    missing,
                )
            )

        hidden = state[hidden_key].detach().cpu()
        output = state[output_key].detach().cpu()
        threshold = state[threshold_key].detach().cpu()
        scale = state[scale_key].detach().cpu()
        states, scores = enumerate_binary_mlp_scores(
            hidden,
            output,
            threshold,
            scale,
        )
        representation = "binary_mlp"

        lut_r_key = prefix + "lut_r"
        lut_i_key = prefix + "lut_i"
        if phase == 4.2 and lut_r_key in state and lut_i_key in state:
            scores = torch.stack(
                (
                    state[lut_r_key].detach().cpu(),
                    state[lut_i_key].detach().cpu(),
                ),
                dim=-1,
            )
            representation = "direct_lut5"

        metrics, raw = analyze_scores(scores, states, representation)
        layers.append((hidden_key, metrics, raw))

    if not layers:
        raise RuntimeError(
            "No operation_coefficients or binary_mlp weights found in checkpoint"
        )
    return layers


def analyze_checkpoint(path):
    checkpoint = torch.load(path, map_location="cpu")
    extracted = extract_layers(checkpoint)
    layer_details = []
    tie_scores = []
    non_tie_scores = []
    amplification = []
    for name, metrics, raw in extracted:
        layer_details.append({"name": name, **metrics})
        tie_scores.append(raw["tie_abs"])
        non_tie_scores.append(raw["non_tie_abs"])
        if raw["amplification"].numel():
            amplification.append(raw["amplification"])

    def sum_field(name):
        return sum(layer[name] for layer in layer_details)

    entries = sum_field("entries")
    ties = sum_field("initial_ties")
    changed = sum_field("changed")
    changed_ties = sum_field("changed_ties")
    changed_non_ties = sum_field("changed_non_ties")
    d_sensitive = sum_field("d_sensitive")
    d_pairs = sum_field("d_pairs")
    aggregate = {
        "layers": len(layer_details),
        "representations": sorted(
            {layer["representation"] for layer in layer_details}
        ),
        "entries": entries,
        "initial_ties": ties,
        "initial_tie_ratio": ties / float(entries),
        "changed": changed,
        "changed_ratio": changed / float(entries),
        "changed_ties": changed_ties,
        "changed_tie_ratio": changed_ties / float(ties),
        "changed_non_ties": changed_non_ties,
        "changed_non_tie_ratio": changed_non_ties / float(entries - ties),
        "d_sensitive": d_sensitive,
        "d_pairs": d_pairs,
        "d_sensitive_ratio": d_sensitive / float(d_pairs),
        "tie_abs_scores": scalar_stats(torch.cat(tie_scores)),
        "non_tie_abs_scores": scalar_stats(torch.cat(non_tie_scores)),
        "hard_amplification_on_ties": scalar_stats(
            torch.cat(amplification) if amplification else torch.empty(0)
        ),
    }
    return {
        "checkpoint": str(path.resolve()),
        "phase": checkpoint.get("flow_phase", checkpoint.get("phase")),
        "epoch": checkpoint.get("epoch"),
        "metrics": checkpoint.get("metrics", {}),
        "aggregate": aggregate,
        "layer_details": layer_details,
    }


def percent(value):
    return "{:.2f}%".format(100.0 * value)


def markdown(results):
    lines = [
        "# MLP Operation Score Diagnostics",
        "",
        "This report reconstructs all 32 physical input states from either the",
        "legacy floating polynomial, the binary MLP latent parameters, or the",
        "direct Phase4.2 LUT5 entries.",
        "",
        "| Checkpoint | Representation | Epoch | Changed | Tie changes | Non-tie changes | d-sensitive | Tie abs median | Tie hard amplification median |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for result in results:
        aggregate = result["aggregate"]
        tie_stats = aggregate["tie_abs_scores"]
        amp_stats = aggregate["hard_amplification_on_ties"]
        lines.append(
            "| `{}` | {} | {} | {} | {} | {} | {} | {:.6f} | {:.2f}x |".format(
                Path(result["checkpoint"]).parent.parent.name,
                ", ".join(aggregate["representations"]),
                result["epoch"],
                percent(aggregate["changed_ratio"]),
                percent(aggregate["changed_tie_ratio"]),
                percent(aggregate["changed_non_tie_ratio"]),
                percent(aggregate["d_sensitive_ratio"]),
                tie_stats.get("q50", 0.0),
                amp_stats.get("q50", 0.0),
            )
        )
    lines.extend(["", "## Per-checkpoint details", ""])
    for result in results:
        aggregate = result["aggregate"]
        tie_stats = aggregate["tie_abs_scores"]
        non_tie_stats = aggregate["non_tie_abs_scores"]
        amp_stats = aggregate["hard_amplification_on_ties"]
        lines.extend(
            [
                "### {} (epoch {})".format(result["checkpoint"], result["epoch"]),
                "",
                "- Representation: `{}`.".format(
                    ", ".join(aggregate["representations"])
                ),
                "- Original tie entries: `{}/{} = {}`.".format(
                    aggregate["initial_ties"],
                    aggregate["entries"],
                    percent(aggregate["initial_tie_ratio"]),
                ),
                "- Hard changes: `{}/{} = {}`; tie changes `{}`; non-tie changes `{}`.".format(
                    aggregate["changed"],
                    aggregate["entries"],
                    percent(aggregate["changed_ratio"]),
                    percent(aggregate["changed_tie_ratio"]),
                    percent(aggregate["changed_non_tie_ratio"]),
                ),
                "- Fifth-bit hard sensitivity: `{}/{} = {}`.".format(
                    aggregate["d_sensitive"],
                    aggregate["d_pairs"],
                    percent(aggregate["d_sensitive_ratio"]),
                ),
                "- Original-tie abs score: mean `{:.6f}`, median `{:.6f}`, p90 `{:.6f}`.".format(
                    tie_stats["mean"], tie_stats["q50"], tie_stats["q90"]
                ),
                "- Original-non-tie abs score: mean `{:.6f}`, median `{:.6f}`, p10 `{:.6f}`.".format(
                    non_tie_stats["mean"],
                    non_tie_stats["q50"],
                    non_tie_stats["q10"],
                ),
                "- Tie hard-amplification `2/abs(score)`: median `{:.2f}x`, p90 `{:.2f}x`.".format(
                    amp_stats.get("q50", 0.0),
                    amp_stats.get("q90", 0.0),
                ),
                "",
            ]
        )
    return "\n".join(lines)


def main():
    args = parse_args()
    results = [analyze_checkpoint(Path(path)) for path in args.checkpoints]
    prefix = Path(args.output_prefix)
    prefix.parent.mkdir(parents=True, exist_ok=True)
    json_path = prefix.with_suffix(".json")
    markdown_path = prefix.with_suffix(".md")
    json_path.write_text(
        json.dumps({"checkpoints": results}, indent=2),
        encoding="utf-8",
    )
    markdown_path.write_text(markdown(results), encoding="utf-8")
    print("Wrote {}".format(markdown_path.resolve()))
    print("Wrote {}".format(json_path.resolve()))


if __name__ == "__main__":
    main()

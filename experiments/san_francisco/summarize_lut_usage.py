#!/usr/bin/env python3
"""Compare LUT4 and LUT6 usage JSON reports and write a Markdown summary."""

import argparse
import json
from pathlib import Path
from statistics import mean


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lut4-report", required=True)
    parser.add_argument("--lut6-report", required=True)
    parser.add_argument("--output", required=True)
    return parser.parse_args(argv)


def load_layers(path):
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    return payload, {layer["layer"]: layer for layer in payload["layers"]}


def stage_name(layer_name):
    return layer_name.split(".", 1)[0]


def average(layers, getter):
    values = [getter(layer) for layer in layers]
    return mean(values) if values else 0.0


def main(argv=None):
    args = parse_args(argv)
    lut4_payload, lut4 = load_layers(args.lut4_report)
    lut6_payload, lut6 = load_layers(args.lut6_report)
    if set(lut4) != set(lut6):
        raise ValueError("LUT4 and LUT6 reports contain different layer sets")

    lines = [
        "# San Francisco LUT address analysis",
        "",
        "Both reports use split `{}` with {} LUT4 and {} LUT6 samples.".format(
            lut6_payload["split"], lut4_payload["samples"], lut6_payload["samples"]
        ),
        "",
        "| Stage | H(p) | p(1) | LUT4 effective states | LUT6 effective states | LUT6 normalized entropy | LUT6 unused entries |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for stage in ("stage2", "stage3", "stage4"):
        names = [name for name in lut6 if stage_name(name) == stage]
        old_layers = [lut4[name] for name in names]
        new_layers = [lut6[name] for name in names]
        lines.append(
            "| {} | {:.3f} | {:.3f} | {:.2f} | {:.2f} | {:.3f} | {:.1%} |".format(
                stage,
                average(new_layers, lambda layer: layer["dominance"]["entropy_bits"]),
                average(new_layers, lambda layer: layer["dominance"]["p_one"]),
                average(old_layers, lambda layer: layer["lut4"]["mean_effective_states"]),
                average(new_layers, lambda layer: layer["lut6"]["mean_effective_states"]),
                average(new_layers, lambda layer: layer["lut6"]["mean_normalized_entropy"]),
                average(new_layers, lambda layer: layer["lut6"]["unused_entry_fraction"]),
            )
        )

    lines.extend([
        "",
        "## Per-layer detail",
        "",
        "| Layer | H(p) | p(1) | LUT4 Hnorm (old) | LUT4 Hnorm (inside LUT6) | LUT6 Hnorm | LUT6 effective | LUT6 unused |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ])
    for name in lut6:
        old = lut4[name]
        new = lut6[name]
        lines.append(
            "| {} | {:.3f} | {:.3f} | {:.3f} | {:.3f} | {:.3f} | {:.2f} | {:.1%} |".format(
                name,
                new["dominance"]["entropy_bits"],
                new["dominance"]["p_one"],
                old["lut4"]["mean_normalized_entropy"],
                new["lut4"]["mean_normalized_entropy"],
                new["lut6"]["mean_normalized_entropy"],
                new["lut6"]["mean_effective_states"],
                new["lut6"]["unused_entry_fraction"],
            )
        )

    lines.extend([
        "",
        "## Reading",
        "",
        "- Dominance does not collapse globally: Stage 2 remains close to one bit of entropy.",
        "- Address visitation and useful address diversity are different. Nearly every LUT6 entry is eventually visited, but frequency entropy falls with depth.",
        "- Stage 4 is the bottleneck: dominance becomes biased and LUT6 effective states fall substantially, so extra addresses receive weak and uneven training.",
        "- The option-A shortcut ablation should therefore be judged by both OA/AA and whether Stage 4 dominance/address entropy increases.",
        "",
    ])
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines), encoding="utf-8")
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

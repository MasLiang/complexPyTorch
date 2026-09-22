"""OpenSARShip native Sentinel-1 SLC protocol for the shared complex LUT flow."""

from datasets.opensarship_slc import DEFAULT_CLASSES, build_opensarship_slc_datasets
from experiments.flow.contracts import DatasetBundle, DatasetProtocol, StageSpec
from experiments.san_francisco.baseline import SanFranciscoFPComplexCNN
from experiments.san_francisco.models import build_san_francisco_bireal_model


class OpenSARShipSLCProtocol(DatasetProtocol):
    name = "opensarship_slc"

    def add_arguments(self, parser):
        parser.add_argument("--data-root", default="datasets_external/opensarship")
        parser.add_argument("--chip-size", type=int, default=128)
        parser.add_argument("--split-seed", type=int, default=0)
        parser.add_argument("--force-manifest", action="store_true")
        parser.add_argument("--start-filters", type=int, default=16)
        parser.add_argument("--num-blocks", type=int, default=3)

    def build_bundle(self, args):
        datasets, report = build_opensarship_slc_datasets(
            args.data_root,
            chip_size=args.chip_size,
            classes=DEFAULT_CLASSES,
            split_seed=args.split_seed,
            force_manifest=args.force_manifest,
        )
        return DatasetBundle(
            datasets["train"], datasets["validation"], datasets["test"], len(DEFAULT_CLASSES),
            {"selection_split": "validation", "dataset_protocol": "scene-disjoint split with every cross-split MMSI removed", **report},
        )

    def build_model(self, stage, args, num_classes):
        if stage.name == "fp":
            return SanFranciscoFPComplexCNN(in_channels=2, num_classes=num_classes)
        model_stage = {"bireal_fp": "bireal_fp", "bireal": "bireal", "lut4": "bireal_lut", "lut6_residual": "lut6_residual"}[stage.name]
        return build_san_francisco_bireal_model(
            model_stage, in_channels=2, num_classes=num_classes,
            start_filters=args.start_filters, num_blocks=args.num_blocks,
            post_bn_mode="covariance", pair_lut_parameterization="categorical",
        )

    def stages(self, args):
        return [
            StageSpec("fp", "fp", args.fp_epochs, 1e-3, "adam", "cosine"),
            StageSpec("bireal_fp", "bireal_fp", args.shared_fp_epochs, 1e-3, "adam", "cosine"),
            StageSpec("bireal", "bireal", args.bireal_epochs, 1e-3, "adam", "cosine", parent="bireal_fp", initialization="matching"),
            StageSpec("lut4", "lut4", args.lut4_epochs, 1e-3, "adam", "cosine", parent="bireal", initialization="compile_lut4", lut_lr=5e-3),
            StageSpec("lut6_residual", "lut6_residual", args.lut6_epochs, 2e-4, "adam", "constant", parent="lut4", initialization="expand_lut6", lut_lr=2e-3, residual_alpha_start=0.1, residual_alpha_end=1.0, residual_ramp_epochs=120),
        ]

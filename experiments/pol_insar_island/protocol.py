"""Pol-InSAR-Island FP1/L T3 protocol for the shared complex LUT flow."""

from datasets.pol_insar_island import CLASS_NAMES, REPRESENTATIONS, build_pol_insar_t3_datasets
from experiments.flow.contracts import DatasetBundle, DatasetProtocol, StageSpec
from experiments.san_francisco.baseline import SanFranciscoFPComplexCNN
from experiments.san_francisco.models import build_san_francisco_bireal_model


class PolInSARIslandT3Protocol(DatasetProtocol):
    name = "pol_insar_island_t3"

    def add_arguments(self, parser):
        parser.add_argument("--data-root", default="datasets_external/pol_insar_island")
        parser.add_argument("--patch-size", type=int, default=11)
        parser.add_argument("--val-fraction", type=float, default=0.15)
        parser.add_argument("--spatial-block-size", type=int, default=64)
        parser.add_argument("--split-seed", type=int, default=0)
        parser.add_argument("--representation", choices=REPRESENTATIONS, default="raw_rms")
        parser.add_argument("--start-filters", type=int, default=16)
        parser.add_argument("--num-blocks", type=int, default=3)
        parser.add_argument(
            "--dominance-grad-mode",
            choices=("stop", "ste"),
            default="stop",
            help="Backward mode for the hard phase/Gray comparator in LUT6",
        )
        parser.add_argument(
            "--dominance-ste-margin",
            type=float,
            default=1.0,
            help="STE surrogate half-width; used only with --dominance-grad-mode ste",
        )

    def build_bundle(self, args):
        datasets, report = build_pol_insar_t3_datasets(
            args.data_root,
            patch_size=args.patch_size,
            val_fraction=args.val_fraction,
            block_size=args.spatial_block_size,
            seed=args.split_seed,
            representation=args.representation,
        )
        return DatasetBundle(
            datasets["train"], datasets["validation"], datasets["test"], len(CLASS_NAMES),
            {"selection_split": "validation", "dataset_protocol": "official FP1/L T3 train/test masks; validation spatially derived from official train only", **report},
        )

    def build_model(self, stage, args, num_classes):
        if stage.name == "fp":
            return SanFranciscoFPComplexCNN(in_channels=6, num_classes=num_classes)
        model_stage = {"bireal_fp": "bireal_fp", "bireal": "bireal", "lut4": "bireal_lut", "lut6_residual": "lut6_residual"}[stage.name]
        return build_san_francisco_bireal_model(
            model_stage, in_channels=6, num_classes=num_classes,
            start_filters=args.start_filters, num_blocks=args.num_blocks,
            post_bn_mode="covariance", pair_lut_parameterization="categorical",
            dominance_grad_mode=getattr(args, "dominance_grad_mode", "stop"),
            dominance_ste_margin=getattr(args, "dominance_ste_margin", 1.0),
        )

    def stages(self, args):
        return [
            StageSpec("fp", "fp", args.fp_epochs, 1e-3, "adam", "cosine"),
            StageSpec("bireal_fp", "bireal_fp", args.shared_fp_epochs, 1e-3, "adam", "cosine"),
            StageSpec("bireal", "bireal", args.bireal_epochs, 1e-3, "adam", "cosine", parent="bireal_fp", initialization="matching"),
            StageSpec("lut4", "lut4", args.lut4_epochs, 1e-3, "adam", "cosine", parent="bireal", initialization="compile_lut4", lut_lr=5e-3),
            StageSpec("lut6_residual", "lut6_residual", args.lut6_epochs, 2e-4, "adam", "constant", parent="lut4", initialization="expand_lut6", lut_lr=2e-3, residual_alpha_start=0.1, residual_alpha_end=1.0, residual_ramp_epochs=120),
        ]

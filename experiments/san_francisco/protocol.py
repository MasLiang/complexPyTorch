"""San Francisco AIRSAR data and protocol configuration for the shared flow."""

from experiments.flow.contracts import DatasetBundle, DatasetProtocol, StageSpec
from experiments.san_francisco.baseline import SanFranciscoFPComplexCNN
from experiments.san_francisco.models import build_san_francisco_bireal_model
from datasets.san_francisco import build_san_francisco_datasets


class SanFranciscoProtocol(DatasetProtocol):
    name = "san_francisco"

    def add_arguments(self, parser):
        parser.add_argument("--data-root", default="data/san_francisco")
        parser.add_argument("--stk-file", default="san_francisco900x1024.stk")
        parser.add_argument("--labels-file", default="SF-AIRSAR-label2d.png")
        parser.add_argument("--patch-size", type=int, default=11)
        parser.add_argument("--split-mode", choices=("spatial", "random"), default="spatial")
        parser.add_argument("--spatial-block-size", type=int, default=64)
        parser.add_argument("--train-fraction", type=float, default=0.1)
        parser.add_argument("--val-fraction", type=float, default=0.1)
        parser.add_argument("--split-seed", type=int, default=0)
        parser.add_argument("--start-filters", type=int, default=16)
        parser.add_argument("--num-blocks", type=int, default=3)

    def build_bundle(self, args):
        bundle = build_san_francisco_datasets(
            args.data_root, stk_file=args.stk_file, labels_file=args.labels_file,
            patch_size=args.patch_size, split_mode=args.split_mode,
            train_fraction=args.train_fraction, val_fraction=args.val_fraction,
            seed=args.split_seed, spatial_block_size=args.spatial_block_size,
            stats_path=None,
        )
        return DatasetBundle(bundle.train, bundle.val, bundle.test, 5, {
            "selection_split": "validation", "split_mode": args.split_mode,
            "split_seed": args.split_seed, "patch_size": args.patch_size,
        })

    def build_model(self, stage, args, num_classes):
        if stage.name == "independent_fp":
            return SanFranciscoFPComplexCNN(in_channels=6, num_classes=num_classes)
        model_stage = {"shared_fp": "bireal_fp", "bireal": "bireal", "lut4": "bireal_lut", "lut6_residual": "lut6_residual"}[stage.name]
        return build_san_francisco_bireal_model(
            model_stage, in_channels=6, num_classes=num_classes,
            start_filters=args.start_filters, num_blocks=args.num_blocks,
            post_bn_mode="covariance", pair_lut_parameterization="categorical",
        )

    def stages(self, args):
        return [
            StageSpec("independent_fp", "independent_fp", args.fp_epochs, 1e-3, "adam", "cosine"),
            StageSpec("shared_fp", "shared_fp", args.shared_fp_epochs, 1e-3, "adam", "cosine"),
            StageSpec("bireal", "bireal", args.bireal_epochs, 1e-3, "adam", "cosine", parent="shared_fp", initialization="matching"),
            StageSpec("lut4", "lut4", args.lut4_epochs, 1e-3, "adam", "cosine", parent="bireal", initialization="compile_lut4", lut_lr=5e-3),
            StageSpec("lut6_residual", "lut6_residual", args.lut6_epochs, 2e-4, "adam", "constant", parent="lut4", initialization="expand_lut6", lut_lr=2e-3, residual_alpha_start=0.1, residual_alpha_end=1.0, residual_ramp_epochs=120),
        ]

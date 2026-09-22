"""CIFAR-10 data and protocol configuration for the shared flow."""

from types import SimpleNamespace

from complexPyTorch.complexBinaryResNet import BinaryComplexResNet
from experiments.flow.contracts import DatasetBundle, DatasetProtocol, StageSpec


class Cifar10Protocol(DatasetProtocol):
    name = "cifar10"

    def add_arguments(self, parser):
        parser.add_argument("--data-root", default="data")
        parser.add_argument("--augmentation", choices=("standard", "real_lut"), default="real_lut")
        parser.add_argument("--with-validation", action="store_false", dest="no_validation", default=True)
        parser.add_argument("--start-filters", type=int, default=16)
        parser.add_argument("--num-blocks", type=int, default=3)
        parser.add_argument("--lut4-parameterization", choices=("independent", "categorical"), default="categorical")

    def build_bundle(self, args):
        import logging
        import training as legacy_cifar

        legacy_args = SimpleNamespace(
            dataset="cifar10",
            datadir=args.data_root,
            no_validation=args.no_validation,
            augmentation=args.augmentation,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            seed=args.seed,
        )
        train, validation, test, classes, _, _ = legacy_cifar.build_datasets(
            legacy_args, logging.getLogger("flow.cifar10"), False, True
        )
        return DatasetBundle(
            train, validation, test, classes,
            {"selection_split": "test" if validation is None else "validation", "augmentation": args.augmentation},
        )

    def build_model(self, stage, args, num_classes):
        phase = {"shared_fp": 1, "bireal": 2, "lut4": 3, "lut6_residual": 3}[stage.name]
        residual = stage.name == "lut6_residual"
        return BinaryComplexResNet(
            in_channels=3, num_blocks=args.num_blocks, start_filters=args.start_filters,
            num_classes=num_classes, spectral_pool_scheme="none", spectral_pool_gamma=0.0,
            per_channel=True, weight_grad_mode="ste", act_grad_mode="bireal",
            binary_stem=False, is_sar_input=False, is_binary=phase >= 2, phase=phase,
            post_bn_mode="covariance", phase3_operator="pair_lut4",
            pair_lut_parameterization="categorical_residual" if residual else args.lut4_parameterization,
            pair_lut_inputs=6 if residual else 4,
            pair_lut_encoding="dominance" if residual else "standard",
            dominance_grad_mode="stop",
        )

    def stages(self, args):
        if args.lut4_parameterization != "categorical":
            lut6_parent = None
        else:
            lut6_parent = "lut4"
        stages = [
            StageSpec("shared_fp", "shared_fp", args.fp_epochs, 0.1, "sgd", "bireal", weight_decay=1e-4),
            StageSpec("bireal", "bireal", args.bireal_epochs, 0.01, "adam", "linear", parent="shared_fp", initialization="matching", label_smoothing=0.1),
            StageSpec("lut4", "lut4", args.lut4_epochs, 0.02, "adam", "multistep", parent="bireal", initialization="compile_lut4", label_smoothing=0.1),
        ]
        if lut6_parent is not None:
            stages.append(StageSpec("lut6_residual", "lut6_residual", args.lut6_epochs, 2e-4, "adam", "constant", parent=lut6_parent, initialization="expand_lut6", lut_lr=2e-3, residual_alpha_start=0.1, residual_alpha_end=1.0, residual_ramp_epochs=120, label_smoothing=0.1))
        return stages

#!/usr/bin/env python3
"""Train the active full-precision and binary complex ResNet phases."""

import argparse
import json
import logging
import math
import os
import random
import socket
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Subset
from torch.utils.data.distributed import DistributedSampler
from torchvision import datasets, transforms

from complexPyTorch.complexBinaryResNet import (
    ACTIVE_PHASES,
    BinaryComplexResNet,
)
from complexPyTorch.complexLayers import (
    BinaryComplexBitActivation,
    BinaryComplexConv2d,
    ComplexBatchNorm2d,
    LUTBinaryConv2d,
    NaiveComplexBatchNorm2d,
    PairLUT4ComplexConv2d,
    TripleLUT6ComplexConv2d,
    TwoLUTComplexConv2d,
)


PHASE_DESCRIPTIONS = {
    1: "full-precision BiReal",
    2: "complex BiReal",
    3: "complex BiReal with 0/1 activation bits and configurable LUT convolution",
}
PHASE_CHECKPOINT_TEMPLATE = "Bestmodel_phase{}.pt"
LAST_CHECKPOINT_TEMPLATE = "Lastmodel_phase{}.pt"
PHASE_METRICS_FILENAME = "phase_metrics.json"
LOG_LEVELS = {
    "debug": logging.DEBUG,
    "info": logging.INFO,
    "warn": logging.WARNING,
    "error": logging.ERROR,
}


class SubtractMean:
    def __init__(self, mean):
        self.mean = mean

    def __call__(self, tensor):
        return tensor - self.mean


class SVHNDataset(datasets.SVHN):
    def __getitem__(self, index):
        image, target = super().__getitem__(index)
        return image, 0 if target == 10 else target


def phase_checkpoint_filename(phase):
    return PHASE_CHECKPOINT_TEMPLATE.format(phase)


def last_checkpoint_filename(phase):
    return LAST_CHECKPOINT_TEMPLATE.format(phase)


def phase_checkpoint_path(workdir, phase):
    return os.path.join(
        workdir,
        "chkpts",
        phase_checkpoint_filename(phase),
    )


def unwrap_model(model):
    while True:
        if isinstance(model, DDP):
            model = model.module
            continue
        original = getattr(model, "_orig_mod", None)
        if original is not None:
            model = original
            continue
        return model


def _strip_state_wrappers(key):
    changed = True
    while changed:
        changed = False
        for prefix in ("module.", "_orig_mod."):
            if key.startswith(prefix):
                key = key[len(prefix):]
                changed = True
    return key


def _checkpoint_phase(checkpoint):
    if not isinstance(checkpoint, dict):
        return None
    phase = checkpoint.get("phase")
    if phase is None and isinstance(checkpoint.get("metrics"), dict):
        phase = checkpoint["metrics"].get("phase")
    if phase is None and isinstance(checkpoint.get("args"), dict):
        phase = checkpoint["args"].get("phase")
    return None if phase is None else int(phase)


def _checkpoint_state_dict(checkpoint):
    if not isinstance(checkpoint, dict):
        raise TypeError("Checkpoint must contain a state dictionary")
    if isinstance(checkpoint.get("model"), dict):
        return checkpoint["model"]
    if isinstance(checkpoint.get("state_dict"), dict):
        return checkpoint["state_dict"]
    if checkpoint and all(torch.is_tensor(value) for value in checkpoint.values()):
        return checkpoint
    raise KeyError("Checkpoint has no model or state_dict entry")


def load_phase_checkpoint(
    model,
    checkpoint_path,
    device="cpu",
    expected_phase=None,
):
    """Load matching tensors and require every trainable parameter to map."""
    checkpoint_path = os.fspath(checkpoint_path)
    print("==> Loading weights from '{}' ...".format(checkpoint_path))
    checkpoint = torch.load(checkpoint_path, map_location=device)
    stored_phase = _checkpoint_phase(checkpoint)
    if (
        expected_phase is not None
        and stored_phase is not None
        and stored_phase != expected_phase
    ):
        raise ValueError(
            "Checkpoint {} belongs to Phase {}, but Phase {} is required.".format(
                checkpoint_path,
                stored_phase,
                expected_phase,
            )
        )

    base_model = unwrap_model(model)
    source_state = _checkpoint_state_dict(checkpoint)
    target_state = base_model.state_dict()
    target_by_clean_key = {
        _strip_state_wrappers(key): key for key in target_state
    }
    mapped_state = {}
    skipped = []
    shape_mismatches = []

    for source_key, source_value in source_state.items():
        clean_key = _strip_state_wrappers(source_key)
        target_key = target_by_clean_key.get(clean_key)
        if target_key is None:
            skipped.append(clean_key)
            continue
        if target_state[target_key].shape != source_value.shape:
            shape_mismatches.append(
                (
                    clean_key,
                    tuple(source_value.shape),
                    tuple(target_state[target_key].shape),
                )
            )
            continue
        mapped_state[target_key] = source_value

    missing_keys, unexpected_keys = base_model.load_state_dict(
        mapped_state,
        strict=False,
    )
    parameter_keys = set(dict(base_model.named_parameters()))
    missing_parameters = [
        key for key in missing_keys if key in parameter_keys
    ]
    if missing_parameters:
        raise RuntimeError(
            "Checkpoint did not initialize model parameters: {}".format(
                missing_parameters
            )
        )
    if shape_mismatches:
        raise RuntimeError(
            "Checkpoint tensors have incompatible shapes: {}".format(
                shape_mismatches
            )
        )
    if unexpected_keys:
        raise RuntimeError(
            "Unexpected mapped checkpoint keys: {}".format(unexpected_keys)
        )

    print(
        "==> Loaded {}/{} checkpoint tensors ({} unmatched, {} missing buffers).".format(
            len(mapped_state),
            len(source_state),
            len(skipped),
            len(missing_keys),
        )
    )
    return checkpoint


def _load_dominance_lut6_warm_start(
    model,
    checkpoint,
    checkpoint_path,
):
    """Expand a categorical LUT4 Phase 3 checkpoint into dominance LUT6."""
    base_model = unwrap_model(model)
    source_state = {
        _strip_state_wrappers(key): value
        for key, value in _checkpoint_state_dict(checkpoint).items()
    }
    target_state = base_model.state_dict()
    target_luts = [
        (name, module)
        for name, module in base_model.named_modules()
        if isinstance(module, PairLUT4ComplexConv2d)
    ]
    if not target_luts or any(
        module.activation_encoding != "dominance"
        for _, module in target_luts
    ):
        raise ValueError(
            "A Phase 3 checkpoint can warm-start only dominance PairLUT6 models"
        )

    lut_state_names = set()
    lut_parameter_names = set()
    for module_name, module in target_luts:
        prefix = module_name + "." if module_name else ""
        own_parameters = [
            name for name, _ in module.named_parameters(recurse=False)
        ]
        lut_state_names.update(
            prefix + name for name in own_parameters
        )
        lut_state_names.update(
            prefix + name
            for name, _ in module.named_buffers(recurse=False)
        )
        lut_parameter_names.update(
            prefix + name for name in own_parameters
        )

    def source_value_for_target(key):
        value = source_state.get(key)
        if value is not None:
            return value
        for target_index in (1, 2):
            marker = ".proj.{}.".format(target_index)
            if marker not in key:
                continue
            legacy_key = key.replace(
                marker,
                ".proj.{}.".format(target_index - 1),
                1,
            )
            value = source_state.get(legacy_key)
            if value is not None:
                return value
        return None

    mapped_state = {}
    shape_mismatches = []
    for key, target_value in target_state.items():
        if key in lut_state_names:
            continue
        source_value = source_value_for_target(key)
        if source_value is None:
            continue
        if source_value.shape != target_value.shape:
            shape_mismatches.append(
                (key, tuple(source_value.shape), tuple(target_value.shape))
            )
            continue
        mapped_state[key] = source_value
    if shape_mismatches:
        raise RuntimeError(
            "LUT4 warm-start shared tensors have incompatible shapes: {}".format(
                shape_mismatches
            )
        )

    missing_keys, unexpected_keys = base_model.load_state_dict(
        mapped_state,
        strict=False,
    )
    parameter_names = set(dict(base_model.named_parameters()))
    missing_shared_parameters = [
        key
        for key in missing_keys
        if key in parameter_names and key not in lut_parameter_names
    ]
    if missing_shared_parameters:
        raise RuntimeError(
            "LUT4 checkpoint did not initialize shared parameters: {}".format(
                missing_shared_parameters
            )
        )
    if unexpected_keys:
        raise RuntimeError(
            "Unexpected LUT4 warm-start keys: {}".format(unexpected_keys)
        )

    expanded_luts = 0
    for module_name, module in target_luts:
        prefix = module_name + "." if module_name else ""
        source_key = prefix + "weight"
        source_weight = source_state.get(source_key)
        if source_weight is None:
            raise RuntimeError(
                "LUT4 checkpoint is missing {}".format(source_key)
            )
        expected_source_shape = (
            module.out_channels,
            module.group_num,
            16,
            4,
        )
        if tuple(source_weight.shape) != expected_source_shape:
            raise RuntimeError(
                "{} has shape {}; expected categorical LUT4 shape {}".format(
                    source_key,
                    tuple(source_weight.shape),
                    expected_source_shape,
                )
            )

        bits = module.state_bits.long()
        old_addresses = (
            bits[:, 0] * 8
            + bits[:, 1] * 4
            + bits[:, 3] * 2
            + bits[:, 4]
        )
        expanded = source_weight.index_select(
            dim=2,
            index=old_addresses.to(source_weight.device),
        )
        with torch.no_grad():
            if module.parameterization == "categorical_residual":
                module.dominance_base.copy_(
                    source_weight.to(
                        device=module.dominance_base.device,
                        dtype=module.dominance_base.dtype,
                    )
                )
                module.weight.zero_()
            else:
                module.weight.copy_(
                    expanded.to(
                        device=module.weight.device,
                        dtype=module.weight.dtype,
                    )
                )
        expanded_luts += 1

    print(
        "==> Loaded {} shared tensors and expanded {} categorical LUT4 "
        "operators into dominance LUT6 from '{}'.".format(
            len(mapped_state),
            expanded_luts,
            checkpoint_path,
        )
    )
    return checkpoint



def load_phase3_shared_checkpoint(model, checkpoint_path, device="cpu"):
    """Load shared Phase 2 state and compile compatible weights into LUTs."""
    checkpoint_path = os.fspath(checkpoint_path)
    print("==> Loading Phase 3 initialization from '{}' ...".format(checkpoint_path))
    checkpoint = torch.load(checkpoint_path, map_location=device)
    stored_phase = _checkpoint_phase(checkpoint)
    if stored_phase == 3:
        stored_args = checkpoint.get("args", {})
        if stored_args.get("pair_lut_parameterization") == "categorical_residual":
            return load_phase_checkpoint(
                model, checkpoint_path, device=device, expected_phase=3
            )
        return _load_dominance_lut6_warm_start(
            model,
            checkpoint,
            checkpoint_path,
        )
    if stored_phase is not None and stored_phase != 2:
        raise ValueError(
            "Checkpoint {} belongs to Phase {}, but Phase 2 is required.".format(
                checkpoint_path, stored_phase
            )
        )

    base_model = unwrap_model(model)
    source_state = {
        _strip_state_wrappers(key): value
        for key, value in _checkpoint_state_dict(checkpoint).items()
    }
    target_state = base_model.state_dict()
    lut_parameter_names = set()
    for module_name, module in base_model.named_modules():
        if isinstance(module, (LUTBinaryConv2d, PairLUT4ComplexConv2d, TripleLUT6ComplexConv2d)):
            prefix = module_name + "." if module_name else ""
            lut_parameter_names.update(
                prefix + name
                for name, _ in module.named_parameters(recurse=False)
            )

    def source_value_for_target(key):
        value = source_state.get(key)
        if value is not None:
            return value
        for target_index in (1, 2):
            marker = ".proj.{}.".format(target_index)
            if marker not in key:
                continue
            legacy_key = key.replace(
                marker,
                ".proj.{}.".format(target_index - 1),
                1,
            )
            value = source_state.get(legacy_key)
            if value is not None:
                return value
        return None
    mapped_state = {}
    shape_mismatches = []
    for key, target_value in target_state.items():
        if key in lut_parameter_names:
            continue
        source_value = source_value_for_target(key)
        if source_value is None:
            continue
        if source_value.shape != target_value.shape:
            shape_mismatches.append(
                (key, tuple(source_value.shape), tuple(target_value.shape))
            )
            continue
        mapped_state[key] = source_value
    if shape_mismatches:
        raise RuntimeError(
            "Shared checkpoint tensors have incompatible shapes: {}".format(
                shape_mismatches
            )
        )

    missing_keys, unexpected_keys = base_model.load_state_dict(
        mapped_state, strict=False
    )
    parameter_names = set(dict(base_model.named_parameters()))
    missing_shared_parameters = [
        key
        for key in missing_keys
        if key in parameter_names and key not in lut_parameter_names
    ]
    if missing_shared_parameters:
        raise RuntimeError(
            "Phase 2 checkpoint did not initialize shared parameters: {}".format(
                missing_shared_parameters
            )
        )
    if unexpected_keys:
        raise RuntimeError(
            "Unexpected mapped checkpoint keys: {}".format(unexpected_keys)
        )

    compiled_pair_luts = 0
    random_pair_luts = 0
    for module_name, module in base_model.named_modules():
        if not isinstance(module, PairLUT4ComplexConv2d):
            continue
        if module.activation_encoding == "dominance":
            random_pair_luts += 1
            continue
        prefix = module_name + "." if module_name else ""
        real_key = prefix + "conv_r.weight"
        imag_key = prefix + "conv_i.weight"
        missing_source = [
            key for key in (real_key, imag_key) if key not in source_state
        ]
        if missing_source:
            raise RuntimeError(
                "Phase 2 checkpoint cannot initialize {}: missing {}".format(
                    module_name, missing_source
                )
            )
        module.initialize_from_binary_complex_weights(
            source_state[real_key],
            source_state[imag_key],
        )
        compiled_pair_luts += 1

    random_lut_tables = len(lut_parameter_names) - compiled_pair_luts
    print(
        "==> Loaded {} shared tensors; compiled {} PairLUT4 operators from "
        "Phase 2 weights; {} LUT parameter sets keep random initialization "
        "({} dominance LUT6 operators).".format(
            len(mapped_state),
            compiled_pair_luts,
            random_lut_tables,
            random_pair_luts,
        )
    )
    return checkpoint


def setup_logger(workdir, loglevel, is_main):
    if not is_main:
        logger = logging.getLogger("train.worker")
        logger.handlers.clear()
        logger.addHandler(logging.NullHandler())
        logger.propagate = False
        return logger

    logdir = Path(workdir) / "logs"
    logdir.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("train")
    logger.handlers.clear()
    logger.setLevel(LOG_LEVELS[loglevel])
    logger.propagate = False

    formatter = logging.Formatter(
        "[%(asctime)s ~~ %(levelname)-8s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)
    file_handler = logging.FileHandler(
        logdir / "train.txt",
        mode="a",
        encoding="utf-8",
    )
    file_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)
    logger.addHandler(file_handler)
    return logger


def init_distributed(args):
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    enabled = bool(args.ddp or world_size > 1)
    if not enabled:
        return False, 0, 0, 1

    backend = args.ddp_backend
    if backend == "nccl" and (args.cpu or not torch.cuda.is_available()):
        backend = "gloo"
    if not dist.is_initialized():
        dist.init_process_group(backend=backend, init_method="env://")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    local_rank = args.local_rank
    if local_rank is None:
        local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    return True, rank, local_rank, world_size


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed % (2**32))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def compute_pixel_mean(dataset, batch_size=256, num_workers=2):
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )
    pixel_sum = None
    count = 0
    for data, _ in loader:
        if pixel_sum is None:
            pixel_sum = torch.zeros_like(data[0])
        pixel_sum += data.sum(dim=0)
        count += data.size(0)
    if count == 0:
        raise RuntimeError("Cannot compute a pixel mean from an empty dataset")
    return (pixel_sum / float(count)).to(torch.float32)


def _dataset_spec(name):
    if name == "cifar10":
        return datasets.CIFAR10, {"train": True}, {"train": False}, 45000, 10
    if name == "cifar100":
        return datasets.CIFAR100, {"train": True}, {"train": False}, 45000, 100
    if name == "svhn":
        return (
            SVHNDataset,
            {"split": "train"},
            {"split": "test"},
            65000,
            10,
        )
    raise ValueError("Unknown dataset: {}".format(name))


def _download_base_dataset(
    dataset_cls,
    datadir,
    train_kwargs,
    ddp_enabled,
    is_main,
):
    common = {
        "root": datadir,
        "transform": transforms.ToTensor(),
        **train_kwargs,
    }
    if not ddp_enabled:
        return dataset_cls(download=True, **common)
    if is_main:
        dataset = dataset_cls(download=True, **common)
        dist.barrier()
        return dataset
    dist.barrier()
    return dataset_cls(download=False, **common)


def build_datasets(args, logger, ddp_enabled, is_main):
    (
        dataset_cls,
        train_kwargs,
        test_kwargs,
        nominal_train_size,
        num_classes,
    ) = _dataset_spec(args.dataset)
    base_train = _download_base_dataset(
        dataset_cls,
        args.datadir,
        train_kwargs,
        ddp_enabled,
        is_main,
    )

    split_rng = np.random.RandomState(0xDEADBEEF)
    shuffled_indices = split_rng.permutation(len(base_train))
    train_size = min(nominal_train_size, len(base_train))
    train_indices = shuffled_indices[:train_size]
    val_indices = shuffled_indices[train_size:]
    mean_indices = (
        shuffled_indices if args.no_validation else train_indices
    )

    pixel_mean = None
    if is_main:
        if args.augmentation == "real_lut":
            if args.dataset != "cifar10":
                raise ValueError(
                    "real_lut augmentation currently supports CIFAR-10 only"
                )
            pixel_mean = torch.tensor(
                (0.4914, 0.4822, 0.4465),
                dtype=torch.float32,
            ).view(3, 1, 1)
        else:
            pixel_mean = compute_pixel_mean(
                Subset(base_train, mean_indices),
                batch_size=max(args.batch_size, 256),
                num_workers=args.num_workers,
            )
    if ddp_enabled:
        objects = [pixel_mean]
        dist.broadcast_object_list(objects, src=0)
        pixel_mean = objects[0]

    if args.augmentation == "real_lut":
        normalize = transforms.Normalize(
            (0.4914, 0.4822, 0.4465),
            (0.2023, 0.1994, 0.2010),
        )
        train_transform = transforms.Compose(
            [
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.AutoAugment(transforms.AutoAugmentPolicy.CIFAR10),
                transforms.ToTensor(),
                normalize,
            ]
        )
        eval_transform = transforms.Compose(
            [transforms.ToTensor(), normalize]
        )
    else:
        train_transform = transforms.Compose(
            [
                transforms.RandomAffine(
                    degrees=0,
                    translate=(0.125, 0.125),
                ),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                SubtractMean(pixel_mean),
            ]
        )
        eval_transform = transforms.Compose(
            [
                transforms.ToTensor(),
                SubtractMean(pixel_mean),
            ]
        )
    train_data = dataset_cls(
        root=args.datadir,
        download=False,
        transform=train_transform,
        **train_kwargs
    )
    test_data = dataset_cls(
        root=args.datadir,
        download=False,
        transform=eval_transform,
        **test_kwargs
    )

    if args.no_validation:
        train_split = Subset(train_data, shuffled_indices)
        val_split = None
    else:
        val_data = dataset_cls(
            root=args.datadir,
            download=False,
            transform=eval_transform,
            **train_kwargs
        )
        train_split = Subset(train_data, train_indices)
        val_split = Subset(val_data, val_indices)

    train_sampler = None
    if ddp_enabled:
        train_sampler = DistributedSampler(
            train_split,
            shuffle=True,
            seed=args.seed,
        )
    if is_main:
        logger.info("Training set size:   %d", len(train_split))
        logger.info(
            "Validation set size: %d",
            0 if val_split is None else len(val_split),
        )
        logger.info("Test set size:       %d", len(test_data))
    return (
        train_split,
        val_split,
        test_data,
        num_classes,
        pixel_mean,
        train_sampler,
    )


def build_model(args, num_classes):
    if args.phase not in ACTIVE_PHASES:
        raise ValueError("Only Phase 1, Phase 2, and Phase 3 are active")
    return BinaryComplexResNet(
        in_channels=3,
        num_blocks=args.num_blocks,
        start_filters=args.start_filter,
        num_classes=num_classes,
        spectral_pool_scheme=args.spectral_pool_scheme,
        spectral_pool_gamma=args.spectral_pool_gamma,
        per_channel=args.binary_weight_scale == "channel",
        weight_grad_mode=args.weight_grad_mode,
        act_grad_mode=args.activation_grad_mode,
        binary_stem=args.binary_stem,
        is_sar_input=False,
        is_binary=args.phase >= 2,
        phase=args.phase,
        post_bn_mode=args.post_bn_mode,
        phase3_operator=args.phase3_operator,
        pair_lut_parameterization=args.pair_lut_parameterization,
        pair_lut_inputs=args.pair_lut_inputs,
        pair_lut_encoding=args.pair_lut_encoding,
        dominance_grad_mode=args.dominance_grad_mode,
        dominance_ste_margin=args.dominance_ste_margin,
    )


def split_weight_decay_params(model):
    base_model = unwrap_model(model)
    no_decay_ids = set()
    for module in base_model.modules():
        if "BatchNorm" in module.__class__.__name__:
            no_decay_ids.update(
                id(parameter)
                for parameter in module.parameters(recurse=False)
            )
        if isinstance(
            module,
            (BinaryComplexConv2d, LUTBinaryConv2d, PairLUT4ComplexConv2d, TripleLUT6ComplexConv2d),
        ):
            no_decay_ids.update(
                id(parameter) for parameter in module.parameters()
            )

    decay = []
    no_decay = []
    for name, parameter in base_model.named_parameters():
        if not parameter.requires_grad:
            continue
        target = no_decay if (
            name.endswith(".bias") or id(parameter) in no_decay_ids
        ) else decay
        target.append(parameter)
    return decay, no_decay


def build_optimizer(args, model):
    decay_params, no_decay_params = split_weight_decay_params(model)
    residual_params = [
        module.weight
        for module in unwrap_model(model).modules()
        if isinstance(module, PairLUT4ComplexConv2d)
        and module.parameterization == "categorical_residual"
    ]
    residual_ids = {id(parameter) for parameter in residual_params}
    decay_params = [
        parameter for parameter in decay_params
        if id(parameter) not in residual_ids
    ]
    no_decay_params = [
        parameter for parameter in no_decay_params
        if id(parameter) not in residual_ids
    ]
    parameter_groups = [
        {
            "name": "decay",
            "params": decay_params,
            "weight_decay": args.weight_decay,
        },
        {
            "name": "no_decay",
            "params": no_decay_params,
            "weight_decay": 0.0,
        },
    ]
    if residual_params:
        parameter_groups.append(
            {
                "name": "dominance_residual",
                "params": residual_params,
                "weight_decay": 0.0,
                "lr_scale": args.dominance_residual_lr / args.lr,
            }
        )
    if args.optimizer in ("sgd", "nag"):
        return torch.optim.SGD(
            parameter_groups,
            lr=args.lr,
            momentum=args.momentum,
            nesterov=args.optimizer == "nag",
        )
    if args.optimizer == "rmsprop":
        return torch.optim.RMSprop(parameter_groups, lr=args.lr)
    if args.optimizer == "adam":
        return torch.optim.Adam(
            parameter_groups,
            lr=args.lr,
            betas=(args.beta1, args.beta2),
        )
    if args.optimizer == "adamw":
        return torch.optim.AdamW(
            parameter_groups,
            lr=args.lr,
            betas=(args.beta1, args.beta2),
        )
    raise ValueError("Unknown optimizer: {}".format(args.optimizer))


def learning_rate_for_epoch(epoch, args):
    if args.schedule == "constant":
        return args.lr
    if args.schedule == "multistep":
        decay_count = sum(
            epoch >= milestone
            for milestone in (90, 140, 180, 220)
        )
        return args.lr * (0.1 ** decay_count)
    if args.schedule == "cosine":
        progress = epoch / float(max(args.num_epochs - 1, 1))
        minimum = args.lr * args.min_lr_factor
        return min(
            args.lr,
            minimum + 0.5 * (args.lr - minimum) * (
                1.0 + math.cos(math.pi * progress)
            ),
        )
    if args.schedule in ("linear", "bireal_reference"):
        return args.lr * max(
            1.0 - epoch / float(max(args.num_epochs, 1)),
            0.0,
        )
    if args.schedule == "bireal":
        warmup_epochs = min(5, args.num_epochs)
        if epoch < warmup_epochs:
            return args.lr * (epoch + 1) / float(warmup_epochs)
        half = int(args.num_epochs * 0.5)
        three_quarters = int(args.num_epochs * 0.75)
        seven_eighths = int(args.num_epochs * 0.875)
        if epoch < half:
            return args.lr
        if epoch < three_quarters:
            return args.lr * 0.1
        if epoch < seven_eighths:
            return args.lr * 0.01
        return args.lr * 0.001
    raise ValueError("Unknown schedule: {}".format(args.schedule))


def set_optimizer_lr(optimizer, learning_rate):
    for group in optimizer.param_groups:
        group["lr"] = learning_rate * group.get("lr_scale", 1.0)


def dominance_residual_alpha_for_epoch(epoch, args):
    if args.dominance_residual_ramp_epochs <= 1:
        return args.dominance_residual_alpha_end
    progress = min(
        max(epoch, 0) / float(args.dominance_residual_ramp_epochs - 1),
        1.0,
    )
    return (
        args.dominance_residual_alpha_start
        + progress * (
            args.dominance_residual_alpha_end
            - args.dominance_residual_alpha_start
        )
    )


def configure_lut_training_flow(model, epoch, args):
    """Report the always-hard binary Phase 3 LUT configuration."""
    modules = dict(model.named_modules())
    units = [
        (
            name,
            module,
            modules.get(name.rsplit(".", 1)[0] + ".act"),
        )
        for name, module in model.named_modules()
        if isinstance(module, (TwoLUTComplexConv2d, PairLUT4ComplexConv2d, TripleLUT6ComplexConv2d))
    ]
    operator_count = len(units)
    residual_operators = [
        operator
        for _, operator, _ in units
        if isinstance(operator, PairLUT4ComplexConv2d)
        and operator.parameterization == "categorical_residual"
    ]
    residual_alpha = None
    if residual_operators:
        residual_alpha = dominance_residual_alpha_for_epoch(epoch, args)
        for operator in residual_operators:
            operator.set_dominance_residual_alpha(residual_alpha)
    if args.phase != 3 or not units:
        return {
            "flow": "not_applicable",
            "stage": "not_applicable",
            "temperature": None,
            "hard_operators": operator_count,
            "total_operators": operator_count,
            "fully_hard": True,
            "dominance_residual_alpha": residual_alpha,
        }

    for name, operator, activation in units:
        if not isinstance(activation, BinaryComplexBitActivation):
            raise TypeError(
                "Phase 3 operator {} is not paired with bit activation".format(
                    name
                )
            )
    return {
        "flow": "hard",
        "stage": (
            "dominance_residual_ramp"
            if residual_alpha is not None
            and epoch < args.dominance_residual_ramp_epochs - 1
            else "fully_hard"
        ),
        "temperature": None,
        "hard_operators": operator_count,
        "total_operators": operator_count,
        "fully_hard": True,
        "dominance_residual_alpha": residual_alpha,
    }


def _gradient_tensor_stats(name, parameter):
    gradient = parameter.grad.detach()
    finite_mask = torch.isfinite(gradient)
    finite_count = int(finite_mask.sum().item())
    total = gradient.numel()
    finite_values = gradient[finite_mask]
    stats = {
        "name": name,
        "shape": list(gradient.shape),
        "dtype": str(gradient.dtype),
        "total": total,
        "finite": finite_count,
        "nan": int(torch.isnan(gradient).sum().item()),
        "pos_inf": int(torch.isposinf(gradient).sum().item()),
        "neg_inf": int(torch.isneginf(gradient).sum().item()),
    }
    if finite_values.numel():
        finite_double = finite_values.double()
        stats["finite_max_abs"] = float(finite_double.abs().max().item())
        stats["finite_l2_norm_float64"] = float(
            torch.linalg.vector_norm(finite_double).item()
        )
    return stats


def _write_nonfinite_gradient_report(
    model,
    workdir,
    epoch,
    batch_index,
    loss,
    output,
    clipnorm,
    clip_error,
):
    base_model = unwrap_model(model)
    gradient_stats = [
        _gradient_tensor_stats(name, parameter)
        for name, parameter in base_model.named_parameters()
        if parameter.grad is not None
    ]
    nonfinite = [
        stats for stats in gradient_stats if stats["finite"] != stats["total"]
    ]
    finite_ranked = sorted(
        (
            stats
            for stats in gradient_stats
            if stats["finite"] == stats["total"]
        ),
        key=lambda stats: stats.get("finite_l2_norm_float64", 0.0),
        reverse=True,
    )
    squared_norm_sum = sum(
        stats.get("finite_l2_norm_float64", 0.0) ** 2
        for stats in gradient_stats
    )
    quantization = []
    for name, module in base_model.named_modules():
        if isinstance(module, (LUTBinaryConv2d, PairLUT4ComplexConv2d, TripleLUT6ComplexConv2d)):
            quantization.append(
                {
                    "name": name,
                    "kind": "lut",
                    "tau": None,
                    "hard": True,
                }
            )
        elif isinstance(module, BinaryComplexBitActivation):
            quantization.append(
                {
                    "name": name,
                    "kind": "activation",
                    "tau": float(module.tau.item()),
                    "hard": bool(module.hard.item()),
                }
            )
    report = {
        "epoch": int(epoch),
        "batch_index": int(batch_index),
        "loss": float(loss.detach().item()),
        "output_max_abs": float(output.detach().abs().max().item()),
        "clipnorm": float(clipnorm),
        "clip_error": str(clip_error),
        "diagnosis": (
            "nonfinite_gradient_elements"
            if nonfinite
            else "float32_global_norm_overflow_with_finite_elements"
        ),
        "gradient_tensor_count": len(gradient_stats),
        "nonfinite_gradient_tensor_count": len(nonfinite),
        "global_finite_l2_norm_float64": math.sqrt(squared_norm_sum),
        "nonfinite_gradients": nonfinite,
        "largest_finite_gradients": finite_ranked[:20],
        "quantization_state": quantization,
    }
    report_dir = Path(workdir) / "debug"
    report_dir.mkdir(parents=True, exist_ok=True)
    report_path = report_dir / "nonfinite_grad_epoch{:04d}_batch{:04d}.json".format(
        epoch,
        batch_index,
    )
    temporary = report_path.with_suffix(report_path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, sort_keys=True)
        handle.write("\n")
    os.replace(temporary, report_path)
    return report_path, report


def train_one_epoch(
    model,
    loader,
    optimizer,
    device,
    clipnorm,
    clipval,
    label_smoothing=0.0,
    epoch=0,
    workdir=".",
    logger=None,
    log_interval=0,
):
    model.train()
    loss_sum = 0.0
    correct = 0
    count = 0
    max_preclip_grad_norm = 0.0
    max_preclip_grad_norm_batch = None
    for batch_index, (data, target) in enumerate(loader):
        data = data.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        output = model(data)
        loss = F.cross_entropy(
            output,
            target,
            label_smoothing=label_smoothing,
        )
        if not torch.isfinite(loss).item():
            poisoned = [
                name
                for name, parameter in model.named_parameters()
                if not torch.isfinite(parameter).all().item()
            ]
            raise FloatingPointError(
                "Non-finite training loss at batch {}. "
                "Non-finite parameters before backward: {}".format(
                    batch_index,
                    poisoned[:8] if poisoned else "none",
                )
            )
        loss.backward()
        if clipnorm > 0:
            try:
                total_norm = torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    clipnorm,
                    error_if_nonfinite=True,
                )
            except RuntimeError as error:
                report_path, report = _write_nonfinite_gradient_report(
                    model,
                    workdir,
                    epoch,
                    batch_index,
                    loss,
                    output,
                    clipnorm,
                    error,
                )
                raise FloatingPointError(
                    "Non-finite gradient norm at epoch {}, batch {}. "
                    "Diagnosis: {}. Report: {}".format(
                        epoch,
                        batch_index,
                        report["diagnosis"],
                        report_path,
                    )
                ) from error
            total_norm_value = float(total_norm.detach().item())
            if total_norm_value > max_preclip_grad_norm:
                max_preclip_grad_norm = total_norm_value
                max_preclip_grad_norm_batch = batch_index
        if clipval > 0:
            torch.nn.utils.clip_grad_value_(model.parameters(), clipval)
        optimizer.step()

        batch_size = data.size(0)
        loss_sum += loss.item() * batch_size
        correct += (output.argmax(dim=1) == target).sum().item()
        count += batch_size
        completed_batches = batch_index + 1
        if logger is not None and log_interval > 0 and (
            completed_batches % log_interval == 0
            or completed_batches == len(loader)
        ):
            logger.info(
                "Epoch %d batch %d/%d train_loss: %.6f train_acc: %.4f",
                epoch,
                completed_batches,
                len(loader),
                loss_sum / count,
                correct / float(count),
            )
    gradient_summary = {
        "max_preclip_norm": max_preclip_grad_norm,
        "max_preclip_norm_batch": max_preclip_grad_norm_batch,
    }
    return loss_sum, correct, count, gradient_summary


def evaluate(model, loader, device):
    model.eval()
    loss_sum = 0.0
    correct = 0
    count = 0
    with torch.no_grad():
        for data, target in loader:
            data = data.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)
            output = model(data)
            loss = F.cross_entropy(output, target)
            batch_size = data.size(0)
            loss_sum += loss.item() * batch_size
            correct += (output.argmax(dim=1) == target).sum().item()
            count += batch_size
    if count == 0:
        return 0.0, 0.0
    return loss_sum / count, correct / float(count)


def _atomic_torch_save(value, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(value, temporary)
    os.replace(temporary, path)
    return str(path)


def save_checkpoint(state, workdir, filename):
    return _atomic_torch_save(
        state,
        Path(workdir) / "chkpts" / filename,
    )


def update_phase_metrics(workdir, phase, metrics):
    path = Path(workdir) / PHASE_METRICS_FILENAME
    all_metrics = {}
    if path.is_file():
        with path.open("r", encoding="utf-8") as handle:
            all_metrics = json.load(handle)
    all_metrics["phase{}".format(phase)] = metrics
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(all_metrics, handle, indent=2, sort_keys=True)
        handle.write("\n")
    os.replace(temporary, path)
    return str(path)


def checkpoint_payload(
    model,
    optimizer,
    args,
    epoch,
    metrics,
):
    payload = {
        "phase": args.phase,
        "model": unwrap_model(model).state_dict(),
        "optimizer": optimizer.state_dict(),
        "epoch": epoch,
        "args": vars(args),
        "metrics": metrics,
    }
    return payload


def _save_histories(workdir, phase, histories):
    for name, values in histories.items():
        path = Path(workdir) / "phase{}_{}.txt".format(phase, name)
        np.savetxt(path, np.asarray(values, dtype=np.float64))


def _resolve_initial_checkpoint(args):
    if args.train_from_scratch:
        if args.checkpoint is not None:
            raise ValueError(
                "--train-from-scratch and --checkpoint cannot be combined"
            )
        return None, None
    if args.checkpoint is not None:
        expected_phase = {2: 1, 3: 2}.get(args.phase)
        return args.checkpoint, expected_phase
    if args.phase == 2:
        return phase_checkpoint_path(args.workdir, 1), 1
    if args.phase == 3:
        return phase_checkpoint_path(args.workdir, 2), 2
    return None, None


def _compile_model(model, args, logger, is_main):
    if not args.compile:
        return model
    if not hasattr(torch, "compile"):
        if is_main:
            logger.warning("torch.compile is unavailable; using eager mode.")
        return model
    backend = args.compile_backend
    if backend == "inductor":
        backend = "aot_eager"
        if is_main:
            logger.info(
                "Complex operators are routed to compile backend aot_eager."
            )
    compiled = torch.compile(
        model,
        backend=backend,
        mode=args.compile_mode,
    )
    if is_main:
        logger.info(
            "Using torch.compile backend=%s, mode=%s",
            backend,
            args.compile_mode,
        )
    return compiled


def train(args):
    if args.phase not in PHASE_DESCRIPTIONS:
        raise ValueError("Only Phase 1, Phase 2, and Phase 3 are active")
    if args.num_epochs < 1:
        raise ValueError("--num-epochs must be at least 1")
    if args.batch_size < 1:
        raise ValueError("--batch-size must be at least 1")
    if args.lr <= 0.0:
        raise ValueError("--lr must be positive")
    if args.dominance_residual_lr <= 0.0:
        raise ValueError("--dominance-residual-lr must be positive")
    if args.dominance_residual_ramp_epochs < 1:
        raise ValueError("--dominance-residual-ramp-epochs must be at least 1")
    if not 0.0 <= args.dominance_residual_alpha_start <= 1.0:
        raise ValueError("--dominance-residual-alpha-start must be in [0, 1]")
    if not 0.0 <= args.dominance_residual_alpha_end <= 1.0:
        raise ValueError("--dominance-residual-alpha-end must be in [0, 1]")
    if args.dominance_residual_alpha_start > args.dominance_residual_alpha_end:
        raise ValueError("dominance residual alpha start must not exceed end")
    if not 0.0 <= args.min_lr_factor <= 1.0:
        raise ValueError("--min-lr-factor must be between 0 and 1")
    if not 0.0 <= args.label_smoothing < 1.0:
        raise ValueError("--label-smoothing must be in [0, 1)")
    ddp_enabled, rank, local_rank, world_size = init_distributed(args)
    is_main = rank == 0
    if ddp_enabled and torch.cuda.is_available() and not args.cpu:
        torch.cuda.set_device(local_rank)
    set_seed(args.seed + rank)
    logger = setup_logger(args.workdir, args.loglevel, is_main)

    if is_main:
        Path(args.workdir).mkdir(parents=True, exist_ok=True)
        logger.info("INVOCATION: %s", " ".join(sys.argv))
        logger.info("HOSTNAME: %s", socket.gethostname())
        logger.info("PWD: %s", os.getcwd())
        logger.info("Phase %d: %s", args.phase, PHASE_DESCRIPTIONS[args.phase])
        logger.info("Configuration: %s", json.dumps(vars(args), sort_keys=True))
        logger.info(
            "Distributed: enabled=%s rank=%d world_size=%d",
            ddp_enabled,
            rank,
            world_size,
        )

    (
        train_dataset,
        val_dataset,
        test_dataset,
        num_classes,
        pixel_mean,
        train_sampler,
    ) = build_datasets(args, logger, ddp_enabled, is_main)
    if is_main:
        torch.save(pixel_mean, Path(args.workdir) / "pixel_mean.pt")

    pin_memory = torch.cuda.is_available() and not args.cpu
    loader_kwargs = {
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
        "pin_memory": pin_memory,
    }
    train_loader = DataLoader(
        train_dataset,
        shuffle=train_sampler is None,
        sampler=train_sampler,
        **loader_kwargs
    )
    val_loader = None
    test_loader = None
    if is_main:
        if val_dataset is not None:
            val_loader = DataLoader(val_dataset, shuffle=False, **loader_kwargs)
        test_loader = DataLoader(test_dataset, shuffle=False, **loader_kwargs)

    device = torch.device(
        "cuda" if torch.cuda.is_available() and not args.cpu else "cpu"
    )
    model = build_model(args, num_classes).to(device)
    checkpoint_path, expected_phase = _resolve_initial_checkpoint(args)
    if checkpoint_path is not None:
        if not os.path.isfile(checkpoint_path):
            raise FileNotFoundError(
                "Initialization checkpoint not found: {}".format(checkpoint_path)
            )
        if args.phase == 3:
            load_phase3_shared_checkpoint(model, checkpoint_path, device=device)
        else:
            load_phase_checkpoint(
                model,
                checkpoint_path,
                device=device,
                expected_phase=expected_phase,
            )
        if is_main:
            logger.info("Initialized Phase %d from %s", args.phase, checkpoint_path)
    elif is_main:
        logger.info("Training Phase %d from scratch.", args.phase)

    if is_main:
        logger.info(
            "Model parameters: %d",
            sum(parameter.numel() for parameter in model.parameters()),
        )
        if args.phase == 3:
            lut_operators = sum(
                isinstance(
                    module,
                    (TwoLUTComplexConv2d, PairLUT4ComplexConv2d, TripleLUT6ComplexConv2d),
                )
                for module in model.modules()
            )
            logger.info(
                "Phase 3 operator=%s count=%d",
                args.phase3_operator,
                lut_operators,
            )
            logger.info(
                "PairLUT4 parameterization=%s inputs=%d encoding=%s dominance_grad=%s",
                args.pair_lut_parameterization,
                args.pair_lut_inputs,
                args.pair_lut_encoding,
                args.dominance_grad_mode,
            )
            logger.info("Phase 3 LUT training flow: fully hard with STE")
        if args.summary:
            logger.info("Model:\n%s", model)

    model = _compile_model(model, args, logger, is_main)
    if ddp_enabled:
        if device.type == "cuda":
            model = DDP(model, device_ids=[local_rank], output_device=local_rank)
        else:
            model = DDP(model)
    optimizer = build_optimizer(args, model)

    histories = {
        "train_loss": [],
        "train_acc": [],
        "val_loss": [],
        "val_acc": [],
        "test_loss": [],
        "test_acc": [],
    }
    best_acc = -math.inf
    best_metrics = None
    previous_lr = None
    previous_lut_stage = None
    previous_hard_operators = None
    if is_main:
        logger.info("Entering training loop.")

    for epoch in range(args.num_epochs):
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)
        current_lr = learning_rate_for_epoch(epoch, args)
        set_optimizer_lr(optimizer, current_lr)
        if is_main and (
            previous_lr is None or not math.isclose(current_lr, previous_lr)
        ):
            logger.info("Epoch %d learning rate: %.8g", epoch + 1, current_lr)
        previous_lr = current_lr

        lut_state = configure_lut_training_flow(
            unwrap_model(model),
            epoch,
            args,
        )
        if is_main and (
            lut_state["stage"] != previous_lut_stage
            or lut_state["hard_operators"] != previous_hard_operators
        ):
            logger.info(
                (
                    "Epoch %d LUT flow: stage=%s tau=%s alpha=%s hard_operators=%d/%d"
                ),
                epoch + 1,
                lut_state["stage"],
                (
                    "n/a"
                    if lut_state["temperature"] is None
                    else "{:.6g}".format(lut_state["temperature"])
                ),
                (
                    "n/a"
                    if lut_state["dominance_residual_alpha"] is None
                    else "{:.6g}".format(
                        lut_state["dominance_residual_alpha"]
                    )
                ),
                lut_state["hard_operators"],
                lut_state["total_operators"],
            )
        previous_lut_stage = lut_state["stage"]
        previous_hard_operators = lut_state["hard_operators"]

        start = time.time()
        (
            train_loss_sum,
            train_correct,
            train_count,
            gradient_summary,
        ) = train_one_epoch(
            model,
            train_loader,
            optimizer,
            device,
            clipnorm=args.clipnorm,
            clipval=args.clipval,
            label_smoothing=args.label_smoothing,
            epoch=epoch + 1,
            workdir=args.workdir,
            logger=logger if is_main else None,
            log_interval=args.log_interval,
        )
        elapsed = time.time() - start
        if ddp_enabled:
            statistics = torch.tensor(
                [train_loss_sum, train_correct, train_count],
                dtype=torch.float64,
                device=device,
            )
            dist.all_reduce(statistics, op=dist.ReduceOp.SUM)
            train_loss_sum = statistics[0].item()
            train_correct = statistics[1].item()
            train_count = statistics[2].item()
        train_loss = train_loss_sum / max(train_count, 1.0)
        train_acc = train_correct / max(train_count, 1.0)

        if is_main:
            val_loss, val_acc = 0.0, 0.0
            if val_loader is not None:
                val_loss, val_acc = evaluate(unwrap_model(model), val_loader, device)
            test_loss, test_acc = evaluate(unwrap_model(model), test_loader, device)
            histories["train_loss"].append(train_loss)
            histories["train_acc"].append(train_acc)
            histories["val_loss"].append(val_loss)
            histories["val_acc"].append(val_acc)
            histories["test_loss"].append(test_loss)
            histories["test_acc"].append(test_acc)
            logger.info(
                (
                    "Epoch {:5d} train_loss: {:.6f}, train_acc: {:.4f}, "
                    "val_loss: {:.6f}, val_acc: {:.4f}, "
                    "test_loss: {:.6f}, test_acc: {:.4f} ({:.2f}s)"
                ).format(
                    epoch + 1,
                    train_loss,
                    train_acc,
                    val_loss,
                    val_acc,
                    test_loss,
                    test_acc,
                    elapsed,
                )
            )
            if gradient_summary["max_preclip_norm_batch"] is not None:
                logger.info(
                    "Epoch %d maximum pre-clip gradient norm: %.6g at batch %d",
                    epoch + 1,
                    gradient_summary["max_preclip_norm"],
                    gradient_summary["max_preclip_norm_batch"],
                )

            metric_loss, metric_acc = (
                (val_loss, val_acc)
                if val_loader is not None
                else (test_loss, test_acc)
            )
            epoch_metrics = {
                "phase": args.phase,
                "phase_name": PHASE_DESCRIPTIONS[args.phase],
                "epoch": epoch + 1,
                "selection_split": (
                    "validation" if val_loader is not None else "test"
                ),
                "selection_acc": float(metric_acc),
                "selection_loss": float(metric_loss),
                "train_acc": float(train_acc),
                "train_loss": float(train_loss),
                "val_acc": float(val_acc),
                "val_loss": float(val_loss),
                "test_acc": float(test_acc),
                "test_loss": float(test_loss),
                "learning_rate": float(current_lr),
                "lut_training_stage": lut_state["stage"],
                "lut_temperature": lut_state["temperature"],
                "lut_hard_operators": lut_state["hard_operators"],
                "lut_total_operators": lut_state["total_operators"],
                "hardware_ready": lut_state["fully_hard"],
                "dominance_residual_alpha": lut_state[
                    "dominance_residual_alpha"
                ],
            }
            payload = checkpoint_payload(
                model, optimizer, args, epoch + 1, epoch_metrics
            )
            last_path = save_checkpoint(
                payload,
                args.workdir,
                last_checkpoint_filename(args.phase),
            )
            if lut_state["fully_hard"] and metric_acc > best_acc:
                best_acc = metric_acc
                best_metrics = dict(epoch_metrics)
                best_metrics["best_acc"] = float(metric_acc)
                best_metrics["best_loss"] = float(metric_loss)
                payload["metrics"] = best_metrics
                best_path = save_checkpoint(
                    payload,
                    args.workdir,
                    phase_checkpoint_filename(args.phase),
                )
                metrics_path = update_phase_metrics(
                    args.workdir, args.phase, best_metrics
                )
                logger.info(
                    "Saved best Phase %d checkpoint to %s", args.phase, best_path
                )
                logger.info("Updated metrics at %s", metrics_path)
            logger.debug("Saved last checkpoint to %s", last_path)

        if ddp_enabled:
            dist.barrier()

    if is_main:
        _save_histories(args.workdir, args.phase, histories)
        logger.info(
            "Finished Phase %d; best selection accuracy %.4f at epoch %d.",
            args.phase,
            best_metrics["best_acc"],
            best_metrics["epoch"],
        )
    if ddp_enabled and dist.is_initialized():
        dist.destroy_process_group()


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Train the clean Phase 1/2/3 complex Bi-Real route"
    )
    parser.add_argument("-d", "--datadir", default=".", type=str)
    parser.add_argument("-w", "--workdir", default=".", type=str)
    parser.add_argument(
        "-l", "--loglevel", default="info", choices=sorted(LOG_LEVELS)
    )
    parser.add_argument(
        "-s", "--seed", default=0xE4223644E98B8E64, type=int
    )
    parser.add_argument(
        "--dataset",
        default="cifar10",
        choices=["cifar10", "cifar100", "svhn"],
    )
    parser.add_argument(
        "--phase", default=1, type=int, choices=sorted(PHASE_DESCRIPTIONS)
    )
    parser.add_argument(
        "--checkpoint",
        default=None,
        help=(
            "Initialization checkpoint. Phase 2 loads Phase 1; Phase 3 loads "
            "shared state from Phase 2, or dominance LUT6 expands a "
            "categorical LUT4 Phase 3 checkpoint."
        ),
    )
    parser.add_argument(
        "--train-from-scratch",
        action="store_true",
        help="Skip checkpoint initialization",
    )
    parser.add_argument("-n", "--num-epochs", default=200, type=int)
    parser.add_argument("-b", "--batch-size", default=128, type=int)
    parser.add_argument("--start-filter", "--sf", default=16, type=int)
    parser.add_argument("--num-blocks", "--nb", default=3, type=int)
    parser.add_argument("--spectral-pool-gamma", default=0.5, type=float)
    parser.add_argument(
        "--spectral-pool-scheme",
        default="none",
        choices=["none", "stagemiddle", "proj", "nodownsample"],
    )
    parser.add_argument("--binary-stem", action="store_true")
    parser.add_argument(
        "--binary-weight-scale", default="channel", choices=["channel", "layer"]
    )
    parser.add_argument(
        "--weight-grad-mode", default="ste", choices=["ste", "bireal"]
    )
    parser.add_argument(
        "--activation-grad-mode", default="bireal", choices=["ste", "bireal"]
    )
    parser.add_argument(
        "--phase3-operator",
        default="pair_lut4",
        choices=["pair_lut4", "triple_lut6", "shared_lut6"],
        help="Phase 3 main convolution implementation",
    )
    parser.add_argument(
        "--pair-lut-parameterization",
        default="independent",
        choices=[
            "independent",
            "categorical",
            "categorical_residual",
        ],
        help="PairLUT4 table parameterization",
    )
    parser.add_argument(
        "--pair-lut-inputs",
        default=4,
        type=int,
        choices=[4, 6],
        help="Number of activation bits per grouped complex LUT",
    )
    parser.add_argument(
        "--pair-lut-encoding",
        default="standard",
        choices=["standard", "dominance"],
        help=(
            "standard groups signed real/imag bits; dominance groups two "
            "complex activations as [sr, si, phase/Gray comparator bit]"
        ),
    )
    parser.add_argument(
        "--dominance-grad-mode",
        default="stop",
        choices=["stop", "ste"],
        help="Whether the hard dominance bit propagates a surrogate gradient",
    )

    parser.add_argument(
        "--dominance-residual-lr",
        default=0.002,
        type=float,
        help="Learning rate for zero-mean dominance LUT residuals",
    )
    parser.add_argument(
        "--dominance-residual-alpha-start", default=0.1, type=float
    )
    parser.add_argument(
        "--dominance-residual-alpha-end", default=1.0, type=float
    )
    parser.add_argument(
        "--dominance-residual-ramp-epochs", default=120, type=int
    )
    parser.add_argument(
        "--dominance-ste-margin",
        default=1.0,
        type=float,
        help="Linear STE half-width for the hard |real|>|imag| bit",
    )
    parser.add_argument(
        "--post-bn-mode",
        default="covariance",
        choices=["covariance", "naive", "none"],
    )
    parser.add_argument("--no-validation", action="store_true")
    parser.add_argument(
        "--augmentation",
        default="complex_default",
        choices=["complex_default", "real_lut"],
    )
    parser.add_argument("--label-smoothing", default=0.0, type=float)
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--num-workers", default=4, type=int)
    parser.add_argument("--log-interval", default=0, type=int)
    parser.add_argument("--summary", action="store_true")

    parser.add_argument(
        "--optimizer",
        "--opt",
        default="sgd",
        choices=["sgd", "nag", "adam", "adamw", "rmsprop"],
    )
    parser.add_argument("--lr", default=0.1, type=float)
    parser.add_argument("--momentum", "--mom", default=0.9, type=float)
    parser.add_argument(
        "--weight-decay", "--l2", dest="weight_decay", default=1e-4, type=float
    )
    parser.add_argument("--clipnorm", "--cn", default=1.0, type=float)
    parser.add_argument("--clipval", "--cv", default=1.0, type=float)
    parser.add_argument(
        "--schedule",
        default="bireal",
        choices=[
            "bireal",
            "bireal_reference",
            "multistep",
            "cosine",
            "constant",
            "linear",
        ],
    )
    parser.add_argument("--min-lr-factor", default=0.01, type=float)
    parser.add_argument("--beta1", default=0.9, type=float)
    parser.add_argument("--beta2", default=0.999, type=float)

    parser.add_argument("--ddp", action="store_true")
    parser.add_argument("--ddp-backend", default="nccl")
    parser.add_argument("--local-rank", default=None, type=int)
    parser.add_argument("--compile", action="store_true")
    parser.add_argument("--compile-backend", default="inductor")
    parser.add_argument("--compile-mode", default="default")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(sys.argv[1:] if argv is None else argv)
    train(args)


if __name__ == "__main__":
    main()

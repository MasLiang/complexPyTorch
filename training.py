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
    BinaryComplexConv2d,
    PairLUTNeuronConv2d,
)


PHASE_DESCRIPTIONS = {
    1: "full-precision BiReal",
    2: "binary BiReal",
    3: "pair-LUT neuron network",
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


def initialize_phase3_from_phase2(
    model,
    checkpoint_path,
    device="cpu",
    logit_init=1.0,
):
    """Replace each Phase 2 binary spatial convolution with pair LUTs."""
    checkpoint_path = os.fspath(checkpoint_path)
    print("==> Converting Phase 2 weights from '{}' ...".format(checkpoint_path))
    checkpoint = torch.load(checkpoint_path, map_location=device)
    stored_phase = _checkpoint_phase(checkpoint)
    if stored_phase is not None and stored_phase != 2:
        raise ValueError(
            "Checkpoint {} belongs to Phase {}, but Phase 2 is required.".format(
                checkpoint_path,
                stored_phase,
            )
        )

    base_model = unwrap_model(model)
    source_state = {
        _strip_state_wrappers(key): value
        for key, value in _checkpoint_state_dict(checkpoint).items()
    }
    target_state = base_model.state_dict()
    mapped_state = {}
    shape_mismatches = []
    for target_key, target_value in target_state.items():
        source_value = source_state.get(target_key)
        if source_value is None:
            continue
        if target_value.shape != source_value.shape:
            shape_mismatches.append(
                (
                    target_key,
                    tuple(source_value.shape),
                    tuple(target_value.shape),
                )
            )
            continue
        mapped_state[target_key] = source_value
    if shape_mismatches:
        raise RuntimeError(
            "Checkpoint tensors have incompatible shapes: {}".format(
                shape_mismatches
            )
        )

    layer_reports = {}
    for module_name, module in base_model.named_modules():
        if not isinstance(module, PairLUTNeuronConv2d):
            continue
        real_key = module_name + ".conv_r.weight"
        imag_key = module_name + ".conv_i.weight"
        missing = [
            key for key in (real_key, imag_key) if key not in source_state
        ]
        if missing:
            raise RuntimeError(
                "Phase 2 checkpoint is missing source weights for {}: {}".format(
                    module_name,
                    missing,
                )
            )
        layer_reports[module_name] = module.initialize_from_phase2_weights(
            source_state[real_key],
            source_state[imag_key],
            logit_init=logit_init,
        )
    if not layer_reports:
        raise RuntimeError("Phase 3 model contains no PairLUTNeuronConv2d layers")

    missing_keys, unexpected_keys = base_model.load_state_dict(
        mapped_state,
        strict=False,
    )
    pair_prefixes = tuple(name + "." for name in layer_reports)
    missing_parameters = [
        key
        for key in missing_keys
        if key in dict(base_model.named_parameters())
        and not key.startswith(pair_prefixes)
    ]
    if missing_parameters:
        raise RuntimeError(
            "Checkpoint did not initialize shared Phase 3 parameters: {}".format(
                missing_parameters
            )
        )
    if unexpected_keys:
        raise RuntimeError(
            "Unexpected mapped checkpoint keys: {}".format(unexpected_keys)
        )

    report = {
        "checkpoint": checkpoint_path,
        "source_phase": stored_phase,
        "layers": len(layer_reports),
        "pairs": sum(item["pairs"] for item in layer_reports.values()),
        "odd_tail_layers": sum(
            int(item["odd_tail"]) for item in layer_reports.values()
        ),
        "zero_weight_components": sum(
            item["zero_weight_components"] for item in layer_reports.values()
        ),
        "layer_reports": layer_reports,
    }
    print(
        "==> Converted {} layers into {} pair LUT neurons ({} odd tails).".format(
            report["layers"],
            report["pairs"],
            report["odd_tail_layers"],
        )
    )
    return checkpoint, report


def initialize_phase3_random(
    model,
    logit_std=1.0,
    init_mode="normal",
    bimodal_negative_mean=-1.0,
    bimodal_negative_std=0.2,
    bimodal_positive_mean=1.0,
    bimodal_positive_std=0.1,
):
    """Initialize every pair-LUT neuron independently without a checkpoint."""
    base_model = unwrap_model(model)
    layer_reports = {}
    for module_name, module in base_model.named_modules():
        if isinstance(module, PairLUTNeuronConv2d):
            if init_mode == "normal":
                layer_reports[module_name] = module.initialize_random(
                    logit_std=logit_std
                )
            elif init_mode == "bimodal":
                layer_reports[module_name] = module.initialize_bimodal(
                    negative_mean=bimodal_negative_mean,
                    negative_std=bimodal_negative_std,
                    positive_mean=bimodal_positive_mean,
                    positive_std=bimodal_positive_std,
                )
            else:
                raise ValueError(
                    "Unknown pair-LUT initialization mode: {}".format(
                        init_mode
                    )
                )
    if not layer_reports:
        raise RuntimeError("Phase 3 model contains no PairLUTNeuronConv2d layers")
    report = {
        "mode": "random_{}".format(init_mode),
        "checkpoint": None,
        "layers": len(layer_reports),
        "pairs": sum(item["pairs"] for item in layer_reports.values()),
        "odd_tail_layers": sum(
            int(item["odd_tail"]) for item in layer_reports.values()
        ),
        "logit_std_requested": float(logit_std),
        "bimodal": {
            "negative_mean": float(bimodal_negative_mean),
            "negative_std": float(bimodal_negative_std),
            "positive_mean": float(bimodal_positive_mean),
            "positive_std": float(bimodal_positive_std),
        },
        "layer_reports": layer_reports,
    }
    print(
        "==> Randomly initialized {} layers containing {} pair LUT neurons "
        "using {} logits.".format(
            report["layers"],
            report["pairs"],
            init_mode,
        )
    )
    return report


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
        lut_logit_init=args.lut_logit_init,
        lut_tau_init=args.lut_tau_min,
        lut_training_mode=args.lut_training_mode,
        lut_kernel_mode=args.lut_kernel_mode,
        pre_bn_mode=args.pre_bn_mode,
        post_bn_mode=args.post_bn_mode,
    )


def split_weight_decay_params(model):
    base_model = unwrap_model(model)
    normalization_parameter_ids = set()
    binary_weight_ids = set()
    lut_parameter_ids = set()
    for module in base_model.modules():
        if "BatchNorm" in module.__class__.__name__:
            normalization_parameter_ids.update(
                id(param)
                for param in module.parameters(recurse=False)
            )
        if isinstance(module, BinaryComplexConv2d):
            binary_weight_ids.update(
                id(param)
                for param in module.parameters(recurse=False)
            )
            binary_weight_ids.update(
                id(param)
                for child in module.children()
                for param in child.parameters(recurse=False)
            )
        if isinstance(module, PairLUTNeuronConv2d):
            lut_parameter_ids.update(
                id(param)
                for param in module.parameters(recurse=False)
            )

    decay = []
    no_decay = []
    lut = []
    for name, parameter in base_model.named_parameters():
        if not parameter.requires_grad:
            continue
        if id(parameter) in lut_parameter_ids:
            lut.append(parameter)
            continue
        no_decay_parameter = (
            name.endswith(".bias")
            or id(parameter) in normalization_parameter_ids
            or id(parameter) in binary_weight_ids
        )
        (no_decay if no_decay_parameter else decay).append(parameter)
    return decay, no_decay, lut


def build_optimizer(args, model):
    decay_params, no_decay_params, lut_params = split_weight_decay_params(model)
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
    if lut_params:
        parameter_groups.append(
            {
                "name": "lut",
                "params": lut_params,
                "weight_decay": 0.0,
                "lr": args.lut_lr,
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
    if args.schedule == "cosine":
        progress = epoch / float(max(args.num_epochs - 1, 1))
        minimum = args.lr * args.min_lr_factor
        return min(
            args.lr,
            minimum + 0.5 * (args.lr - minimum) * (
                1.0 + math.cos(math.pi * progress)
            ),
        )
    if args.schedule == "linear":
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


def lut_learning_rate_for_epoch(epoch, args):
    if args.lut_schedule == "constant":
        return args.lut_lr
    if args.lut_schedule == "cosine":
        progress = epoch / float(max(args.num_epochs - 1, 1))
        minimum = args.lut_lr * args.min_lr_factor
        return minimum + 0.5 * (args.lut_lr - minimum) * (
            1.0 + math.cos(math.pi * progress)
        )
    if args.lut_schedule == "linear":
        return args.lut_lr * max(
            1.0 - epoch / float(max(args.num_epochs, 1)),
            0.0,
        )
    raise ValueError("Unknown LUT schedule: {}".format(args.lut_schedule))


def set_optimizer_lr(optimizer, learning_rate, lut_learning_rate=None):
    for group in optimizer.param_groups:
        if group.get("name") == "lut" and lut_learning_rate is not None:
            group["lr"] = lut_learning_rate
        else:
            group["lr"] = learning_rate


def update_pair_lut_annealing(model, epoch, args):
    if args.phase != 3:
        return {
            "tau": None,
            "hard_ratio": None,
            "fully_hard": True,
        }
    if getattr(args, "lut_training_mode", "anneal") == "real_compatible":
        layers = 0
        for module in unwrap_model(model).modules():
            if isinstance(module, PairLUTNeuronConv2d):
                module.tau.fill_(args.lut_tau_max)
                module.hard_ratio.fill_(1.0)
                layers += 1
        if layers == 0:
            raise RuntimeError("Phase 3 model contains no pair-LUT layers")
        return {
            "tau": float(args.lut_tau_max),
            "hard_ratio": 1.0,
            "fully_hard": True,
        }
    if args.lut_anneal_epochs < 0 or args.lut_hard_transition_epochs < 0:
        raise ValueError("LUT anneal durations must be non-negative")
    if args.lut_tau_min <= 0.0 or args.lut_tau_max <= 0.0:
        raise ValueError("LUT tau values must be positive")
    if args.lut_tau_max < args.lut_tau_min:
        raise ValueError("--lut-tau-max must be at least --lut-tau-min")

    if args.lut_anneal_epochs == 0:
        tau = args.lut_tau_max
    else:
        progress = min(
            epoch / float(max(args.lut_anneal_epochs - 1, 1)),
            1.0,
        )
        tau = args.lut_tau_min * (
            args.lut_tau_max / args.lut_tau_min
        ) ** progress

    if epoch < args.lut_anneal_epochs:
        hard_ratio = 0.0
    elif args.lut_hard_transition_epochs == 0:
        hard_ratio = 1.0
    else:
        hard_ratio = min(
            (epoch - args.lut_anneal_epochs + 1)
            / float(args.lut_hard_transition_epochs),
            1.0,
        )

    layers = 0
    for module in unwrap_model(model).modules():
        if isinstance(module, PairLUTNeuronConv2d):
            module.tau.fill_(tau)
            module.hard_ratio.fill_(hard_ratio)
            layers += 1
    if layers == 0:
        raise RuntimeError("Phase 3 model contains no pair-LUT layers")
    return {
        "tau": float(tau),
        "hard_ratio": float(hard_ratio),
        "fully_hard": math.isclose(hard_ratio, 1.0),
    }


def pair_lut_sign_diff(model):
    result = {"real": 0, "imag": 0, "total": 0, "entries": 0}
    for module in unwrap_model(model).modules():
        if not isinstance(module, PairLUTNeuronConv2d):
            continue
        current = module.hard_sign_diff()
        for key in result:
            result[key] += current[key]
    result["fraction"] = (
        result["total"] / float(result["entries"])
        if result["entries"]
        else 0.0
    )
    return result


def capture_pair_lut_tables(model):
    return {
        name: module.hard_tables()
        for name, module in unwrap_model(model).named_modules()
        if isinstance(module, PairLUTNeuronConv2d)
    }


def train_one_epoch(
    model,
    loader,
    optimizer,
    device,
    clipnorm,
    clipval,
    label_smoothing=0.0,
):
    model.train()
    loss_sum = 0.0
    correct = 0
    count = 0
    for data, target in loader:
        data = data.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        output = model(data)
        loss = F.cross_entropy(
            output,
            target,
            label_smoothing=label_smoothing,
        )
        loss.backward()
        if clipnorm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), clipnorm)
        if clipval > 0:
            torch.nn.utils.clip_grad_value_(model.parameters(), clipval)
        optimizer.step()

        batch_size = data.size(0)
        loss_sum += loss.item() * batch_size
        correct += (output.argmax(dim=1) == target).sum().item()
        count += batch_size
    return loss_sum, correct, count


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
    if args.phase == 3:
        payload["hard_lut_tables"] = capture_pair_lut_tables(model)
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
    if not 0.0 <= args.min_lr_factor <= 1.0:
        raise ValueError("--min-lr-factor must be between 0 and 1")
    if not 0.0 <= args.label_smoothing < 1.0:
        raise ValueError("--label-smoothing must be in [0, 1)")
    if args.phase == 3:
        if args.lut_lr <= 0.0:
            raise ValueError("--lut-lr must be positive")
        if args.lut_logit_init <= 0.0:
            raise ValueError("--lut-logit-init must be positive")
        if args.lut_anneal_epochs < 0:
            raise ValueError("--lut-anneal-epochs must be non-negative")
        if args.lut_hard_transition_epochs < 0:
            raise ValueError(
                "--lut-hard-transition-epochs must be non-negative"
            )
        if args.lut_tau_min <= 0.0 or args.lut_tau_max <= 0.0:
            raise ValueError("LUT tau values must be positive")
        if args.lut_tau_max < args.lut_tau_min:
            raise ValueError(
                "--lut-tau-max must be at least --lut-tau-min"
            )
        if args.lut_training_mode == "anneal":
            epochs_to_hard = args.lut_anneal_epochs + max(
                args.lut_hard_transition_epochs,
                1,
            )
            if args.num_epochs < epochs_to_hard:
                raise ValueError(
                    "Phase 3 needs at least {} epochs to reach fully hard "
                    "mode; got {}".format(epochs_to_hard, args.num_epochs)
                )

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
        logger.info(
            "Phase %d: %s",
            args.phase,
            PHASE_DESCRIPTIONS[args.phase],
        )
        logger.info(
            "Configuration: %s",
            json.dumps(vars(args), sort_keys=True),
        )
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
    ) = build_datasets(
        args,
        logger,
        ddp_enabled,
        is_main,
    )
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
            val_loader = DataLoader(
                val_dataset,
                shuffle=False,
                **loader_kwargs
            )
        test_loader = DataLoader(
            test_dataset,
            shuffle=False,
            **loader_kwargs
        )

    device = torch.device(
        "cuda"
        if torch.cuda.is_available() and not args.cpu
        else "cpu"
    )
    model = build_model(args, num_classes).to(device)
    phase3_initialization = None
    checkpoint_path, expected_phase = _resolve_initial_checkpoint(args)
    if checkpoint_path is not None:
        if not os.path.isfile(checkpoint_path):
            raise FileNotFoundError(
                "Initialization checkpoint not found: {}".format(
                    checkpoint_path
                )
            )
        if args.phase == 3:
            _, phase3_initialization = initialize_phase3_from_phase2(
                model,
                checkpoint_path,
                device=device,
                logit_init=args.lut_logit_init,
            )
        else:
            load_phase_checkpoint(
                model,
                checkpoint_path,
                device=device,
                expected_phase=expected_phase,
            )
        if is_main:
            logger.info(
                "Initialized Phase %d from %s",
                args.phase,
                checkpoint_path,
            )
    else:
        if args.phase == 3:
            phase3_initialization = initialize_phase3_random(
                model,
                logit_std=args.lut_logit_init,
                init_mode=args.lut_init_mode,
                bimodal_negative_mean=args.lut_bimodal_negative_mean,
                bimodal_negative_std=args.lut_bimodal_negative_std,
                bimodal_positive_mean=args.lut_bimodal_positive_mean,
                bimodal_positive_std=args.lut_bimodal_positive_std,
            )
        if is_main:
            logger.info("Training Phase %d from scratch.", args.phase)

    if is_main:
        logger.info(
            "Model parameters: %d",
            sum(parameter.numel() for parameter in model.parameters()),
        )
        if args.summary:
            logger.info("Model:\n%s", model)

    model = _compile_model(model, args, logger, is_main)
    if ddp_enabled:
        if device.type == "cuda":
            model = DDP(
                model,
                device_ids=[local_rank],
                output_device=local_rank,
            )
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
    previous_lut_lr = None
    if is_main:
        logger.info("Entering training loop.")

    for epoch in range(args.num_epochs):
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)
        current_lr = learning_rate_for_epoch(epoch, args)
        current_lut_lr = lut_learning_rate_for_epoch(epoch, args)
        set_optimizer_lr(optimizer, current_lr, current_lut_lr)
        annealing = update_pair_lut_annealing(model, epoch, args)
        if is_main and (
            previous_lr is None
            or not math.isclose(current_lr, previous_lr)
        ):
            logger.info(
                "Epoch %d learning rate: %.8g",
                epoch + 1,
                current_lr,
            )
        previous_lr = current_lr
        if is_main and args.phase == 3 and (
            previous_lut_lr is None
            or not math.isclose(current_lut_lr, previous_lut_lr)
        ):
            logger.info(
                "Epoch %d LUT learning rate: %.8g",
                epoch + 1,
                current_lut_lr,
            )
        if is_main and args.phase == 3 and (
            epoch == 0
            or epoch % 10 == 0
            or annealing["fully_hard"]
        ):
            logger.info(
                "[Pair-LUT Annealing] Epoch %d: tau=%.6f, hard_ratio=%.4f, fully_hard=%s",
                epoch + 1,
                annealing["tau"],
                annealing["hard_ratio"],
                annealing["fully_hard"],
            )
        previous_lut_lr = current_lut_lr

        start = time.time()
        train_loss_sum, train_correct, train_count = train_one_epoch(
            model,
            train_loader,
            optimizer,
            device,
            clipnorm=args.clipnorm,
            clipval=args.clipval,
            label_smoothing=args.label_smoothing,
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

        val_loss, val_acc = 0.0, 0.0
        test_loss, test_acc = 0.0, 0.0
        evaluation_model = unwrap_model(model)
        if is_main:
            if val_loader is not None:
                val_loss, val_acc = evaluate(
                    evaluation_model,
                    val_loader,
                    device,
                )
            test_loss, test_acc = evaluate(
                evaluation_model,
                test_loader,
                device,
            )
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

            metric_loss, metric_acc = (
                (val_loss, val_acc)
                if val_loader is not None
                else (test_loss, test_acc)
            )
            sign_diff = (
                pair_lut_sign_diff(evaluation_model)
                if args.phase == 3
                else None
            )
            if sign_diff is not None and (
                epoch == 0
                or epoch % 10 == 0
                or annealing["fully_hard"]
            ):
                logger.info(
                    "[Pair-LUT Sign Diff] %d/%d (%.6f), real=%d, imag=%d",
                    sign_diff["total"],
                    sign_diff["entries"],
                    sign_diff["fraction"],
                    sign_diff["real"],
                    sign_diff["imag"],
                )
            epoch_metrics = {
                "phase": args.phase,
                "phase_name": PHASE_DESCRIPTIONS[args.phase],
                "epoch": epoch + 1,
                "selection_split": (
                    "validation"
                    if val_loader is not None
                    else "test"
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
            }
            if args.phase == 3:
                epoch_metrics.update(
                    {
                        "lut_learning_rate": float(current_lut_lr),
                        "lut_tau": annealing["tau"],
                        "lut_hard_ratio": annealing["hard_ratio"],
                        "lut_fully_hard": annealing["fully_hard"],
                        "lut_sign_diff": sign_diff,
                    }
                )
            payload = checkpoint_payload(
                model,
                optimizer,
                args,
                epoch + 1,
                epoch_metrics,
            )
            if phase3_initialization is not None:
                payload["phase3_initialization"] = phase3_initialization
            last_path = save_checkpoint(
                payload,
                args.workdir,
                last_checkpoint_filename(args.phase),
            )
            can_select_best = (
                args.phase != 3 or annealing["fully_hard"]
            )
            if can_select_best and metric_acc > best_acc:
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
                    args.workdir,
                    args.phase,
                    best_metrics,
                )
                logger.info(
                    "Saved best Phase %d checkpoint to %s",
                    args.phase,
                    best_path,
                )
                logger.info("Updated metrics at %s", metrics_path)
            logger.debug("Saved last checkpoint to %s", last_path)

        if ddp_enabled:
            dist.barrier()

    if is_main:
        _save_histories(args.workdir, args.phase, histories)
        if best_metrics is None:
            raise RuntimeError(
                "No deployable best checkpoint was produced; Phase 3 must reach fully hard mode"
            )
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
        description="Train the active Phase 1/2/3 complex ResNet route"
    )
    parser.add_argument("-d", "--datadir", default=".", type=str)
    parser.add_argument("-w", "--workdir", default=".", type=str)
    parser.add_argument(
        "-l",
        "--loglevel",
        default="info",
        choices=sorted(LOG_LEVELS),
    )
    parser.add_argument(
        "-s",
        "--seed",
        default=0xE4223644E98B8E64,
        type=int,
    )
    parser.add_argument(
        "--dataset",
        default="cifar10",
        choices=["cifar10", "cifar100", "svhn"],
    )
    parser.add_argument(
        "--phase",
        default=1,
        type=int,
        choices=sorted(PHASE_DESCRIPTIONS),
    )
    parser.add_argument(
        "--checkpoint",
        default=None,
        help=(
            "Initialization checkpoint; Phase 2 expects Phase 1 and "
            "Phase 3 expects Phase 2"
        ),
    )
    parser.add_argument(
        "--train-from-scratch",
        action="store_true",
        help="Skip checkpoint initialization and randomly initialize the model",
    )
    parser.add_argument("-n", "--num-epochs", default=200, type=int)
    parser.add_argument("-b", "--batch-size", default=128, type=int)
    parser.add_argument("--start-filter", "--sf", default=11, type=int)
    parser.add_argument("--num-blocks", "--nb", default=3, type=int)
    parser.add_argument(
        "--spectral-pool-gamma",
        default=0.5,
        type=float,
    )
    parser.add_argument(
        "--spectral-pool-scheme",
        default="none",
        choices=["none", "stagemiddle", "proj", "nodownsample"],
    )
    parser.add_argument("--binary-stem", action="store_true")
    parser.add_argument(
        "--binary-weight-scale",
        default="channel",
        choices=["channel", "layer"],
    )
    parser.add_argument(
        "--weight-grad-mode",
        default="ste",
        choices=["ste", "bireal"],
    )
    parser.add_argument(
        "--activation-grad-mode",
        default="bireal",
        choices=["ste", "bireal"],
    )
    parser.add_argument(
        "--pre-bn-mode",
        default="covariance",
        choices=["covariance", "naive", "none"],
        help="Complex normalization before each residual activation",
    )
    parser.add_argument(
        "--post-bn-mode",
        default="covariance",
        choices=["covariance", "naive", "none"],
        help="Complex normalization after main and projection convolutions",
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
        "--weight-decay",
        "--l2",
        dest="weight_decay",
        default=1e-4,
        type=float,
    )
    parser.add_argument("--clipnorm", "--cn", default=1.0, type=float)
    parser.add_argument("--clipval", "--cv", default=1.0, type=float)
    parser.add_argument(
        "--schedule",
        default="bireal",
        choices=["bireal", "cosine", "constant", "linear"],
    )
    parser.add_argument("--min-lr-factor", default=0.01, type=float)
    parser.add_argument("--lut-lr", default=0.01, type=float)
    parser.add_argument(
        "--lut-schedule",
        default="constant",
        choices=["constant", "cosine", "linear"],
    )
    parser.add_argument(
        "--lut-training-mode",
        default="anneal",
        choices=["anneal", "real_compatible"],
    )
    parser.add_argument(
        "--lut-kernel-mode",
        default="auto",
        choices=["auto", "floating", "binary"],
    )
    parser.add_argument(
        "--lut-init-mode",
        default="normal",
        choices=["normal", "bimodal"],
    )
    parser.add_argument("--lut-logit-init", default=1.0, type=float)
    parser.add_argument("--lut-bimodal-negative-mean", default=-1.0, type=float)
    parser.add_argument("--lut-bimodal-negative-std", default=0.2, type=float)
    parser.add_argument("--lut-bimodal-positive-mean", default=1.0, type=float)
    parser.add_argument("--lut-bimodal-positive-std", default=0.1, type=float)
    parser.add_argument("--lut-tau-min", default=0.5, type=float)
    parser.add_argument("--lut-tau-max", default=10.0, type=float)
    parser.add_argument("--lut-anneal-epochs", default=160, type=int)
    parser.add_argument(
        "--lut-hard-transition-epochs",
        default=40,
        type=int,
    )
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

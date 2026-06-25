#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import logging
import os
import json
import random
import socket
import sys
import time
import math

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Subset
from torch.utils.data.distributed import DistributedSampler
from torchvision import datasets, transforms

from complexPyTorch.complexBinaryResNet import BinaryComplexResNet


LOGLEVELS = {
    "none": logging.NOTSET,
    "debug": logging.DEBUG,
    "info": logging.INFO,
    "warn": logging.WARN,
    "err": logging.ERROR,
    "crit": logging.CRITICAL,
}

PHASE_DESCRIPTIONS = {
    1: "full-precision BiReal",
    2: "binary BiReal",
    3: "LUT-aware BNN",
    4: "LUTNN",
}

PHASE_CHECKPOINT_TEMPLATE = "Bestmodel_phase{}.pt"
LEGACY_BEST_CHECKPOINT_FILENAME = "Bestmodel.pt"
PHASE_METRICS_FILENAME = "phase_metrics.json"


def phase_checkpoint_filename(phase):
    return PHASE_CHECKPOINT_TEMPLATE.format(phase)


def phase_checkpoint_path(workdir, phase):
    return os.path.join(workdir, "chkpts", phase_checkpoint_filename(phase))


def legacy_best_checkpoint_path(workdir):
    return os.path.join(workdir, "chkpts", LEGACY_BEST_CHECKPOINT_FILENAME)


def load_phase_checkpoint(model, checkpoint_path, device='cpu', expected_phase=None):
    print(f"==> Loading and Mapping weights from '{checkpoint_path}'...")
    checkpoint = torch.load(checkpoint_path, map_location=device)

    stored_phase = checkpoint.get("phase") if isinstance(checkpoint, dict) else None
    if stored_phase is None and isinstance(checkpoint, dict):
        metrics = checkpoint.get("metrics")
        if isinstance(metrics, dict):
            stored_phase = metrics.get("phase")
    if stored_phase is None and isinstance(checkpoint, dict):
        checkpoint_args = checkpoint.get("args")
        if isinstance(checkpoint_args, dict):
            stored_phase = checkpoint_args.get("phase")
    if expected_phase is not None and stored_phase is not None and int(stored_phase) != expected_phase:
        raise ValueError(
            "Checkpoint {} belongs to Phase {}, but Phase {} is required.".format(
                checkpoint_path, stored_phase, expected_phase
            )
        )

    state_dict = checkpoint['model'] if 'model' in checkpoint else (checkpoint['state_dict'] if 'state_dict' in checkpoint else checkpoint)
    target_state = model.state_dict()
    new_state_dict = {}
    skipped_keys = []

    def strip_wrappers(key):
        changed = True
        while changed:
            changed = False
            for prefix in ('_orig_mod.', 'module.'):
                if key.startswith(prefix):
                    key = key[len(prefix):]
                    changed = True
        return key

    clean_target_to_actual = {}
    for target_key in target_state.keys():
        clean_target_to_actual[strip_wrappers(target_key)] = target_key

    def phase3_to_phase4_candidates(key):
        candidates = [key]
        if '.conv.conv_r.weight' in key:
            candidates.append(key.replace('.conv.conv_r.weight', '.conv.weight_r'))
        if '.conv.conv_i.weight' in key:
            candidates.append(key.replace('.conv.conv_i.weight', '.conv.weight_i'))
        return candidates

    for k, v in state_dict.items():
        clean_k = strip_wrappers(k)
        mapped_key = None
        mapped_value = v

        for candidate in phase3_to_phase4_candidates(clean_k):
            actual_target_key = clean_target_to_actual.get(candidate)
            if actual_target_key is None:
                continue

            target_value = target_state[actual_target_key]
            if target_value.shape == v.shape:
                mapped_key = actual_target_key
                break
            if (
                candidate.endswith((".lut_r", ".lut_i"))
                and target_value.ndim == 2
                and target_value.shape[1] == 16
                and (v.shape == torch.Size([16]) or v.shape == torch.Size([1, 16]))
            ):
                mapped_key = actual_target_key
                source_lut = v.unsqueeze(0) if v.ndim == 1 else v
                mapped_value = source_lut.expand_as(target_value).clone()
                break

        if mapped_key is None:
            skipped_keys.append(clean_k)
            continue

        new_state_dict[mapped_key] = mapped_value

    missing_keys, unexpected_keys = model.load_state_dict(new_state_dict, strict=False)

    print(f"==> Checkpoint mapped and loaded! ({len(new_state_dict)}/{len(state_dict)} tensors used)")

    if missing_keys:
        expected_missing_keywords = ['lut_r', 'lut_i', 'lut_set_ids', 'flat_c', 'flat_dy', 'flat_dx', 'shifts', 'tau', 'hard']
        unexpected_missing = [
            k for k in missing_keys
            if not any(keyword in k for keyword in expected_missing_keywords)
        ]

        if unexpected_missing:
            print(f"[Warning] Found unexpected missing keys in model:\n{unexpected_missing}")
        else:
            print(f"[Info] All missing keys are LUT-specific initialized buffers/parameters as expected.")

    if unexpected_keys:
        print(f"[Warning] Found unexpected keys after loading mapped checkpoint:\n{unexpected_keys}")

    if skipped_keys:
        preview = skipped_keys[:20]
        suffix = '...' if len(skipped_keys) > len(preview) else ''
        print(f"[Info] Skipped {len(skipped_keys)} checkpoint tensors with no compatible target key/shape: {preview}{suffix}")

    return model


def load_bnn_to_lut_model(model, checkpoint_path, device='cpu'):
    return load_phase_checkpoint(model, checkpoint_path, device=device)


class SubtractMean(object):
    def __init__(self, mean):
        self.mean = mean

    def __call__(self, tensor):
        return tensor - self.mean


class SVHNDataset(datasets.SVHN):
    def __getitem__(self, index):
        image, target = super().__getitem__(index)
        if target == 10:
            target = 0
        return image, target


def setup_logging(workdir, loglevel, is_main):
    if not is_main:
        logging.basicConfig(level=logging.ERROR)
        train_logger = logging.getLogger("train")
        entry_logger = logging.getLogger("entry")
        null_handler = logging.NullHandler()
        train_logger.addHandler(null_handler)
        entry_logger.addHandler(null_handler)
        return entry_logger, train_logger

    if not os.path.isdir(workdir):
        os.makedirs(workdir)
    logdir = os.path.join(workdir, "logs")
    if not os.path.isdir(logdir):
        os.makedirs(logdir)

    formatter = logging.Formatter(
        "[%(asctime)s ~~ %(levelname)-8s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    stdout_handler = logging.StreamHandler(sys.stdout)
    stdout_handler.setLevel(LOGLEVELS[loglevel])
    stdout_handler.setFormatter(formatter)

    train_handler = logging.FileHandler(
        os.path.join(logdir, "train.txt"), mode="a", encoding="utf-8"
    )
    train_handler.setLevel(LOGLEVELS[loglevel])
    train_handler.setFormatter(formatter)

    entry_handler = logging.FileHandler(
        os.path.join(logdir, "entry.txt"), mode="a", encoding="utf-8"
    )
    entry_handler.setLevel(LOGLEVELS[loglevel])
    entry_handler.setFormatter(formatter)

    logging.basicConfig(level=LOGLEVELS[loglevel], handlers=[stdout_handler])

    train_logger = logging.getLogger("train")
    train_logger.setLevel(LOGLEVELS[loglevel])
    train_logger.addHandler(train_handler)

    entry_logger = logging.getLogger("entry")
    entry_logger.setLevel(LOGLEVELS[loglevel])
    entry_logger.addHandler(entry_handler)

    return entry_logger, train_logger


def init_distributed(args):
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    use_ddp = args.ddp or world_size > 1
    if not use_ddp:
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
    torch.cuda.manual_seed_all(seed)


def summarize_envvar(var):
    if var in os.environ:
        return "{}={}".format(var, os.environ.get(var))
    return "{} unset".format(var)


def compute_pixel_mean(dataset, batch_size=256, num_workers=2):
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    pixel_sum = None
    count = 0
    for data, _ in loader:
        if pixel_sum is None:
            pixel_sum = torch.zeros_like(data[0])
        pixel_sum += data.sum(dim=0)
        count += data.size(0)
    return (pixel_sum / float(count)).type(torch.float32)


def split_weight_decay_params(model):
    decay = []
    no_decay = []
    
    # 获取模型是否为二值化模型的标志
    is_binary_model = 'Binary' in model.__class__.__name__

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
            
        # 1. Bias 和 BatchNorm 参数绝对不加正则化
        if len(param.shape) == 1 or param.ndim == 1:
            no_decay.append(param)
        else:
            lut_keywords = ['conv_r.weight', 'conv_i.weight', 'weight_r', 'weight_i', 'lut_r', 'lut_i']
            if is_binary_model and any(kw in name for kw in lut_keywords):
                no_decay.append(param)
            else:
                # 全精度权重（或者 Stem 层的权重）正常加正则化
                decay.append(param)

    return decay, no_decay


def get_lr_for_epoch(epoch, args):
    if args.schedule == "bireal":
        warmup_epochs = 5
        if epoch < warmup_epochs:
            return args.lr * float(epoch + 1) / float(warmup_epochs)
        t1 = int(args.num_epochs * 0.5)
        t2 = int(args.num_epochs * 0.75)
        t3 = int(args.num_epochs * 0.875)
        if epoch < t1:
            return args.lr
        if epoch < t2:
            return args.lr * 0.1
        if epoch < t3:
            return args.lr * 0.01
        return args.lr * 0.001

    if epoch < 10:
        return 0.01
    if epoch < 100:
        return 0.1
    if epoch < 120:
        return 0.01
    if epoch < 150:
        return 0.001
    return 0.0001


def set_optimizer_lr(optimizer, lr):
    for group in optimizer.param_groups:
        group["lr"] = lr


def build_datasets(args, train_logger, ddp_enabled, is_main):
    if args.dataset == "cifar10":
        dataset_cls = datasets.CIFAR10
        n_train = 45000
        n_classes = 10
        train_kwargs = {"train": True}
        test_kwargs = {"train": False}
    elif args.dataset == "cifar100":
        dataset_cls = datasets.CIFAR100
        n_train = 45000
        n_classes = 100
        train_kwargs = {"train": True}
        test_kwargs = {"train": False}
    elif args.dataset == "svhn":
        dataset_cls = SVHNDataset
        n_train = 65000
        n_classes = 10
        train_kwargs = {"split": "train"}
        test_kwargs = {"split": "test"}
    else:
        raise ValueError("Unknown dataset: {}".format(args.dataset))

    if ddp_enabled:
        if is_main:
            base_train = dataset_cls(
                root=args.datadir,
                download=True,
                transform=transforms.ToTensor(),
                **train_kwargs
            )
            dist.barrier()
        else:
            dist.barrier()
            base_train = dataset_cls(
                root=args.datadir,
                download=False,
                transform=transforms.ToTensor(),
                **train_kwargs
            )
    else:
        base_train = dataset_cls(
            root=args.datadir,
            download=True,
            transform=transforms.ToTensor(),
            **train_kwargs
        )

    shuf_inds = np.arange(len(base_train))
    np.random.seed(0xDEADBEEF)
    np.random.shuffle(shuf_inds)
    train_inds = shuf_inds[:n_train]
    val_inds = shuf_inds[n_train:]

    pixel_mean = None
    if is_main:
        pixel_mean = compute_pixel_mean(Subset(base_train, train_inds), batch_size=args.batch_size)
    if ddp_enabled:
        obj_list = [pixel_mean]
        dist.broadcast_object_list(obj_list, src=0)
        pixel_mean = obj_list[0]

    train_transform = transforms.Compose(
        [
            transforms.RandomAffine(degrees=0, translate=(0.125, 0.125)),
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

    train_dataset = dataset_cls(
        root=args.datadir,
        download=False,
        transform=train_transform,
        **train_kwargs
    )
    val_dataset = dataset_cls(
        root=args.datadir,
        download=False,
        transform=eval_transform,
        **train_kwargs
    )
    test_dataset = dataset_cls(
        root=args.datadir,
        download=False,
        transform=eval_transform,
        **test_kwargs
    )

    train_split = Subset(train_dataset, train_inds)
    val_split = Subset(val_dataset, val_inds)

    if train_logger is not None:
        train_logger.info("Training   set size: {}".format(len(train_split)))
        train_logger.info("Validation set size: {}".format(len(val_split)))
        train_logger.info("Test       set size: {}".format(len(test_dataset)))

    if args.no_validation:
        full_train_dataset = dataset_cls(
            root=args.datadir,
            download=False,
            transform=train_transform,
            **train_kwargs
        )
        full_indices = shuf_inds
        full_split = Subset(full_train_dataset, full_indices)
        train_sampler = DistributedSampler(full_split, shuffle=True, seed=args.seed) if ddp_enabled else None
        return full_split, None, test_dataset, n_classes, pixel_mean, train_sampler

    train_sampler = DistributedSampler(train_split, shuffle=True, seed=args.seed) if ddp_enabled else None
    return train_split, val_split, test_dataset, n_classes, pixel_mean, train_sampler


def build_model(args, num_classes):
    if args.phase not in PHASE_DESCRIPTIONS:
        raise ValueError("phase must be one of {}".format(sorted(PHASE_DESCRIPTIONS)))
    if args.lut_sets < 1:
        raise ValueError("lut_sets must be at least 1")
    if args.lut_sets_per_channel < 1:
        raise ValueError("lut_sets_per_channel must be at least 1")

    return BinaryComplexResNet(
        in_channels=3,
        num_blocks=args.num_blocks,
        start_filters=args.start_filter,
        num_classes=num_classes,
        spectral_pool_scheme=args.spectral_pool_scheme,
        spectral_pool_gamma=args.spectral_pool_gamma,
        binary_stem=args.binary_stem,
        is_sar_input=False,
        is_binary=(args.phase >= 2),
        phase=args.phase,
        lut_sets=args.lut_sets,
        lut_allocation=args.lut_allocation,
        lut_sets_per_channel=args.lut_sets_per_channel,
    )


def build_optimizer(args, model):
    base_model = model.module if hasattr(model, "module") else model
    decay_params, no_decay_params = split_weight_decay_params(base_model)
    param_groups = [
        {"params": decay_params, "weight_decay": args.l2},
        {"params": no_decay_params, "weight_decay": 0.0},
    ]

    if args.optimizer in ["sgd", "nag"]:
        return torch.optim.SGD(
            param_groups,
            lr=args.lr,
            momentum=args.momentum,
            nesterov=(args.optimizer == "nag"),
        )
    if args.optimizer == "rmsprop":
        return torch.optim.RMSprop(
            param_groups,
            lr=args.lr,
        )
    if args.optimizer == "adam":
        return torch.optim.Adam(
            param_groups,
            lr=args.lr,
            betas=(args.beta1, args.beta2),
        )
    if args.optimizer == "adamw":
        return torch.optim.AdamW(
            param_groups,
            lr=args.lr,
            betas=(args.beta1, args.beta2),
        )
    raise ValueError("Unknown optimizer: {}".format(args.optimizer))


def train_one_epoch(model, loader, optimizer, device, clipnorm, clipval):
    model.train()
    loss_sum = 0.0
    correct = 0
    count = 0
    for data, target in loader:
        data = data.to(device)
        target = target.to(device)

        optimizer.zero_grad()
        output = model(data)
        loss = F.cross_entropy(output, target)
        loss.backward()

        if clipnorm is not None and clipnorm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), clipnorm)
        if clipval is not None and clipval > 0:
            torch.nn.utils.clip_grad_value_(model.parameters(), clipval)

        optimizer.step()

        loss_sum += loss.item() * data.size(0)
        pred = output.argmax(dim=1)
        correct += (pred == target).sum().item()
        count += data.size(0)

    return loss_sum, correct, count


def evaluate(model, loader, device):
    model.eval()
    loss_sum = 0.0
    correct = 0
    count = 0
    with torch.no_grad():
        for data, target in loader:
            data = data.to(device)
            target = target.to(device)

            output = model(data)
            loss = F.cross_entropy(output, target)
            loss_sum += loss.item() * data.size(0)
            pred = output.argmax(dim=1)
            correct += (pred == target).sum().item()
            count += data.size(0)
    if count == 0:
        return 0.0, 0.0
    return loss_sum / count, correct / float(count)


def save_checkpoint(state, workdir, filename):
    chkpt_dir = os.path.join(workdir, "chkpts")
    if not os.path.isdir(chkpt_dir):
        os.makedirs(chkpt_dir)
    path = os.path.join(chkpt_dir, filename)
    torch.save(state, path)
    return path

def update_lut_annealing(model, epoch, num_epochs, phase, train_logger, is_main):
    if phase != 4:
        return True

    # Phase 4 keeps the soft LUT annealing period for training, but best-checkpoint
    # selection starts only when hard mode is active.
    hard_epoch_start = int(num_epochs * 0.9)
    tau_min, tau_max = 1.0, 10.0

    if epoch < hard_epoch_start:
        ratio = epoch / max(1, hard_epoch_start)
        current_tau = tau_min * math.pow((tau_max / tau_min), ratio)
        is_hard = 0.0
    else:
        current_tau = tau_max
        is_hard = 1.0

    for name, module in model.named_modules():
        if module.__class__.__name__ == "ComplexLUTConv2d" or module.__class__.__name__ == "UltimateComplexLUTConv2d":
            if hasattr(module, "tau") and hasattr(module, "hard"):
                module.tau.fill_(current_tau)
                module.hard.fill_(is_hard)

    if is_main and (epoch == 0 or epoch == hard_epoch_start or epoch % 10 == 0):
        if train_logger is not None:
            train_logger.info(f"[Phase 4 Annealing Scheduler] Epoch {epoch}: tau={current_tau:.4f}, hard_mode={bool(is_hard)}")

    return bool(is_hard)


def update_phase_metrics(workdir, phase, metrics):
    path = os.path.join(workdir, PHASE_METRICS_FILENAME)
    all_metrics = {}
    if os.path.isfile(path):
        with open(path, "r", encoding="utf-8") as f:
            all_metrics = json.load(f)
    all_metrics[f"phase{phase}"] = metrics
    with open(path, "w", encoding="utf-8") as f:
        json.dump(all_metrics, f, indent=2, sort_keys=True)
    return path


def train(args):
    if args.phase not in PHASE_DESCRIPTIONS:
        raise ValueError("phase must be one of {}".format(sorted(PHASE_DESCRIPTIONS)))
    args.binary = args.phase >= 2

    ddp_enabled, rank, local_rank, world_size = init_distributed(args)
    is_main = rank == 0

    if ddp_enabled and torch.cuda.is_available() and not args.cpu:
        torch.cuda.set_device(local_rank)

    entry_logger, train_logger = setup_logging(args.workdir, args.loglevel, is_main)
    set_seed(args.seed)

    if args.bireal_tune:
        args.num_epochs = 400
        args.batch_size = 128
        args.schedule = "bireal"
        args.l2 = 1e-4
        args.decay = 0.0

        if args.optimizer in ["adam", "adamw"]:
            args.lr = 0.001  # Adam 系的微调黄金学习率
            args.momentum = 0.9 # 保留占位符
        else:
            args.lr = 0.1    # SGD 依然用 0.1
            args.optimizer = "sgd"
            args.momentum = 0.9
        if args.phase==4:
            args.lr = args.lr/10

    entry_logger.info("INVOCATION:     " + " ".join(sys.argv))
    entry_logger.info("HOSTNAME:       " + socket.gethostname())
    entry_logger.info("PWD:            " + os.getcwd())

    summary = []
    summary.append("Environment:")
    summary.append(summarize_envvar("CUDA_VISIBLE_DEVICES"))
    summary.append("")
    summary.append("Software Versions:")
    summary.append("Torch:                   " + torch.__version__)
    summary.append("")
    summary.append("Arguments:")
    summary.append("Path to Datasets:        " + str(args.datadir))
    summary.append("Path to Workspace:       " + str(args.workdir))
    summary.append("Model:                   " + str(args.model))
    summary.append("Dataset:                 " + str(args.dataset))
    summary.append("Phase:                   {} ({})".format(args.phase, PHASE_DESCRIPTIONS[args.phase]))
    summary.append("LUT Allocation:          " + str(args.lut_allocation))
    summary.append("LUT Sets per Layer:      " + str(args.lut_sets))
    summary.append("LUT Sets per Channel:    " + str(args.lut_sets_per_channel))
    summary.append("Number of Epochs:        " + str(args.num_epochs))
    summary.append("Batch Size:              " + str(args.batch_size))
    summary.append("Number of Start Filters: " + str(args.start_filter))
    summary.append("Number of Blocks/Stage:  " + str(args.num_blocks))
    summary.append("Dropout Probability:     " + str(args.dropout))
    summary.append("Spectral Param:          " + str(args.spectral_param))
    summary.append("Spectral Pool Gamma:     " + str(args.spectral_pool_gamma))
    summary.append("Spectral Pool Scheme:    " + str(args.spectral_pool_scheme))
    summary.append("Activation:              " + str(args.act))
    summary.append("Advanced Activation:     " + str(args.aact))
    summary.append("Complex Init:            " + str(args.comp_init))
    summary.append("Optimizer:               " + str(args.optimizer))
    summary.append("Learning Rate:           " + str(args.lr))
    summary.append("Learning Rate Decay:     " + str(args.decay))
    summary.append("Learning Rate Schedule:  " + str(args.schedule))
    summary.append("Clipping Norm:           " + str(args.clipnorm))
    summary.append("Clipping Value:          " + str(args.clipval))
    summary.append("L1 Penalty:              " + str(args.l1))
    summary.append("L2 Penalty:              " + str(args.l2))
    summary.append("Compile:                 " + str(args.compile))
    summary.append("BiReal Tune:             " + str(args.bireal_tune))
    entry_logger.info("\n".join(summary))

    if is_main:
        train_logger.info("Loading dataset {} ...".format(args.dataset))
    train_dataset, val_dataset, test_dataset, num_classes, pixel_mean, train_sampler = build_datasets(
        args,
        train_logger if is_main else None,
        ddp_enabled,
        is_main,
    )
    if is_main:
        torch.save(pixel_mean, os.path.join(args.workdir, "pixel_mean.pt"))

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    val_loader = None
    test_loader = None
    if is_main:
        if val_dataset is not None:
            val_loader = DataLoader(
                val_dataset,
                batch_size=args.batch_size,
                shuffle=False,
                num_workers=args.num_workers,
                pin_memory=True,
            )
        test_loader = DataLoader(
            test_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=True,
        )

    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    model = build_model(args, num_classes).to(device)
    print(model)

    effective_compile_backend = args.compile_backend
    if args.compile and effective_compile_backend == "inductor":
        effective_compile_backend = "aot_eager"
        if is_main:
            train_logger.info(
                "Inductor does not support complex operators; falling back to aot_eager."
            )

    if args.compile:
        if hasattr(torch, "compile"):
            model = torch.compile(
                model,
                backend=effective_compile_backend,
                mode=args.compile_mode,
            )
            if is_main:
                train_logger.info(
                    "Using torch.compile backend={}, mode={}".format(
                        effective_compile_backend,
                        args.compile_mode,
                    )
                )
        elif is_main:
            train_logger.info("torch.compile not available in this PyTorch version.")

    if ddp_enabled:
        if device.type == "cuda":
            model = DDP(model, device_ids=[local_rank], output_device=local_rank)
        else:
            model = DDP(model)

    optimizer = build_optimizer(args, model)

    unwrapped_model = model.module if ddp_enabled else model
    if args.summary and is_main:
        entry_logger.info(str(unwrapped_model))
    if is_main:
        entry_logger.info(
            "# of Parameters:              {:10d}".format(
                sum(p.numel() for p in unwrapped_model.parameters())
            )
        )

    initial_epoch = 0

    if args.phase > 1:
        previous_phase = args.phase - 1
        if args.checkpoint is not None:
            ckpt_path = args.checkpoint
            if not os.path.isfile(ckpt_path):
                raise FileNotFoundError("Checkpoint not found: {}".format(ckpt_path))
        else:
            ckpt_path = phase_checkpoint_path(args.workdir, previous_phase)
            legacy_ckpt_path = legacy_best_checkpoint_path(args.workdir)
            if not os.path.isfile(ckpt_path):
                if os.path.isfile(legacy_ckpt_path):
                    ckpt_path = legacy_ckpt_path
                    if is_main:
                        train_logger.info(
                            "[Compatibility] Phase-specific checkpoint missing; using legacy {}.".format(ckpt_path)
                        )
                else:
                    raise FileNotFoundError(
                        "Phase {} expects Phase {} checkpoint at {}".format(
                            args.phase, previous_phase, ckpt_path
                        )
                    )
        if is_main:
            train_logger.info(
                "==> Loading Phase {} initialization for Phase {} from {} ...".format(
                    previous_phase, args.phase, ckpt_path
                )
            )
        load_phase_checkpoint(
            model=unwrapped_model,
            checkpoint_path=ckpt_path,
            device=device,
            expected_phase=previous_phase,
        )
        if is_main:
            train_logger.info("==> Successfully initialized phase {} model from previous best checkpoint.".format(args.phase))


    if is_main:
        train_logger.info("**********************************************")
        if initial_epoch > 0:
            train_logger.info("*** Reentering Training Loop @ Epoch {:5d} ***".format(initial_epoch + 1))
        else:
            train_logger.info("***  Entering Training Loop  @ First Epoch ***")
        train_logger.info("**********************************************")

    train_loss_hist = []
    train_acc_hist = []
    val_loss_hist = []
    val_acc_hist = []
    test_loss_hist = []
    test_acc_hist = []

    best_acc = 0.0

    for epoch in range(initial_epoch, args.num_epochs):
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)
        if args.schedule in ["default", "bireal"]:
            lr = get_lr_for_epoch(epoch, args)
            set_optimizer_lr(optimizer, lr)
            if is_main:
                if args.schedule == "default" and epoch in [0, 10, 100, 120, 150]:
                    train_logger.info("Current learning rate value is {}".format(lr))
                if args.schedule == "bireal":
                    warmup_epochs = 5
                    t1 = int(args.num_epochs * 0.5)
                    t2 = int(args.num_epochs * 0.75)
                    t3 = int(args.num_epochs * 0.875)
                    if epoch in [0, warmup_epochs - 1, t1, t2, t3]:
                        train_logger.info("Current learning rate value is {}".format(lr))

        phase_can_save_best = True
        if args.phase == 4:
            phase_can_save_best = update_lut_annealing(
                model=model, 
                epoch=epoch, 
                num_epochs=args.num_epochs, 
                phase=args.phase, 
                train_logger=train_logger if is_main else None, 
                is_main=is_main
            )
        t0 = time.time()
        train_loss_sum, train_correct, train_count = train_one_epoch(
            model,
            train_loader,
            optimizer,
            device,
            clipnorm=args.clipnorm,
            clipval=args.clipval,
        )
        elapsed = time.time() - t0

        if ddp_enabled:
            stats = torch.tensor(
                [train_loss_sum, train_correct, train_count],
                device=device,
                dtype=torch.float64,
            )
            dist.all_reduce(stats, op=dist.ReduceOp.SUM)
            train_loss_sum = stats[0].item()
            train_correct = stats[1].item()
            train_count = stats[2].item()

        train_loss = train_loss_sum / max(train_count, 1.0)
        train_acc = train_correct / max(train_count, 1.0)

        val_loss, val_acc = (0.0, 0.0)
        if val_loader is not None and is_main:
            val_loss, val_acc = evaluate(model, val_loader, device)

        test_loss, test_acc = (0.0, 0.0)
        if test_loader is not None and is_main:
            test_loss, test_acc = evaluate(model, test_loader, device)

        train_loss_hist.append(train_loss)
        train_acc_hist.append(train_acc)
        val_loss_hist.append(val_loss)
        val_acc_hist.append(val_acc)
        test_loss_hist.append(test_loss)
        test_acc_hist.append(test_acc)

        if is_main:
            train_logger.info(
                "Epoch {:5d} train_loss: {:.6f}, train_acc: {:.4f}, val_loss: {:.6f}, val_acc: {:.4f}, test_loss: {:.6f}, test_acc: {:.4f} ({:.2f}s)".format(
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


        metric_loss, metric_acc = (val_loss, val_acc) if val_loader is not None else (test_loss, test_acc)
        if is_main and phase_can_save_best and metric_acc > best_acc:
            best_acc = metric_acc
            best_metrics = {
                "phase": args.phase,
                "phase_name": PHASE_DESCRIPTIONS[args.phase],
                "epoch": epoch + 1,
                "best_acc": float(metric_acc),
                "best_loss": float(metric_loss),
                "val_acc": float(val_acc),
                "val_loss": float(val_loss),
                "test_acc": float(test_acc),
                "test_loss": float(test_loss),
                "train_acc": float(train_acc),
                "train_loss": float(train_loss),
                "hard_mode": bool(args.phase != 4 or phase_can_save_best),
            }
            checkpoint = {
                "phase": args.phase,
                "model": unwrapped_model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "epoch": epoch + 1,
                "args": vars(args),
                "metrics": best_metrics,
            }
            best_path = save_checkpoint(checkpoint, args.workdir, phase_checkpoint_filename(args.phase))
            metrics_path = update_phase_metrics(args.workdir, args.phase, best_metrics)
            train_logger.info("Saved best model to {} at epoch {:5d}".format(best_path, epoch + 1))
            train_logger.info("Updated phase metrics at {}".format(metrics_path))

        if ddp_enabled:
            dist.barrier()

    if is_main:
        np.savetxt(os.path.join(args.workdir, "train_loss.txt"), np.asarray(train_loss_hist))
        np.savetxt(os.path.join(args.workdir, "train_acc.txt"), np.asarray(train_acc_hist))
        np.savetxt(os.path.join(args.workdir, "val_loss.txt"), np.asarray(val_loss_hist))
        np.savetxt(os.path.join(args.workdir, "val_acc.txt"), np.asarray(val_acc_hist))
        np.savetxt(os.path.join(args.workdir, "test_loss.txt"), np.asarray(test_loss_hist))
        np.savetxt(os.path.join(args.workdir, "test_acc.txt"), np.asarray(test_acc_hist))

    if ddp_enabled and dist.is_initialized():
        dist.destroy_process_group()


def parse_args(argv):
    parser = argparse.ArgumentParser(description="Complex ResNet training")
    parser.add_argument("-d", "--datadir", default=".", type=str)
    parser.add_argument("-w", "--workdir", default=".", type=str)
    parser.add_argument("-l", "--loglevel", default="info", type=str, choices=LOGLEVELS.keys())
    parser.add_argument("-s", "--seed", default=0xE4223644E98B8E64, type=int)
    parser.add_argument("--summary", action="store_true")
    parser.add_argument("--model", default="complex", type=str, choices=["real", "complex"])
    parser.add_argument("--dataset", default="cifar10", type=str, choices=["cifar10", "cifar100", "svhn"])
    parser.add_argument("--dropout", default=0.0, type=float)
    parser.add_argument("-n", "--num-epochs", default=200, type=int)
    parser.add_argument("-b", "--batch-size", default=64, type=int)
    parser.add_argument("--start-filter", "--sf", dest="start_filter", default=11, type=int)
    parser.add_argument("--num-blocks", "--nb", dest="num_blocks", default=3, type=int)
    parser.add_argument("--spectral-pool-gamma", default=0.5, type=float)
    parser.add_argument(
        "--spectral-pool-scheme",
        default="none",
        type=str,
        choices=["none", "stagemiddle", "proj", "nodownsample"],
    )
    parser.add_argument("--spectral-param", action="store_true")
    parser.add_argument("--act", default="relu", type=str, choices=["relu"])
    parser.add_argument("--aact", default="modrelu", type=str, choices=["modrelu"])
    parser.add_argument("--comp_init", default="complex_independent", type=str)
    parser.add_argument("--no-validation", action="store_true")
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--num-workers", default=4, type=int)
    parser.add_argument("--binary", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--binary-stem", action="store_true")
    parser.add_argument("--bireal-tune", action="store_true")
    parser.add_argument("--ddp", action="store_true")
    parser.add_argument("--ddp-backend", default="nccl", type=str)
    parser.add_argument("--local-rank", default=None, type=int)
    parser.add_argument("--compile", action="store_true")
    parser.add_argument("--compile-backend", default="inductor", type=str)
    parser.add_argument("--compile-mode", default="default", type=str)
    parser.add_argument("--phase", default=1, type=int, choices=sorted(PHASE_DESCRIPTIONS))
    parser.add_argument("--checkpoint", default=None, type=str,
                        help="Explicit checkpoint path to initialize phases greater than 1")
    parser.add_argument("--lut-sets", default=1, type=int,
                        help="Number of independently learnable LUT pairs per ComplexLUTConv2d layer")
    parser.add_argument("--lut-allocation", default="layer", choices=["layer", "channel"],
                        help="Allocate LUT sets from a layer-shared pool or from per-output-channel pools")
    parser.add_argument("--lut-sets-per-channel", default=1, type=int,
                        help="Number of independently learnable LUT pairs owned by each output channel when --lut-allocation=channel")

    opt = parser.add_argument_group("Optimizers")
    opt.add_argument("--optimizer", "--opt", default="sgd", type=str, choices=["sgd", "nag", "adam", "adamw", "rmsprop"])
    opt.add_argument("--clipnorm", "--cn", default=1.0, type=float)
    opt.add_argument("--clipval", "--cv", default=1.0, type=float)
    opt.add_argument("--l1", default=0.0, type=float)
    opt.add_argument("--l2", default=0.0, type=float)
    opt.add_argument("--lr", default=1e-3, type=float)
    opt.add_argument("--momentum", "--mom", default=0.9, type=float)
    opt.add_argument("--decay", default=0.0, type=float)
    opt.add_argument("--schedule", default="default", type=str)

    opt = parser.add_argument_group("Adam")
    opt.add_argument("--beta1", default=0.9, type=float)
    opt.add_argument("--beta2", default=0.999, type=float)

    return parser.parse_args(argv)


def main(argv):
    args = parse_args(argv[1:])
    train(args)


if __name__ == "__main__":
    main(sys.argv) 

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
from complexPyTorch.complexLayers import (
    C8ComplexActivation,
    C8LUTAwareComplexQATConv2d,
    ComplexLUTConv2d,
    LUT5AwareComplexQATConv2d,
    LUTAwareComplexBinaryConv2d,
)


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
    2.1: "3-bit activation QAT (C8 or learned LUT5)",
    3: "LUT-aware BNN",
    3.1: "C8 local-comparator-aware QNN",
    3.5: "LUT5-aware activation QAT",
    3.6: "analytic C8 LUT-aware QAT",
    4: "LUTNN",
    5: "anchored LUT5 search",
}

PHASE_CHECKPOINT_TEMPLATE = "Bestmodel_phase{}.pt"
LEGACY_BEST_CHECKPOINT_FILENAME = "Bestmodel.pt"
PHASE_METRICS_FILENAME = "phase_metrics.json"


def phase_tag(phase):
    numeric = float(phase)
    if numeric.is_integer():
        return str(int(numeric))
    return str(numeric).replace(".", "p")


def parse_phase(value):
    token = str(value).strip().lower().removeprefix("phase")
    token = token.replace("p", ".")
    try:
        phase = float(token)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "phase must be one of 1, 2, 2.1, 3, 3.1, 3.5, 3.6, 4, or 5"
        ) from exc
    if phase.is_integer():
        phase = int(phase)
    if phase not in PHASE_DESCRIPTIONS:
        raise argparse.ArgumentTypeError(
            "phase must be one of 1, 2, 2.1, 3, 3.1, 3.5, 3.6, 4, or 5"
        )
    return phase


def phase_checkpoint_filename(phase):
    return PHASE_CHECKPOINT_TEMPLATE.format(phase_tag(phase))


def phase_checkpoint_path(workdir, phase):
    return os.path.join(workdir, "chkpts", phase_checkpoint_filename(phase))

def phase_predecessors(phase):
    return (phase - 1,)


def phase_initialization_sources(phase, lut_inputs):
    if phase == 2.1:
        return (1,)
    if phase == 3.1:
        return (2.1,)
    if phase == 3.5:
        return (2,)
    if phase == 3.6:
        return (3,)
    if phase == 3 and lut_inputs == 5:
        return (3,)
    if phase == 4 and lut_inputs == 5:
        return (3.5, 3)
    if phase == 5:
        return (4,)
    return phase_predecessors(phase)



def legacy_best_checkpoint_path(workdir):
    return os.path.join(workdir, "chkpts", LEGACY_BEST_CHECKPOINT_FILENAME)


def duplicate_lut4_table_to_lut5(source_lut, target_shape=None):
    """Duplicate a LUT4 table across the inserted Phase 5 routing bit."""
    source_lut = source_lut.unsqueeze(0) if source_lut.ndim == 1 else source_lut
    if source_lut.ndim != 2 or source_lut.shape[1] != 16:
        raise ValueError(
            "LUT4 source must have shape [sets, 16], got {}".format(
                tuple(source_lut.shape)
            )
        )

    state5 = torch.arange(32, dtype=torch.long, device=source_lut.device)
    # LUT4 bits are [x_r, x_i, w_r, w_i]. Phase 5 inserts d at bit 2:
    # [x_r, x_i, d, w_r, w_i]. Removing d recovers the LUT4 address.
    state4 = ((state5 & 0b11000) >> 1) | (state5 & 0b00011)
    duplicated = source_lut[:, state4]

    if target_shape is None:
        return duplicated.clone()

    target_rows, target_entries = tuple(target_shape)
    if target_entries != 32:
        raise ValueError(
            "LUT5 target must have 32 entries, got {}".format(target_entries)
        )
    if duplicated.shape[0] == target_rows:
        return duplicated.clone()
    if duplicated.shape[0] == 1:
        return duplicated.expand(target_rows, -1).clone()
    raise ValueError(
        "Cannot map {} LUT4 set(s) to {} LUT5 set(s)".format(
            duplicated.shape[0], target_rows
        )
    )


def load_phase_checkpoint(
    model,
    checkpoint_path,
    device='cpu',
    expected_phase=None,
    expected_lut_inputs=None,
    expected_c8_codebook=None,
    expected_c8_grad_mode=None,
    expected_phase2p1_mode=None,
    duplicate_lut4_to_lut5=False,
):
    print(f"==> Loading and Mapping weights from '{checkpoint_path}'...")
    checkpoint = torch.load(checkpoint_path, map_location=device)

    checkpoint_args = (
        checkpoint.get("args") if isinstance(checkpoint, dict) else None
    )
    stored_phase = checkpoint.get("phase") if isinstance(checkpoint, dict) else None
    if stored_phase is None and isinstance(checkpoint, dict):
        metrics = checkpoint.get("metrics")
        if isinstance(metrics, dict):
            stored_phase = metrics.get("phase")
    if stored_phase is None and isinstance(checkpoint, dict):
        if isinstance(checkpoint_args, dict):
            stored_phase = checkpoint_args.get("phase")
    if expected_phase is not None and stored_phase is not None:
        expected_phases = (
            tuple(expected_phase)
            if isinstance(expected_phase, (tuple, list, set))
            else (expected_phase,)
        )
        if not any(
            float(stored_phase) == float(candidate)
            for candidate in expected_phases
        ):
            raise ValueError(
                "Checkpoint {} belongs to Phase {}, but Phase {} is required.".format(
                    checkpoint_path, stored_phase, expected_phase
                )
            )

    stored_lut_inputs = None
    if isinstance(checkpoint, dict):
        for metadata in (
            checkpoint.get("metrics"),
            checkpoint.get("args"),
            checkpoint,
        ):
            if isinstance(metadata, dict) and metadata.get("lut_inputs") is not None:
                stored_lut_inputs = int(metadata["lut_inputs"])
                break
    if (
        expected_lut_inputs is not None
        and stored_lut_inputs is not None
        and stored_lut_inputs != int(expected_lut_inputs)
    ):
        raise ValueError(
            "Checkpoint {} uses LUT{}, but LUT{} is required.".format(
                checkpoint_path,
                stored_lut_inputs,
                expected_lut_inputs,
            )
        )

    stored_c8_codebook = None
    stored_c8_grad_mode = None
    if isinstance(checkpoint_args, dict):
        stored_c8_codebook = checkpoint_args.get("c8_codebook")
        stored_c8_grad_mode = checkpoint_args.get("c8_grad_mode")
    if isinstance(checkpoint, dict):
        checkpoint_metrics = checkpoint.get("metrics")
        if isinstance(checkpoint_metrics, dict):
            c8_branch = checkpoint_metrics.get("c8_branch")
            if isinstance(c8_branch, dict):
                stored_c8_codebook = (
                    stored_c8_codebook or c8_branch.get("codebook_mode")
                )
                stored_c8_grad_mode = (
                    stored_c8_grad_mode or c8_branch.get("gradient_mode")
                )
    for label, stored_value, expected_value in (
        ("C8 codebook", stored_c8_codebook, expected_c8_codebook),
        ("C8 gradient mode", stored_c8_grad_mode, expected_c8_grad_mode),
        (
            "Phase2.1 mode",
            (
                checkpoint_args.get("phase2p1_mode")
                if isinstance(checkpoint_args, dict)
                else None
            ),
            expected_phase2p1_mode,
        ),
    ):
        if (
            expected_value is not None
            and stored_value is not None
            and stored_value != expected_value
        ):
            raise ValueError(
                "Checkpoint {} uses {} '{}', but '{}' is required.".format(
                    checkpoint_path,
                    label,
                    stored_value,
                    expected_value,
                )
            )

    state_dict = checkpoint['model'] if 'model' in checkpoint else (checkpoint['state_dict'] if 'state_dict' in checkpoint else checkpoint)
    target_state = model.state_dict()
    new_state_dict = {}
    skipped_keys = []
    duplicated_lut_keys = []

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

    def is_lut_table_key(key):
        return key in ("lut_r", "lut_i") or key.endswith((".lut_r", ".lut_i"))

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
            if is_lut_table_key(candidate) and target_value.ndim == 2:
                source_lut = v.unsqueeze(0) if v.ndim == 1 else v
                if source_lut.ndim != 2:
                    continue
                if (
                    duplicate_lut4_to_lut5
                    and source_lut.shape[1] == 16
                    and target_value.shape[1] == 32
                ):
                    mapped_key = actual_target_key
                    mapped_value = duplicate_lut4_table_to_lut5(
                        source_lut,
                        target_value.shape,
                    )
                    duplicated_lut_keys.append(actual_target_key)
                    break
                if source_lut.shape[1] == target_value.shape[1] and (
                    source_lut.shape[0] == target_value.shape[0] or source_lut.shape[0] == 1
                ):
                    mapped_key = actual_target_key
                    mapped_value = source_lut.expand_as(target_value).clone()
                    break

        if mapped_key is None:
            skipped_keys.append(clean_k)
            continue

        new_state_dict[mapped_key] = mapped_value

    missing_keys, unexpected_keys = model.load_state_dict(new_state_dict, strict=False)

    print(
        "==> Checkpoint mapped and loaded! ({} target tensors initialized from "
        "{} checkpoint tensors)".format(len(new_state_dict), len(state_dict))
    )
    if duplicate_lut4_to_lut5:
        if not duplicated_lut_keys:
            raise RuntimeError(
                "Phase 5 initialization did not find any LUT4 tables to duplicate"
            )
        print(
            "==> Duplicated {} LUT4 table tensor(s) into identical d=0/d=1 "
            "LUT5 slices.".format(len(duplicated_lut_keys))
        )
    if missing_keys:
        expected_missing_keywords = [
            'lut_r', 'lut_i', 'lut_set_ids', 'flat_c', 'flat_dy',
            'flat_dx', 'shifts', 'tau', 'hard', 'phase_mix',
            'phase_strength', 'phase_beta', 'c8_', 'lut_generator_',
            'binary_mlp_', 'dominance_beta', 'operation_set_ids',
            'flat_y', 'flat_x', 'mag_threshold_raw',
        ]
        unexpected_missing = [
            k for k in missing_keys
            if not any(keyword in k for keyword in expected_missing_keywords)
        ]

        if unexpected_missing:
            print(f"[Warning] Found unexpected missing keys in model:\n{unexpected_missing}")
        else:
            print(f"[Info] All missing keys are phase-specific initialized buffers/parameters as expected.")

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


def collect_batchnorm_params(model):
    params = []
    seen = set()
    for module in model.modules():
        if "BatchNorm" not in module.__class__.__name__:
            continue
        for param in module.parameters(recurse=False):
            if param.requires_grad and id(param) not in seen:
                params.append(param)
                seen.add(id(param))
    return params


def split_weight_decay_params(model, excluded_param_ids=None):
    decay = []
    no_decay = []
    excluded_param_ids = excluded_param_ids or set()

    is_binary_model = 'Binary' in model.__class__.__name__

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if id(param) in excluded_param_ids:
            continue
        if (
            is_lut_trainable_parameter_name(name)
            or is_magnitude_threshold_parameter_name(name)
        ):
            continue

        # Bias、BatchNorm、阈值类一维参数不加正则化。
        if len(param.shape) == 1 or param.ndim == 1:
            no_decay.append(param)
        else:
            lut_keywords = ['conv_r.weight', 'conv_i.weight', 'weight_r', 'weight_i']
            if is_binary_model and any(kw in name for kw in lut_keywords):
                no_decay.append(param)
            else:
                decay.append(param)

    return decay, no_decay


def collect_named_params(model, suffixes):
    params = []
    bare_names = tuple(suffix[1:] for suffix in suffixes if suffix.startswith('.'))
    for name, param in model.named_parameters():
        if param.requires_grad and (name in bare_names or name.endswith(suffixes)):
            params.append(param)
    return params


def is_lut_trainable_parameter_name(name):
    return (
        name in ("lut_r", "lut_i")
        or name.endswith((".lut_r", ".lut_i"))
        or ".lut_generator_" in name
        or name.startswith("lut_generator_")
    )


def is_magnitude_threshold_parameter_name(name):
    return name == "mag_threshold_raw" or name.endswith(
        ".mag_threshold_raw"
    )


def collect_lut_trainable_params(model):
    return [
        param
        for name, param in model.named_parameters()
        if param.requires_grad and is_lut_trainable_parameter_name(name)
    ]


def collect_magnitude_threshold_params(model):
    return [
        param
        for name, param in model.named_parameters()
        if param.requires_grad
        and is_magnitude_threshold_parameter_name(name)
    ]


def iter_lut_modules(model):
    for module in model.modules():
        if module.__class__.__name__ in (
            "ComplexLUTConv2d",
            "UltimateComplexLUTConv2d",
        ):
            yield module


@torch.no_grad()
def verify_lut5_slices_duplicated(model):
    module_count = 0
    pair_count = 0
    value_mismatches = 0
    sign_mismatches = 0
    max_abs_diff = 0.0

    for module in iter_lut_modules(model):
        if module.phase != 5 or module.lut_inputs != 5:
            continue
        if module.physical_lut_size != 32:
            raise RuntimeError("Phase 5 LUT must contain exactly 32 entries")
        state = torch.arange(32, dtype=torch.long, device=module.lut_r.device)
        slice0 = state[((state >> 2) & 1) == 0]
        slice1 = slice0 | 0b00100
        for parameter_name in ("lut_r", "lut_i"):
            table = getattr(module, parameter_name).detach()
            left = table[:, slice0]
            right = table[:, slice1]
            difference = (left - right).abs()
            pair_count += left.numel()
            value_mismatches += int((left != right).sum().item())
            sign_mismatches += int(((left >= 0) != (right >= 0)).sum().item())
            max_abs_diff = max(max_abs_diff, float(difference.max().item()))
        module_count += 1

    if module_count == 0:
        raise RuntimeError("Phase 5 initialization found no LUT5 modules")
    stats = {
        "modules": module_count,
        "pairs": pair_count,
        "value_mismatches": value_mismatches,
        "sign_mismatches": sign_mismatches,
        "max_abs_diff": max_abs_diff,
    }
    if value_mismatches:
        raise RuntimeError(
            "Phase 5 must start from identical LUT5 slices, got {}".format(stats)
        )
    return stats


def begin_lut_flip_search(
    model,
    init_flip_prob,
    score_temperature,
    anchor_slice=None,
    ties_only=False,
):
    module_count = 0
    entry_count = 0
    eligible_entry_count = 0
    for module in iter_lut_modules(model):
        module.begin_lut_flip_search(
            init_flip_prob,
            score_temperature,
            anchor_slice=anchor_slice,
            ties_only=ties_only,
        )
        module_count += 1
        entry_count += module.lut_r.numel() + module.lut_i.numel()
        eligible_entry_count += int(module.lut_search_eligible_r.sum().item())
        eligible_entry_count += int(module.lut_search_eligible_i.sum().item())
    if module_count == 0:
        raise RuntimeError("LUT flip search requested, but the model has no ComplexLUTConv2d modules")
    return module_count, entry_count, eligible_entry_count


def lut_flip_sparsity_loss(model):
    probabilities = []
    for module in iter_lut_modules(model):
        prob_r, prob_i = module.lut_flip_probabilities()
        probabilities.extend(
            (
                prob_r[module.lut_search_eligible_r],
                prob_i[module.lut_search_eligible_i],
            )
        )
    if not probabilities:
        raise RuntimeError("LUT flip sparsity requested, but no active LUT search modules were found")
    return torch.cat(probabilities).mean()


def reset_optimizer_param_entries(optimizer, param, indices):
    if not indices:
        return
    state = optimizer.state.get(param)
    if not state:
        return
    for value in state.values():
        if torch.is_tensor(value) and value.shape == param.shape:
            index_tensor = torch.as_tensor(indices, dtype=torch.long, device=value.device)
            value.reshape(-1)[index_tensor] = 0


@torch.no_grad()
def microcommit_lut_flip_search(
    model,
    optimizer,
    threshold,
    max_flips,
    max_flips_per_layer,
    cooldown_commits,
):
    modules = list(iter_lut_modules(model))
    candidates = []
    all_probabilities = []
    eligible_count = 0

    for module_index, module in enumerate(modules):
        prob_r, prob_i = module.lut_flip_probabilities()
        flat_prob_r = prob_r.reshape(-1)
        flat_prob_i = prob_i.reshape(-1)
        combined_prob = torch.cat([flat_prob_r, flat_prob_i])
        combined_cooldown = torch.cat([
            module.lut_search_cooldown_r.reshape(-1),
            module.lut_search_cooldown_i.reshape(-1),
        ])
        all_probabilities.append(combined_prob)

        eligible = (combined_prob >= threshold) & (combined_cooldown == 0)
        eligible_indices = torch.nonzero(eligible, as_tuple=False).reshape(-1)
        eligible_count += int(eligible_indices.numel())
        layer_budget = min(max_flips_per_layer, int(eligible_indices.numel()))
        if layer_budget == 0:
            continue

        eligible_values = combined_prob[eligible_indices]
        top_values, top_positions = torch.topk(eligible_values, k=layer_budget)
        top_indices = eligible_indices[top_positions]
        for value, combined_index in zip(top_values.tolist(), top_indices.tolist()):
            candidates.append((float(value), module_index, int(combined_index)))

    candidates.sort(key=lambda item: item[0], reverse=True)
    selected = candidates[:max_flips]
    selected_by_module = {
        module_index: {"real": [], "imag": []}
        for module_index in range(len(modules))
    }
    selected_probabilities = []
    selected_entries = []
    for probability, module_index, combined_index in selected:
        module = modules[module_index]
        real_entries = module.lut_r.numel()
        if combined_index < real_entries:
            component = "real"
            entry_index = combined_index
        else:
            component = "imag"
            entry_index = combined_index - real_entries
        selected_by_module[module_index][component].append(entry_index)
        selected_entries.append(
            {
                "module_index": module_index,
                "component": component,
                "entry_index": entry_index,
                "probability": probability,
            }
        )
        selected_probabilities.append(probability)

    for module in modules:
        module.advance_lut_flip_cooldown()

    real_flips = 0
    imag_flips = 0
    for module_index, module in enumerate(modules):
        real_indices = selected_by_module[module_index]["real"]
        imag_indices = selected_by_module[module_index]["imag"]
        module.microcommit_lut_flip_entries(
            real_indices,
            imag_indices,
            cooldown_commits=cooldown_commits,
        )
        reset_optimizer_param_entries(optimizer, module.lut_r, real_indices)
        reset_optimizer_param_entries(optimizer, module.lut_i, imag_indices)
        real_flips += len(real_indices)
        imag_flips += len(imag_indices)

    if not all_probabilities:
        raise RuntimeError("LUT micro-commit found no active LUT modules")
    values = torch.cat(all_probabilities)
    total_flips = real_flips + imag_flips
    return {
        "real_flips": real_flips,
        "imag_flips": imag_flips,
        "total_flips": total_flips,
        "entries": int(values.numel()),
        "flip_ratio": total_flips / float(values.numel()),
        "eligible_candidates": eligible_count,
        "layer_limited_candidates": len(candidates),
        "mean_flip_prob": float(values.mean().item()),
        "max_flip_prob": float(values.max().item()),
        "min_committed_prob": (
            float(min(selected_probabilities)) if selected_probabilities else None
        ),
        "selected_entries": selected_entries,
    }


@torch.no_grad()
def finalize_lut_flip_search(model, logit_abs_value):
    totals = {
        "net_real_flips": 0,
        "net_imag_flips": 0,
        "entries": 0,
        "eligible_real_entries": 0,
        "eligible_imag_entries": 0,
        "frozen_real_diff": 0,
        "frozen_imag_diff": 0,
    }
    for module in iter_lut_modules(model):
        stats = module.finalize_lut_flip_search(logit_abs_value)
        for key in totals:
            totals[key] += stats[key]

    if totals["entries"] == 0:
        raise RuntimeError("LUT flip search finalization found no LUT entries")
    frozen_diff = totals["frozen_real_diff"] + totals["frozen_imag_diff"]
    if frozen_diff:
        raise RuntimeError(
            "Frozen LUT5 anchor changed in {} entries".format(frozen_diff)
        )
    totals["net_total_flips"] = totals["net_real_flips"] + totals["net_imag_flips"]
    totals["net_flip_ratio"] = totals["net_total_flips"] / totals["entries"]
    totals["eligible_entries"] = (
        totals["eligible_real_entries"] + totals["eligible_imag_entries"]
    )
    totals["eligible_flip_ratio"] = (
        totals["net_total_flips"] / max(totals["eligible_entries"], 1)
    )
    return totals


def effective_lut_logits(module):
    if hasattr(module, "effective_lut_logits"):
        return module.effective_lut_logits()
    return module.lut_r, module.lut_i


@torch.no_grad()
def capture_lut_reference(model):
    reference = {}
    for name, module in model.named_modules():
        if module.__class__.__name__ not in ("ComplexLUTConv2d", "UltimateComplexLUTConv2d"):
            continue
        lut_r, lut_i = effective_lut_logits(module)
        reference[name] = {
            "lut_r": lut_r.detach().clone(),
            "lut_i": lut_i.detach().clone(),
        }
    if not reference:
        raise RuntimeError("No LUT modules found while capturing LUT reference")
    return reference


def capture_lut_unperturbed_reference(model):
    reference = {}
    for name, module in model.named_modules():
        if module.__class__.__name__ not in ("ComplexLUTConv2d", "UltimateComplexLUTConv2d"):
            continue
        lut_r = getattr(module, "lut_unperturbed_reference_r", module.lut_r)
        lut_i = getattr(module, "lut_unperturbed_reference_i", module.lut_i)
        reference[name] = {
            "lut_r": lut_r.detach().clone(),
            "lut_i": lut_i.detach().clone(),
        }
    if not reference:
        raise RuntimeError("No LUT modules found while capturing unperturbed LUT reference")
    return reference


@torch.no_grad()
def analyze_lut_change(model, reference):
    thresholds = (0.05, 0.1, 0.25, 0.5)
    totals = {
        "entries": 0,
        "sign_diff": 0,
        "abs_delta_sum": 0.0,
        "abs_delta_max": 0.0,
        "near_zero": {str(value): 0 for value in thresholds},
        "layers": [],
    }

    for name, module in model.named_modules():
        if name not in reference:
            continue
        layer = {
            "name": name,
            "entries": 0,
            "sign_diff": 0,
            "abs_delta_sum": 0.0,
            "abs_delta_max": 0.0,
            "near_zero": {str(value): 0 for value in thresholds},
        }
        current_tables = dict(
            zip(("lut_r", "lut_i"), effective_lut_logits(module))
        )
        for parameter_name in ("lut_r", "lut_i"):
            current = current_tables[parameter_name].detach()
            initial = reference[name][parameter_name].to(current.device)
            delta = (current - initial).abs()
            count = current.numel()
            sign_diff = int(((current >= 0) != (initial >= 0)).sum().item())

            layer["entries"] += count
            layer["sign_diff"] += sign_diff
            layer["abs_delta_sum"] += float(delta.sum().item())
            layer["abs_delta_max"] = max(
                layer["abs_delta_max"],
                float(delta.max().item()),
            )
            for threshold in thresholds:
                key = str(threshold)
                layer["near_zero"][key] += int(
                    (current.abs() < threshold).sum().item()
                )

        layer["sign_diff_ratio"] = layer["sign_diff"] / float(layer["entries"])
        layer["abs_delta_mean"] = layer["abs_delta_sum"] / float(layer["entries"])
        totals["entries"] += layer["entries"]
        totals["sign_diff"] += layer["sign_diff"]
        totals["abs_delta_sum"] += layer["abs_delta_sum"]
        totals["abs_delta_max"] = max(
            totals["abs_delta_max"],
            layer["abs_delta_max"],
        )
        for key, value in layer["near_zero"].items():
            totals["near_zero"][key] += value
        totals["layers"].append(layer)

    if totals["entries"] == 0:
        raise RuntimeError("No LUT entries found while analyzing LUT change")
    totals["sign_diff_ratio"] = totals["sign_diff"] / float(totals["entries"])
    totals["abs_delta_mean"] = totals["abs_delta_sum"] / float(totals["entries"])
    return totals


@torch.no_grad()
def analyze_lut5_slice_divergence(model):
    """Measure whether the hard LUT5 output actually depends on input bit d."""
    totals = {
        "modules": 0,
        "pairs": 0,
        "hard_slice_diff": 0,
        "soft_abs_diff_sum": 0.0,
        "soft_abs_diff_max": 0.0,
        "outputs": {},
    }

    for name, module in model.named_modules():
        if module.__class__.__name__ not in (
            "ComplexLUTConv2d",
            "UltimateComplexLUTConv2d",
        ):
            continue
        if getattr(module, "lut_inputs", None) != 5:
            continue

        current_tables = dict(
            zip(("lut_r", "lut_i"), effective_lut_logits(module))
        )
        state = torch.arange(32, device=current_tables["lut_r"].device)
        slice0 = state[((state >> 2) & 1) == 0]
        slice1 = slice0 | 0b00100
        totals["modules"] += 1

        for output_name, parameter_name in (
            ("real", "lut_r"),
            ("imag", "lut_i"),
        ):
            table = current_tables[parameter_name].detach()
            left = table.index_select(1, slice0)
            right = table.index_select(1, slice1)
            delta = (right - left).abs()
            hard_diff = int(
                ((right >= 0) != (left >= 0)).sum().item()
            )
            pair_count = int(delta.numel())

            output = totals["outputs"].setdefault(
                output_name,
                {
                    "pairs": 0,
                    "hard_slice_diff": 0,
                    "soft_abs_diff_sum": 0.0,
                    "soft_abs_diff_max": 0.0,
                },
            )
            output["pairs"] += pair_count
            output["hard_slice_diff"] += hard_diff
            output["soft_abs_diff_sum"] += float(delta.sum().item())
            output["soft_abs_diff_max"] = max(
                output["soft_abs_diff_max"],
                float(delta.max().item()),
            )

            totals["pairs"] += pair_count
            totals["hard_slice_diff"] += hard_diff
            totals["soft_abs_diff_sum"] += float(delta.sum().item())
            totals["soft_abs_diff_max"] = max(
                totals["soft_abs_diff_max"],
                float(delta.max().item()),
            )

    if totals["modules"] == 0:
        raise RuntimeError(
            "No trainable LUT5 modules found while analyzing d-slice divergence"
        )

    for stats in list(totals["outputs"].values()) + [totals]:
        stats["hard_slice_diff_ratio"] = (
            stats["hard_slice_diff"] / float(stats["pairs"])
        )
        stats["soft_abs_diff_mean"] = (
            stats["soft_abs_diff_sum"] / float(stats["pairs"])
        )
    return totals


@torch.no_grad()
def analyze_lut_activation_bit_sensitivity(model, bit_names):
    if len(bit_names) != 3:
        raise ValueError("LUT5 activation sensitivity requires three bit names")
    totals = {
        name: {
            "pairs": 0,
            "hard_diff": 0,
            "soft_abs_diff_sum": 0.0,
            "soft_abs_diff_max": 0.0,
        }
        for name in bit_names
    }
    module_count = 0
    for module in iter_lut_modules(model):
        if getattr(module, "lut_inputs", None) != 5:
            continue
        lut_r, lut_i = effective_lut_logits(module)
        state = torch.arange(32, device=lut_r.device)
        module_count += 1
        for activation_index, name in enumerate(bit_names):
            shift = 4 - activation_index
            left_index = state[((state >> shift) & 1) == 0]
            right_index = left_index | (1 << shift)
            stats = totals[name]
            for table in (lut_r.detach(), lut_i.detach()):
                left = table.index_select(1, left_index)
                right = table.index_select(1, right_index)
                delta = (right - left).abs()
                stats["pairs"] += int(delta.numel())
                stats["hard_diff"] += int(
                    ((right >= 0) != (left >= 0)).sum().item()
                )
                stats["soft_abs_diff_sum"] += float(delta.sum().item())
                stats["soft_abs_diff_max"] = max(
                    stats["soft_abs_diff_max"],
                    float(delta.max().item()),
                )
    if module_count == 0:
        raise RuntimeError(
            "No LUT5 modules found while analyzing activation-bit sensitivity"
        )
    for stats in totals.values():
        stats["hard_diff_ratio"] = (
            stats["hard_diff"] / float(stats["pairs"])
        )
        stats["soft_abs_diff_mean"] = (
            stats["soft_abs_diff_sum"] / float(stats["pairs"])
        )
    return {
        "modules": module_count,
        "bits": totals,
    }


@torch.no_grad()
def capture_hard_lut_tables(model):
    tables = {}
    for name, module in model.named_modules():
        if module.__class__.__name__ not in (
            "ComplexLUTConv2d",
            "UltimateComplexLUTConv2d",
        ):
            continue
        lut_r, lut_i = effective_lut_logits(module)
        tables[name] = {
            "real": (lut_r >= 0).to(torch.uint8).cpu(),
            "imag": (lut_i >= 0).to(torch.uint8).cpu(),
            "input_bits": int(module.lut_inputs),
            "parameterization": getattr(
                module,
                "lut_parameterization",
                "direct",
            ),
        }
    if not tables:
        raise RuntimeError("No LUT modules found while exporting hard tables")
    return tables




@torch.no_grad()
def materialize_hard_lut(model, logit_abs_value):
    logit_abs_value = max(abs(float(logit_abs_value)), 1.0)
    for module in iter_lut_modules(model):
        if getattr(module, "lut_parameterization", "direct") != "direct":
            raise RuntimeError(
                "materialize_hard_lut only supports direct LUT parameters"
            )
        module.lut_r.copy_(
            torch.where(
                module.lut_r >= 0,
                torch.full_like(module.lut_r, logit_abs_value),
                torch.full_like(module.lut_r, -logit_abs_value),
            )
        )
        module.lut_i.copy_(
            torch.where(
                module.lut_i >= 0,
                torch.full_like(module.lut_i, logit_abs_value),
                torch.full_like(module.lut_i, -logit_abs_value),
            )
        )
        module.hard.fill_(1.0)


def update_lut_soft_only_stage(
    model,
    epoch,
    stage_epochs,
    tau_min,
    tau_max,
    train_logger=None,
    hard_forward=False,
):
    if tau_min <= 0.0 or tau_max <= 0.0:
        raise ValueError("LUT soft-only tau values must be positive")
    progress = min(max(epoch / max(stage_epochs - 1, 1), 0.0), 1.0)
    current_tau = tau_min * math.pow(tau_max / tau_min, progress)
    for module in iter_lut_modules(model):
        module.tau.fill_(current_tau)
        module.hard.fill_(1.0 if hard_forward else 0.0)
    if train_logger is not None and (
        epoch == 0 or epoch == stage_epochs - 1 or (epoch + 1) % 10 == 0
    ):
        train_logger.info(
            "[LUT {}-Only] Epoch {} / {}: tau={:.4f}, hard={}".format(
                "Hard+BN" if hard_forward else "Soft",
                epoch + 1,
                stage_epochs,
                current_tau,
                hard_forward,
            )
        )
    return current_tau


def set_lut_soft_then_weight_stage_lr(
    optimizer,
    base_lr,
    lut_stage,
    lut_lr=None,
    lut_schedule="follow_base",
    train_bn=False,
):
    inactive_groups = set()
    operation_groups = {"lut", "mag_threshold"}
    if train_bn:
        operation_groups.add("bn")
    for group in optimizer.param_groups:
        group_name = group.get("name", "")
        is_operation_group = group_name in operation_groups
        is_active = (
            True
            if train_bn and group_name == "bn"
            else (is_operation_group if lut_stage else not is_operation_group)
        )
        group["lr"] = (
            get_optimizer_group_lr(group, base_lr, lut_lr, lut_schedule)
            if is_active
            else 0.0
        )
        if not is_active:
            inactive_groups.add(group_name)
    return inactive_groups


def get_optimizer_group_lr(group, base_lr, lut_lr=None, lut_schedule="follow_base"):
    group_name = group.get("name", "")
    if group_name == "lut" and lut_schedule == "constant":
        return lut_lr
    if (
        group_name == "mag_threshold"
        and group.get("lr_schedule") == "constant"
    ):
        return group["constant_lr"]
    return base_lr * group.get("lr_multiplier", 1.0)


def set_lut_search_stage_lr(
    optimizer,
    base_lr,
    searching,
    lut_lr=None,
    lut_schedule="follow_base",
):
    inactive_groups = set()
    for group in optimizer.param_groups:
        group_name = group.get("name", "")
        is_active = (group_name == "lut") if searching else (group_name != "lut")
        group["lr"] = (
            get_optimizer_group_lr(group, base_lr, lut_lr, lut_schedule)
            if is_active
            else 0.0
        )
        if not is_active:
            inactive_groups.add(group_name)
    return inactive_groups


def clear_optimizer_group_grads(optimizer, inactive_groups):
    for group in optimizer.param_groups:
        if group.get("name", "") not in inactive_groups:
            continue
        for param in group["params"]:
            param.grad = None


class MagnitudeThresholdGradientTracker:
    """Aggregate fifth-bit threshold gradients before gradient clipping."""

    def __init__(self, model):
        self.parameters = collect_magnitude_threshold_params(model)
        if not self.parameters:
            raise RuntimeError(
                "magnitude gradient tracker found no threshold parameters"
            )
        self.reset()

    def reset(self):
        self.batches = 0
        self.element_count = 0
        self.batches_with_nonzero = None
        self.nonzero_count = None
        self.abs_sum = None
        self.abs_max = None

    @torch.no_grad()
    def update(self):
        self.batches += 1
        grads = [
            param.grad.detach().reshape(-1)
            for param in self.parameters
            if param.grad is not None
        ]
        if not grads:
            return
        flat_grad = torch.cat(grads)
        finite = flat_grad[torch.isfinite(flat_grad)]
        if finite.numel() == 0:
            return
        abs_grad = finite.abs()
        batch_abs_sum = abs_grad.sum()
        batch_abs_max = abs_grad.max()
        batch_nonzero = (abs_grad > 0.0).sum(dtype=torch.int64)
        batch_has_nonzero = (batch_nonzero > 0).to(torch.int64)

        self.element_count += int(abs_grad.numel())
        if self.abs_sum is None:
            self.abs_sum = batch_abs_sum
            self.abs_max = batch_abs_max
            self.nonzero_count = batch_nonzero
            self.batches_with_nonzero = batch_has_nonzero
        else:
            self.abs_sum = self.abs_sum + batch_abs_sum
            self.abs_max = torch.maximum(self.abs_max, batch_abs_max)
            self.nonzero_count = self.nonzero_count + batch_nonzero
            self.batches_with_nonzero = (
                self.batches_with_nonzero + batch_has_nonzero
            )

    def finish(self):
        batches_with_nonzero = (
            0
            if self.batches_with_nonzero is None
            else int(self.batches_with_nonzero.item())
        )
        nonzero_count = (
            0
            if self.nonzero_count is None
            else int(self.nonzero_count.item())
        )
        abs_sum = (
            0.0 if self.abs_sum is None else float(self.abs_sum.item())
        )
        abs_max = (
            0.0 if self.abs_max is None else float(self.abs_max.item())
        )
        return {
            "batches": self.batches,
            "batches_with_nonzero": batches_with_nonzero,
            "batch_nonzero_ratio": (
                batches_with_nonzero / float(self.batches)
                if self.batches
                else 0.0
            ),
            "elements": self.element_count,
            "nonzero_elements": nonzero_count,
            "nonzero_ratio": (
                nonzero_count / float(self.element_count)
                if self.element_count
                else 0.0
            ),
            "mean_abs": (
                abs_sum / float(self.element_count)
                if self.element_count
                else 0.0
            ),
            "max_abs": abs_max,
        }


@torch.no_grad()
def analyze_magnitude_thresholds(model):
    values = []
    for module in iter_lut_modules(model):
        if getattr(module, "lut_extra_bit", "phase") != "magnitude_ste":
            continue
        values.append(module.physical_mag_threshold().reshape(-1))
    if not values:
        raise RuntimeError("No magnitude_ste thresholds found")
    values = torch.cat(values)
    return {
        "count": int(values.numel()),
        "min": float(values.min().item()),
        "mean": float(values.mean().item()),
        "max": float(values.max().item()),
    }


class PhaseOccupancyTracker:
    """Accumulate the configured LUT5 extra bit during an eval pass."""

    def __init__(self, model):
        self.enabled = False
        self.layers = {}
        self.handles = []
        for name, module in model.named_modules():
            if not isinstance(
                module,
                (
                    LUTAwareComplexBinaryConv2d,
                    LUT5AwareComplexQATConv2d,
                    ComplexLUTConv2d,
                ),
            ) or module.lut_inputs != 5:
                continue
            self.layers[name] = {
                "ones": None,
                "margin_sum": None,
                "near_boundary": None,
                "boundary_width": None,
                "total": 0,
            }
            self.handles.append(
                module.register_forward_pre_hook(
                    self._make_hook(name),
                    with_kwargs=True,
                )
            )
        if not self.handles:
            raise RuntimeError("phase occupancy tracker found no LUT5 modules")

    def _make_hook(self, name):
        def hook(module, args, kwargs):
            if not self.enabled:
                return
            phase_source = kwargs.get("phase_source")
            if phase_source is None and len(args) > 1:
                phase_source = args[1]
            if (
                phase_source is None
                and kwargs.get("activation_bits") is not None
                and len(args) > 0
            ):
                phase_source = args[0]
            if phase_source is None:
                raise RuntimeError(
                    "LUT5 occupancy hook did not receive phase_source"
                )
            with torch.no_grad():
                if (
                    isinstance(module, ComplexLUTConv2d)
                    and getattr(module, "lut_extra_bit", "phase")
                    == "magnitude_ste"
                ):
                    threshold = module.physical_mag_threshold().view(
                        1, -1, 1, 1
                    )
                    phase_score = torch.abs(phase_source) - threshold
                    boundary_width = 1.0 / max(
                        float(module.magnitude_bit_beta),
                        1e-12,
                    )
                else:
                    phase_score = (
                        phase_source.real.abs()
                        - phase_source.imag.abs()
                    )
                    boundary_width = None
                bit_values = phase_score >= 0
                one_count = bit_values.sum(dtype=torch.int64)
                margin_sum = phase_score.abs().sum(dtype=torch.float64)
                layer = self.layers[name]
                layer["ones"] = (
                    one_count.detach()
                    if layer["ones"] is None
                    else layer["ones"] + one_count.detach()
                )
                layer["margin_sum"] = (
                    margin_sum.detach()
                    if layer["margin_sum"] is None
                    else layer["margin_sum"] + margin_sum.detach()
                )
                if isinstance(module, LUT5AwareComplexQATConv2d):
                    boundary_width = 1.0 / max(
                        float(module.phase_beta.item()),
                        1e-12,
                    )
                if boundary_width is not None:
                    near_count = (
                        phase_score.abs() <= boundary_width
                    ).sum(dtype=torch.int64)
                    layer["near_boundary"] = (
                        near_count.detach()
                        if layer["near_boundary"] is None
                        else layer["near_boundary"] + near_count.detach()
                    )
                    layer["boundary_width"] = boundary_width
                layer["total"] += bit_values.numel()

        return hook

    def start(self):
        for layer in self.layers.values():
            layer["ones"] = None
            layer["margin_sum"] = None
            layer["near_boundary"] = None
            layer["boundary_width"] = None
            layer["total"] = 0
        self.enabled = True

    def finish(self):
        self.enabled = False
        layers = []
        total_ones = 0
        total_entries = 0
        total_margin = 0.0
        total_near_boundary = 0
        total_near_entries = 0
        for name, values in self.layers.items():
            entries = int(values["total"])
            ones = (
                0
                if values["ones"] is None
                else int(values["ones"].item())
            )
            zeros = entries - ones
            margin_sum = (
                0.0
                if values["margin_sum"] is None
                else float(values["margin_sum"].item())
            )
            layer_metrics = {
                "name": name,
                "entries": entries,
                "ones": ones,
                "zeros": zeros,
                "one_ratio": ones / float(entries) if entries else 0.0,
                "mean_abs_margin": (
                    margin_sum / float(entries) if entries else 0.0
                ),
            }
            if values["near_boundary"] is not None:
                near_boundary = int(values["near_boundary"].item())
                layer_metrics.update({
                    "boundary_width": float(values["boundary_width"]),
                    "near_boundary": near_boundary,
                    "near_boundary_ratio": (
                        near_boundary / float(entries) if entries else 0.0
                    ),
                })
                total_near_boundary += near_boundary
                total_near_entries += entries
            layers.append(layer_metrics)
            total_ones += ones
            total_entries += entries
            total_margin += margin_sum
        if total_entries == 0:
            raise RuntimeError("phase occupancy tracker captured no entries")
        return {
            "entries": total_entries,
            "ones": total_ones,
            "zeros": total_entries - total_ones,
            "one_ratio": total_ones / float(total_entries),
            "mean_abs_margin": total_margin / float(total_entries),
            "near_boundary": total_near_boundary,
            "near_boundary_ratio": (
                total_near_boundary / float(total_near_entries)
                if total_near_entries
                else None
            ),
            "layers": layers,
        }

    def close(self):
        self.enabled = False
        for handle in self.handles:
            handle.remove()
        self.handles.clear()




class C8CodeOccupancyTracker:
    """Collect C8 code usage, entropy, and angular quantization error."""

    def __init__(self, model):
        self.enabled = False
        self.layers = {}
        self.handles = []
        self.codebook_modes = set()
        named_modules = list(model.named_modules())
        direct_lut_blocks = [
            (name, module)
            for name, module in named_modules
            if getattr(module, "learned_lut5_phase3p1", False)
        ]
        if direct_lut_blocks:
            for name, block in direct_lut_blocks:
                tracked_name = name + ".act"
                activation = block.act
                self._add_layer(tracked_name, activation)
                self.handles.append(
                    block.conv.register_forward_pre_hook(
                        self._make_direct_lut_hook(
                            tracked_name,
                            activation,
                        ),
                        with_kwargs=True,
                    )
                )
        else:
            has_activation_modules = any(
                isinstance(module, C8ComplexActivation)
                for _, module in named_modules
            )
            for name, module in named_modules:
                if has_activation_modules:
                    if not isinstance(module, C8ComplexActivation):
                        continue
                elif not isinstance(module, C8LUTAwareComplexQATConv2d):
                    continue
                self._add_layer(name, module)
                self.handles.append(
                    module.register_forward_pre_hook(
                        self._make_hook(name),
                        with_kwargs=True,
                    )
                )
        if not self.handles:
            raise RuntimeError("C8 occupancy tracker found no C8 modules")
        if len(self.codebook_modes) != 1:
            raise RuntimeError(
                "C8 occupancy tracker requires one codebook mode, got {}".format(
                    sorted(self.codebook_modes)
                )
            )
        self.codebook_mode = next(iter(self.codebook_modes))

    def _add_layer(self, name, module):
        self.codebook_modes.add(
            getattr(module, "c8_codebook_mode", "roots")
        )
        self.layers[name] = {
            "counts": None,
            "total": 0,
            "angle_error_sum": 0.0,
            "angle_error_max": 0.0,
        }

    def _record(self, name, analyzer, phase_source):
        if phase_source is None:
            raise RuntimeError(
                "C8 occupancy hook did not receive its activation source"
            )
        with torch.no_grad():
            scores = analyzer._phase_scores(phase_source)
            phase_index = analyzer.hard_phase_indices(phase_source)
            counts = torch.bincount(
                phase_index.reshape(-1),
                minlength=8,
            ).to(torch.int64)
            selected_score = torch.gather(
                scores,
                -1,
                phase_index.unsqueeze(-1),
            ).squeeze(-1)
            angle_error = torch.acos(selected_score.clamp(-1.0, 1.0))
            layer = self.layers[name]
            layer["counts"] = (
                counts.detach()
                if layer["counts"] is None
                else layer["counts"] + counts.detach()
            )
            layer["total"] += phase_index.numel()
            layer["angle_error_sum"] += float(angle_error.sum().item())
            layer["angle_error_max"] = max(
                layer["angle_error_max"],
                float(angle_error.max().item()),
            )

    def _make_direct_lut_hook(self, name, activation):
        def hook(module, args, kwargs):
            del module
            if not self.enabled:
                return
            phase_source = (
                args[0]
                if args
                else kwargs.get("phase_source", kwargs.get("inp"))
            )
            self._record(name, activation, phase_source)

        return hook

    def _make_hook(self, name):
        def hook(module, args, kwargs):
            if not self.enabled:
                return
            if isinstance(module, C8ComplexActivation):
                phase_source = args[0] if args else None
            else:
                phase_source = kwargs.get("phase_source")
                if phase_source is None and len(args) > 1:
                    phase_source = args[1]
            self._record(name, module, phase_source)

        return hook

    @staticmethod
    def _entropy(ratios):
        entropy = -sum(
            ratio * math.log(ratio)
            for ratio in ratios
            if ratio > 0.0
        )
        return entropy, entropy / math.log(8.0)

    def start(self):
        for layer in self.layers.values():
            layer["counts"] = None
            layer["total"] = 0
            layer["angle_error_sum"] = 0.0
            layer["angle_error_max"] = 0.0
        self.enabled = True

    def finish(self):
        self.enabled = False
        total_counts = [0] * 8
        total_angle_error = 0.0
        max_angle_error = 0.0
        layers = []
        for name, values in self.layers.items():
            if values["counts"] is None:
                continue
            counts = [int(value) for value in values["counts"].cpu().tolist()]
            total = int(values["total"])
            ratios = [
                count / float(total) if total else 0.0
                for count in counts
            ]
            entropy, normalized_entropy = self._entropy(ratios)
            layer_stats = {
                "name": name,
                "counts": counts,
                "ratios": ratios,
                "total": total,
                "entropy": entropy,
                "normalized_entropy": normalized_entropy,
                "mean_angle_error_rad": (
                    values["angle_error_sum"] / float(total)
                    if total
                    else 0.0
                ),
                "max_angle_error_rad": values["angle_error_max"],
                "odd_index_ratio": (
                    sum(counts[1::2]) / float(total)
                    if total
                    else 0.0
                ),
            }
            if self.codebook_mode == "roots":
                layer_stats["added_axis_ratio"] = layer_stats[
                    "odd_index_ratio"
                ]
            layers.append(layer_stats)
            total_counts = [
                left + right
                for left, right in zip(total_counts, counts)
            ]
            total_angle_error += values["angle_error_sum"]
            max_angle_error = max(
                max_angle_error,
                values["angle_error_max"],
            )
        total = sum(total_counts)
        if total == 0:
            raise RuntimeError("C8 occupancy tracker captured no entries")
        ratios = [count / float(total) for count in total_counts]
        entropy, normalized_entropy = self._entropy(ratios)
        mean_angle_error = total_angle_error / float(total)
        result = {
            "codebook_mode": self.codebook_mode,
            "counts": total_counts,
            "ratios": ratios,
            "total": total,
            "odd_index_count": sum(total_counts[1::2]),
            "odd_index_ratio": sum(total_counts[1::2]) / float(total),
            "active_codes": sum(count > 0 for count in total_counts),
            "entropy": entropy,
            "normalized_entropy": normalized_entropy,
            "mean_angle_error_rad": mean_angle_error,
            "mean_angle_error_deg": math.degrees(mean_angle_error),
            "max_angle_error_rad": max_angle_error,
            "max_angle_error_deg": math.degrees(max_angle_error),
            "layers": layers,
        }
        if self.codebook_mode == "roots":
            result.update(
                {
                    "old_c4_count": sum(total_counts[0::2]),
                    "added_axis_count": result["odd_index_count"],
                    "added_axis_ratio": result["odd_index_ratio"],
                }
            )
        return result

    def close(self):
        self.enabled = False
        for handle in self.handles:
            handle.remove()
        self.handles.clear()


def phase3p6_qat_state_for_epoch(
    epoch,
    warmup_epochs,
    transition_epochs,
    beta_start,
    beta_end,
    schedule,
):
    if warmup_epochs < 0:
        raise ValueError("Phase3.6 warmup_epochs must be non-negative")
    if transition_epochs < 2:
        raise ValueError("Phase3.6 transition_epochs must be at least 2")
    if beta_start <= 0.0 or beta_end <= 0.0:
        raise ValueError("Phase3.6 beta values must be positive")
    if schedule not in ("linear", "cosine"):
        raise ValueError("Phase3.6 schedule must be linear or cosine")

    if epoch < warmup_epochs:
        raw_progress = 0.0
    else:
        raw_progress = (
            (epoch - warmup_epochs) / float(transition_epochs - 1)
        )
        raw_progress = min(max(raw_progress, 0.0), 1.0)
    if schedule == "cosine":
        code_progress = 0.5 - 0.5 * math.cos(math.pi * raw_progress)
    else:
        code_progress = raw_progress
    return {
        "raw_progress": raw_progress,
        "code_progress": code_progress,
        "beta": beta_start + (beta_end - beta_start) * code_progress,
        "hard_c8_ready": raw_progress >= 1.0,
    }


def set_phase3p6_qat_state(model, progress, beta):
    module_count = 0
    for module in model.modules():
        if isinstance(module, C8LUTAwareComplexQATConv2d):
            module.set_qat_state(progress, beta)
            module_count += 1
    if module_count == 0:
        raise RuntimeError("Phase3.6 QAT found no C8 LUT-aware layers")
    return module_count


def set_fixed_c8_qat_state(model, beta):
    activation_count = 0
    local_compare_count = 0
    for module in model.modules():
        if isinstance(module, C8ComplexActivation):
            module.set_beta(beta)
            activation_count += 1
        elif (
            isinstance(module, C8LUTAwareComplexQATConv2d)
            and getattr(module, "phase", None) == 3.1
        ):
            module.set_qat_state(progress=1.0, beta=beta)
            local_compare_count += 1
    if activation_count == 0:
        raise RuntimeError("Fixed C8 phase found no C8 activation layers")
    return {
        "activations": activation_count,
        "local_comparators": local_compare_count,
    }


def phase3p1_local_transition_state_for_epoch(
    epoch,
    transition_epochs,
    schedule,
):
    if transition_epochs == 0:
        raw_progress = 1.0
    else:
        raw_progress = min(
            max(epoch / float(transition_epochs - 1), 0.0),
            1.0,
        )
    if schedule == "cosine":
        local_progress = 0.5 - 0.5 * math.cos(math.pi * raw_progress)
    else:
        local_progress = raw_progress
    return {
        "raw_progress": raw_progress,
        "local_progress": local_progress,
        "hard_local_compare_ready": raw_progress >= 1.0,
    }


def set_phase3p1_local_compare_progress(model, progress):
    module_count = 0
    for module in model.modules():
        if (
            isinstance(module, C8LUTAwareComplexQATConv2d)
            and getattr(module, "phase", None) == 3.1
        ):
            module.set_local_compare_progress(progress)
            module_count += 1
    if module_count == 0:
        raise RuntimeError("Phase3.1 found no C8 local-comparator layers")
    return module_count



def phase3_lut5_mix_for_epoch(epoch, transition_epochs):
    if transition_epochs < 2:
        raise ValueError("phase3 LUT5 transition requires at least 2 epochs")
    progress = epoch / float(transition_epochs - 1)
    return min(max(progress, 0.0), 1.0)


def set_phase3_lut5_mix(model, value):
    module_count = 0
    for module in model.modules():
        if (
            isinstance(module, LUTAwareComplexBinaryConv2d)
            and module.lut_inputs == 5
        ):
            module.set_phase_mix(value)
            module_count += 1
    if module_count == 0:
        raise RuntimeError("phase3 LUT5 transition found no LUT5-aware layers")
    return module_count


def phase3p5_qat_state_for_epoch(
    epoch,
    warmup_epochs,
    transition_epochs,
    beta_start,
    beta_end,
):
    if warmup_epochs < 0:
        raise ValueError("Phase3.5 warmup_epochs must be non-negative")
    if transition_epochs < 2:
        raise ValueError("Phase3.5 transition_epochs must be at least 2")
    if beta_start <= 0.0 or beta_end <= 0.0:
        raise ValueError("Phase3.5 comparator beta values must be positive")

    if epoch < warmup_epochs:
        progress = 0.0
    else:
        progress = (epoch - warmup_epochs) / float(transition_epochs - 1)
        progress = min(max(progress, 0.0), 1.0)
    smooth_progress = 0.5 - 0.5 * math.cos(math.pi * progress)
    return {
        "progress": progress,
        "strength": smooth_progress,
        "beta": beta_start + (beta_end - beta_start) * smooth_progress,
        "hard_lut5_ready": progress >= 1.0,
    }


def set_phase3p5_qat_state(model, strength, beta):
    module_count = 0
    for module in model.modules():
        if isinstance(module, LUT5AwareComplexQATConv2d):
            module.set_qat_state(strength, beta)
            module_count += 1
    if module_count == 0:
        raise RuntimeError("Phase3.5 QAT found no LUT5-aware QAT layers")
    return module_count


def reset_optimizer_state(optimizer):
    optimizer.state.clear()


@torch.no_grad()
def capture_lut_search_cycle_snapshot(model):
    search_buffer_names = (
        "lut_search_base_r",
        "lut_search_base_i",
        "lut_search_origin_r",
        "lut_search_origin_i",
        "lut_search_eligible_r",
        "lut_search_eligible_i",
        "lut_search_cooldown_r",
        "lut_search_cooldown_i",
    )
    model_state = {
        name: value.detach().cpu().clone()
        for name, value in model.state_dict().items()
    }
    search_state = []
    for module in iter_lut_modules(model):
        search_state.append(
            {
                "buffers": {
                    name: getattr(module, name).detach().cpu().clone()
                    for name in search_buffer_names
                },
                "active": bool(module.lut_flip_search_active),
                "temperature": float(module.lut_flip_score_temperature),
                "initial_score": float(module.lut_flip_initial_score),
                "anchor_slice": module.lut_search_anchor_slice,
                "ties_only": bool(module.lut_search_ties_only),
            }
        )
    if not search_state:
        raise RuntimeError("Cannot snapshot Phase 5 cycle without LUT modules")
    return {"model": model_state, "search": search_state}


@torch.no_grad()
def restore_lut_search_cycle_snapshot(model, snapshot):
    model.load_state_dict(snapshot["model"], strict=True)
    modules = list(iter_lut_modules(model))
    if len(modules) != len(snapshot["search"]):
        raise RuntimeError("LUT module count changed while restoring Phase 5 cycle")
    for module, state in zip(modules, snapshot["search"]):
        for name, value in state["buffers"].items():
            target = getattr(module, name)
            target.copy_(value.to(device=target.device, dtype=target.dtype))
        module.lut_flip_search_active = state["active"]
        module.lut_flip_score_temperature = state["temperature"]
        module.lut_flip_initial_score = state["initial_score"]
        module.lut_search_anchor_slice = state["anchor_slice"]
        module.lut_search_ties_only = state["ties_only"]


@torch.no_grad()
def cooldown_rejected_lut_entries(model, selected_entries, cooldown_commits):
    modules = list(iter_lut_modules(model))
    cooldown_commits = max(int(cooldown_commits), 1)
    for entry in selected_entries:
        module = modules[entry["module_index"]]
        component = entry["component"]
        entry_index = entry["entry_index"]
        cooldown = getattr(module, "lut_search_cooldown_{}".format(
            "r" if component == "real" else "i"
        ))
        score = module.lut_r if component == "real" else module.lut_i
        cooldown.reshape(-1)[entry_index] = cooldown_commits
        score.reshape(-1)[entry_index] = module.lut_flip_initial_score


def freeze_batchnorm_running_stats(model):
    for module in model.modules():
        if isinstance(module, torch.nn.modules.batchnorm._BatchNorm) or module.__class__.__name__ == "ComplexBatchNorm2d":
            module.eval()


def get_lr_for_epoch(epoch, args, num_epochs=None):
    schedule_epochs = args.num_epochs if num_epochs is None else num_epochs
    if args.schedule == "constant":
        return args.lr

    if args.schedule == "cosine":
        progress = min(max(epoch / max(schedule_epochs - 1, 1), 0.0), 1.0)
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        min_factor = args.min_lr_factor
        return args.lr * (min_factor + (1.0 - min_factor) * cosine)

    if args.schedule == "bireal":
        warmup_epochs = 5
        if epoch < warmup_epochs:
            return args.lr * float(epoch + 1) / float(warmup_epochs)
        t1 = int(schedule_epochs * 0.5)
        t2 = int(schedule_epochs * 0.75)
        t3 = int(schedule_epochs * 0.875)
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


def set_optimizer_lr(optimizer, lr, lut_lr=None, lut_schedule="follow_base"):
    for group in optimizer.param_groups:
        group["lr"] = get_optimizer_group_lr(group, lr, lut_lr, lut_schedule)


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

    mean_inds = shuf_inds if args.no_validation else train_inds
    pixel_mean = None
    if is_main:
        pixel_mean = compute_pixel_mean(Subset(base_train, mean_inds), batch_size=args.batch_size)
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
        if args.no_validation:
            train_logger.info("Training   set size: {}".format(len(shuf_inds)))
            train_logger.info("Validation set size: disabled")
        else:
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
    if args.lut_inputs not in (4, 5):
        raise ValueError("lut_inputs must be 4 or 5")
    if args.lut_extra_bit not in ("phase", "magnitude_ste"):
        raise ValueError(
            "lut_extra_bit must be 'phase' or 'magnitude_ste'"
        )
    if args.lut_extra_bit == "magnitude_ste":
        if args.phase != 4 or args.lut_inputs != 5:
            raise ValueError(
                "magnitude_ste requires --phase 4 --lut-inputs 5"
            )
        if args.lut_init_mode != "binary":
            raise ValueError(
                "magnitude_ste requires --lut-init-mode binary"
            )
        if args.lut_sign_flip_prob != 0.0:
            raise ValueError(
                "magnitude_ste starts from an exact duplicated LUT4; "
                "set --lut-sign-flip-prob 0"
            )
        if args.magnitude_threshold_init <= 0.0:
            raise ValueError("magnitude_threshold_init must be positive")
        if args.magnitude_bit_beta <= 0.0:
            raise ValueError("magnitude_bit_beta must be positive")
        if not 0.0 < args.magnitude_shadow_epsilon < 0.5:
            raise ValueError(
                "magnitude_shadow_epsilon must be in (0, 0.5)"
            )
    if args.lut_init_mode not in ("binary", "raw", "random"):
        raise ValueError("lut_init_mode must be 'binary', 'raw', or 'random'")
    if not 0.0 <= args.lut_sign_flip_prob <= 1.0:
        raise ValueError("lut_sign_flip_prob must be between 0 and 1")
    if args.comp_init not in ("complex_independent", "bimodal"):
        raise ValueError("comp_init must be 'complex_independent' or 'bimodal'")
    if args.c8_beta <= 0.0:
        raise ValueError("c8_beta must be positive")
    if args.phase2p1_mode not in ("c8", "learned_lut5"):
        raise ValueError(
            "phase2p1_mode must be 'c8' or 'learned_lut5'"
        )
    if args.phase3p1_mode not in (
        "fixed",
        "learned_lut5",
        "semantic_lut5",
        "neural_lut5",
        "pure_mlp",
    ):
        raise ValueError(
            "phase3p1_mode must be 'fixed', 'learned_lut5', "
            "'semantic_lut5', 'neural_lut5', or 'pure_mlp'"
        )
    if args.neural_lut_hidden < 1:
        raise ValueError("neural_lut_hidden must be at least 1")
    if args.neural_lut_residual_scale <= 0.0:
        raise ValueError("neural_lut_residual_scale must be positive")
    if args.neural_lut_hard_eval_interval < 0:
        raise ValueError("neural_lut_hard_eval_interval must be non-negative")
    if args.c8_codebook not in ("roots", "octants"):
        raise ValueError("c8_codebook must be 'roots' or 'octants'")
    if args.c8_grad_mode not in (
        "softmax",
        "bireal_ste",
        "semantic_ste",
        "semantic_phase_ste",
    ):
        raise ValueError(
            "c8_grad_mode must be 'softmax', 'bireal_ste', "
            "'semantic_ste', or 'semantic_phase_ste'"
        )
    if (
        args.c8_grad_mode in ("semantic_ste", "semantic_phase_ste")
        and args.c8_codebook != "octants"
    ):
        raise ValueError(
            "c8_grad_mode {} requires c8_codebook octants".format(
                args.c8_grad_mode
            )
        )
    if args.phase in (2.1, 3.1) and args.lut_inputs != 5:
        raise ValueError("Phase2.1 and Phase3.1 require --lut-inputs 5")

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
        lut_inputs=args.lut_inputs,
        lut_logit_init=args.lut_logit_init,
        lut_init_mode=args.lut_init_mode,
        lut_sign_flip_prob=args.lut_sign_flip_prob,
        comp_init=args.comp_init,
        c8_beta=args.c8_beta,
        c8_codebook=args.c8_codebook,
        c8_grad_mode=args.c8_grad_mode,
        phase3p1_padding_mode=args.phase3p1_padding_mode,
        phase2p1_mode=args.phase2p1_mode,
        phase3p1_mode=args.phase3p1_mode,
        lut_init_tau=args.lut_tau_min,
        neural_lut_hidden=args.neural_lut_hidden,
        neural_lut_residual_scale=args.neural_lut_residual_scale,
        lut_extra_bit=args.lut_extra_bit,
        magnitude_threshold_init=args.magnitude_threshold_init,
        magnitude_bit_beta=args.magnitude_bit_beta,
        magnitude_shadow_epsilon=args.magnitude_shadow_epsilon,
    )


def build_optimizer(args, model):
    base_model = model.module if hasattr(model, "module") else model
    bn_params = collect_batchnorm_params(base_model)
    decay_params, no_decay_params = split_weight_decay_params(
        base_model, {id(param) for param in bn_params}
    )
    lut_params = collect_lut_trainable_params(base_model)
    magnitude_threshold_params = collect_magnitude_threshold_params(
        base_model
    )

    param_groups = []
    if decay_params:
        param_groups.append({"params": decay_params, "weight_decay": args.l2, "lr_multiplier": 1.0, "lr": args.lr, "name": "decay"})
    if no_decay_params:
        param_groups.append({"params": no_decay_params, "weight_decay": 0.0, "lr_multiplier": 1.0, "lr": args.lr, "name": "no_decay"})
    if bn_params:
        param_groups.append({"params": bn_params, "weight_decay": 0.0, "lr_multiplier": 1.0, "lr": args.lr, "name": "bn"})
    if lut_params:
        lut_multiplier = (args.lut_lr / args.lr) if args.lut_lr is not None else 1.0
        param_groups.append({"params": lut_params, "weight_decay": 0.0, "lr_multiplier": lut_multiplier, "lr": args.lr * lut_multiplier, "name": "lut"})
    if magnitude_threshold_params:
        threshold_multiplier = args.magnitude_threshold_lr / args.lr
        param_groups.append({
            "params": magnitude_threshold_params,
            "weight_decay": 0.0,
            "lr_multiplier": threshold_multiplier,
            "lr": args.magnitude_threshold_lr,
            "name": "mag_threshold",
            "lr_schedule": args.magnitude_threshold_schedule,
            "constant_lr": args.magnitude_threshold_lr,
        })

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


def clip_optimizer_grad_norm_(
    model,
    max_norm,
):
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)



def train_one_epoch(
    model,
    loader,
    optimizer,
    device,
    clipnorm,
    clipval,
    inactive_optimizer_groups=None,
    freeze_batchnorm_stats=False,
    lut_search_sparsity=0.0,
    magnitude_gradient_tracker=None,
):
    model.train()
    if freeze_batchnorm_stats:
        freeze_batchnorm_running_stats(model)

    inactive_optimizer_groups = inactive_optimizer_groups or set()

    loss_sum = 0.0
    correct = 0
    count = 0
    for data, target in loader:
        data = data.to(device)
        target = target.to(device)

        optimizer.zero_grad(set_to_none=True)
        output = model(data)
        loss = F.cross_entropy(output, target)
        if lut_search_sparsity > 0.0:
            loss = loss + lut_search_sparsity * lut_flip_sparsity_loss(model)
        loss.backward()
        if magnitude_gradient_tracker is not None:
            magnitude_gradient_tracker.update()

        clear_optimizer_group_grads(optimizer, inactive_optimizer_groups)
        if clipnorm is not None and clipnorm > 0:
            clip_optimizer_grad_norm_(model, clipnorm)
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


def evaluate_hard_lut_projection(model, val_loader, test_loader, device):
    base_model = model.module if hasattr(model, "module") else model
    modules = list(iter_lut_modules(base_model))
    if not modules:
        raise RuntimeError("Hard LUT projection found no LUT modules")
    hard_values = [module.hard.detach().clone() for module in modules]
    try:
        for module in modules:
            module.hard.fill_(1.0)
        val_loss, val_acc = (
            evaluate(model, val_loader, device)
            if val_loader is not None
            else (0.0, 0.0)
        )
        test_loss, test_acc = (
            evaluate(model, test_loader, device)
            if test_loader is not None
            else (0.0, 0.0)
        )
    finally:
        for module, hard_value in zip(modules, hard_values):
            module.hard.copy_(hard_value)
    return {
        "val_loss": float(val_loss),
        "val_acc": float(val_acc),
        "test_loss": float(test_loss),
        "test_acc": float(test_acc),
    }


def save_checkpoint(state, workdir, filename):
    chkpt_dir = os.path.join(workdir, "chkpts")
    if not os.path.isdir(chkpt_dir):
        os.makedirs(chkpt_dir)
    path = os.path.join(chkpt_dir, filename)
    torch.save(state, path)
    return path

def update_lut_annealing(model, epoch, num_epochs, phase, train_logger, is_main, args):
    learned_lut5_phase2p1 = (
        phase == 2.1 and args.phase2p1_mode == "learned_lut5"
    )
    learned_lut5_phase3p1 = (
        phase == 3.1
        and args.phase3p1_mode in (
            "learned_lut5",
            "semantic_lut5",
            "neural_lut5",
        )
    )
    neural_lut5_phase3p1 = (
        phase == 3.1 and args.phase3p1_mode == "neural_lut5"
    )
    if (
        phase != 4
        and not learned_lut5_phase2p1
        and not learned_lut5_phase3p1
    ):
        return True

    # Phase 4 keeps the soft LUT annealing period for training, but best-checkpoint
    # selection starts only when hard mode is active. With --lut-hard-ste, the
    # forward path is hard from epoch 0 while the LUT logits still receive STE gradients.
    if args.lut_hard_ste:
        hard_epoch_start = 0
        current_tau = args.lut_tau_max
        hard_ratio = 1.0
    elif args.lut_anneal_epochs is not None:
        hard_epoch_start = min(max(int(args.lut_anneal_epochs), 0), num_epochs)
    else:
        hard_epoch_start = int(num_epochs * args.lut_hard_epoch_fraction)

    if not args.lut_hard_ste:
        tau_min, tau_max = args.lut_tau_min, args.lut_tau_max
        if epoch < hard_epoch_start:
            warmup_epochs = min(
                max(int(getattr(args, "lut_soft_warmup_epochs", 0)), 0),
                hard_epoch_start,
            )
            ratio = max(0, epoch - warmup_epochs) / max(
                1,
                hard_epoch_start - warmup_epochs,
            )
            current_tau = tau_min * math.pow((tau_max / tau_min), ratio)
            hard_ratio = 0.0
        else:
            current_tau = tau_max
            transition_epochs = max(int(args.lut_hard_transition_epochs), 0)
            if transition_epochs > 0:
                hard_ratio = min(1.0, (epoch - hard_epoch_start + 1) / transition_epochs)
            else:
                hard_ratio = 1.0

    for name, module in model.named_modules():
        if module.__class__.__name__ == "ComplexLUTConv2d" or module.__class__.__name__ == "UltimateComplexLUTConv2d":
            if hasattr(module, "tau") and hasattr(module, "hard"):
                module.tau.fill_(current_tau)
                module.hard.fill_(hard_ratio)

    is_fully_hard = hard_ratio >= 1.0

    if is_main and (epoch == 0 or epoch == hard_epoch_start or epoch % 10 == 0):
        if train_logger is not None:
            train_logger.info(
                f"[{('Phase2.1 Learned LUT5' if learned_lut5_phase2p1 else ('Phase3.1 Neural LUT5' if neural_lut5_phase3p1 else ('Phase3.1 Learned LUT5' if learned_lut5_phase3p1 else 'Phase 4')))} "
                f"Annealing Scheduler] Epoch {epoch}: "
                f"tau={current_tau:.4f}, hard_ratio={hard_ratio:.4f}, hard_mode={is_fully_hard}"
            )

    return bool(is_fully_hard)


def update_phase_metrics(workdir, phase, metrics):
    path = os.path.join(workdir, PHASE_METRICS_FILENAME)
    all_metrics = {}
    if os.path.isfile(path):
        with open(path, "r", encoding="utf-8") as f:
            all_metrics = json.load(f)
    all_metrics["phase{}".format(phase_tag(phase))] = metrics
    with open(path, "w", encoding="utf-8") as f:
        json.dump(all_metrics, f, indent=2, sort_keys=True)
    return path


def train(args):
    if args.phase not in PHASE_DESCRIPTIONS:
        raise ValueError("phase must be one of {}".format(sorted(PHASE_DESCRIPTIONS)))
    args.binary = args.phase >= 2

    learned_lut5_phase2p1 = (
        args.phase == 2.1 and args.phase2p1_mode == "learned_lut5"
    )
    learned_lut5_phase3p1 = (
        args.phase == 3.1
        and args.phase3p1_mode in (
            "learned_lut5",
            "semantic_lut5",
            "neural_lut5",
            "pure_mlp",
        )
    )
    semantic_lut5_phase3p1 = (
        args.phase == 3.1 and args.phase3p1_mode == "semantic_lut5"
    )
    neural_lut5_phase3p1 = (
        args.phase == 3.1 and args.phase3p1_mode == "neural_lut5"
    )
    pure_mlp_phase3p1 = (
        args.phase == 3.1 and args.phase3p1_mode == "pure_mlp"
    )
    fixed_phase3p1 = (
        args.phase == 3.1 and args.phase3p1_mode == "fixed"
    )
    magnitude_lut5_enabled = (
        args.phase == 4
        and args.lut_inputs == 5
        and args.lut_extra_bit == "magnitude_ste"
    )
    learned_lut_training_enabled = (
        learned_lut5_phase2p1
        or learned_lut5_phase3p1
        or magnitude_lut5_enabled
    )
    annealed_lut_training_enabled = (
        learned_lut5_phase2p1
        or magnitude_lut5_enabled
        or (
            learned_lut5_phase3p1
            and not pure_mlp_phase3p1
        )
    )
    learned_lut_stage_label = (
        "Phase4 Magnitude-STE LUT5"
        if magnitude_lut5_enabled
        else (
            "Phase3.1 Pure MLP5"
            if pure_mlp_phase3p1
            else (
                "Phase3.1 Semantic LUT5"
                if semantic_lut5_phase3p1
                else (
                    "Phase3.1 Neural LUT5"
                    if neural_lut5_phase3p1
                    else (
                        "Phase3.1 Learned LUT5"
                        if learned_lut5_phase3p1
                        else "Phase2.1 Learned LUT5"
                    )
                )
            )
        )
    )
    learned_lut_slice_label = (
        "magnitude"
        if magnitude_lut5_enabled
        else (
            "dominance"
            if semantic_lut5_phase3p1
            else "c8_gray_bit_0"
            if learned_lut5_phase3p1
            else "dominance"
        )
    )
    learned_lut_bit_names = (
        ("sign_r", "sign_i", "magnitude")
        if magnitude_lut5_enabled
        else (
            ("sign_r", "sign_i", "dominance")
            if semantic_lut5_phase3p1
            else ("c8_gray_bit_2", "c8_gray_bit_1", "c8_gray_bit_0")
            if learned_lut5_phase3p1
            else ("sign_r", "sign_i", "dominance")
        )
    )
    fixed_c8_phase_enabled = (
        (args.phase == 2.1 and args.phase2p1_mode == "c8")
        or args.phase == 3.1
    )
    if args.lut_extra_bit != "phase" and not magnitude_lut5_enabled:
        raise ValueError(
            "--lut-extra-bit magnitude_ste is only valid with "
            "--phase 4 --lut-inputs 5"
        )
    if magnitude_lut5_enabled:
        if args.train_from_scratch:
            raise ValueError(
                "magnitude_ste requires a Phase 3 checkpoint; remove "
                "--train-from-scratch"
            )
        if args.lut_init_mode != "binary":
            raise ValueError(
                "magnitude_ste requires --lut-init-mode binary"
            )
        if args.lut_sign_flip_prob != 0.0:
            raise ValueError(
                "magnitude_ste requires --lut-sign-flip-prob 0"
            )
        if args.lut_search_epochs > 0:
            raise ValueError(
                "magnitude_ste uses joint annealing and cannot be combined "
                "with hard-flip LUT search"
            )
        if args.magnitude_threshold_lr <= 0.0:
            raise ValueError("magnitude_threshold_lr must be positive")
        if args.magnitude_threshold_init <= 0.0:
            raise ValueError("magnitude_threshold_init must be positive")
        if args.magnitude_bit_beta <= 0.0:
            raise ValueError("magnitude_bit_beta must be positive")
        if not 0.0 < args.magnitude_shadow_epsilon < 0.5:
            raise ValueError(
                "magnitude_shadow_epsilon must be in (0, 0.5)"
            )
    if args.phase != 2.1 and args.phase2p1_mode != "c8":
        raise ValueError(
            "--phase2p1-mode learned_lut5 is only valid with --phase 2.1"
        )
    if args.phase != 3.1 and args.phase3p1_mode != "fixed":
        raise ValueError(
            "--phase3p1-mode learned_lut5/semantic_lut5/neural_lut5/"
            "pure_mlp is only valid with --phase 3.1"
        )
    phase3p1_transition_enabled = (
        fixed_phase3p1 and args.phase3p1_transition_epochs > 0
    )
    if fixed_c8_phase_enabled:
        if args.lut_inputs != 5:
            raise ValueError("Phase2.1 and Phase3.1 require --lut-inputs 5")
        if args.train_from_scratch:
            raise ValueError(
                "Phase2.1 must initialize from Phase1 and Phase3.1 must "
                "initialize from Phase2.1"
            )
        if args.bireal_tune:
            raise ValueError(
                "Phase2.1 and Phase3.1 do not support --bireal-tune"
            )
        if (
            not learned_lut5_phase3p1
            and args.lut_sign_flip_prob != 0.0
        ):
            raise ValueError(
                "Phase2.1 and Phase3.1 have no trainable LUT table; "
                "set --lut-sign-flip-prob 0"
            )
    if learned_lut5_phase2p1:
        if args.lut_inputs != 5:
            raise ValueError(
                "Phase2.1 learned_lut5 requires --lut-inputs 5"
            )
        if args.train_from_scratch:
            raise ValueError(
                "Phase2.1 learned_lut5 must initialize spatial weights from "
                "a Phase1 checkpoint"
            )
        if args.bireal_tune:
            raise ValueError(
                "Phase2.1 learned_lut5 does not support --bireal-tune"
            )
        if args.schedule not in ("constant", "cosine"):
            raise ValueError(
                "Phase2.1 learned_lut5 requires --schedule constant or cosine"
            )
        if args.lut_init_mode != "raw":
            raise ValueError(
                "Phase2.1 learned_lut5 requires --lut-init-mode raw so the "
                "duplicated LUT4 tie entries start at logit 0"
            )
        if args.lut_sign_flip_prob != 0.0:
            raise ValueError(
                "Phase2.1 learned_lut5 requires --lut-sign-flip-prob 0 so "
                "the d=0 and d=1 slices start identical"
            )
    if learned_lut5_phase3p1:
        if args.lut_inputs != 5:
            raise ValueError(
                "Phase3.1 learned_lut5 requires --lut-inputs 5"
            )
        if args.train_from_scratch:
            raise ValueError(
                "Phase3.1 learned_lut5 requires a pretrained C8 Phase2.1 "
                "checkpoint"
            )
        if args.bireal_tune:
            raise ValueError(
                "Phase3.1 learned_lut5 does not support --bireal-tune"
            )
        if args.schedule not in ("constant", "cosine"):
            raise ValueError(
                "Phase3.1 learned_lut5 requires --schedule constant or cosine"
            )
        if args.lut_init_mode != "raw":
            raise ValueError(
                "Phase3.1 learned_lut5 requires --lut-init-mode raw"
            )
        if args.lut_sign_flip_prob != 0.0:
            raise ValueError(
                "Phase3.1 learned_lut5 requires --lut-sign-flip-prob 0"
            )
        if args.phase3p1_transition_epochs != 0:
            raise ValueError(
                "Phase3.1 learned_lut5 uses LUT annealing; set "
                "--phase3p1-transition-epochs 0"
            )
        if args.phase3p1_padding_mode != "low_code":
            raise ValueError(
                "Phase3.1 learned_lut5 currently supports only "
                "--phase3p1-padding-mode low_code"
            )
        if semantic_lut5_phase3p1:
            if args.c8_codebook != "octants":
                raise ValueError(
                    "Phase3.1 semantic_lut5 requires --c8-codebook octants"
                )
            if args.c8_grad_mode != "semantic_ste":
                raise ValueError(
                    "Phase3.1 semantic_lut5 requires "
                    "--c8-grad-mode semantic_ste"
                )
        if pure_mlp_phase3p1:
            if args.neural_lut_hidden < 32:
                raise ValueError(
                    "Phase3.1 pure_mlp requires --neural-lut-hidden >= 32 "
                    "to exactly preserve all 32 C8/weight states at "
                    "initialization"
                )
            if args.c8_codebook != "octants":
                raise ValueError(
                    "Phase3.1 pure_mlp requires --c8-codebook octants"
                )
            if args.lut_anneal_epochs is not None:
                raise ValueError(
                    "Phase3.1 pure_mlp has no LUT annealing; omit "
                    "--lut-anneal-epochs"
                )
            if args.lut_soft_warmup_epochs != 0:
                raise ValueError(
                    "Phase3.1 pure_mlp has no LUT warmup; set "
                    "--lut-soft-warmup-epochs 0"
                )
            if args.lut_hard_transition_epochs != 0:
                raise ValueError(
                    "Phase3.1 pure_mlp maps to a LUT only for hard "
                    "evaluation; set --lut-hard-transition-epochs 0"
                )
            if args.lut_hard_ste:
                raise ValueError(
                    "Phase3.1 pure_mlp does not use LUT hard STE"
                )
    elif args.phase == 3.1:
        if (
            args.phase3p1_transition_epochs == 1
            or args.phase3p1_transition_epochs < 0
        ):
            raise ValueError(
                "phase3p1_transition_epochs must be 0 or at least 2"
            )
        if args.phase3p1_transition_epochs > args.num_epochs:
            raise ValueError(
                "phase3p1_transition_epochs must not exceed num_epochs"
            )
    elif args.phase3p1_transition_epochs != 0:
        raise ValueError(
            "--phase3p1-transition-epochs is only valid with --phase 3.1"
        )

    if args.lut_soft_warmup_epochs < 0:
        raise ValueError("lut_soft_warmup_epochs must be non-negative")
    if (
        args.lut_soft_warmup_epochs > 0
        and args.phase != 4
        and not learned_lut5_phase2p1
        and not annealed_lut_training_enabled
    ):
        raise ValueError(
            "--lut-soft-warmup-epochs is only valid for trainable LUT phases"
        )
    if (
        args.lut_anneal_epochs is not None
        and args.lut_soft_warmup_epochs > args.lut_anneal_epochs
    ):
        raise ValueError(
            "lut_soft_warmup_epochs must not exceed lut_anneal_epochs"
        )

    phase5_enabled = args.phase == 5
    if phase5_enabled:
        if args.lut_inputs != 5:
            raise ValueError("Phase 5 requires --lut-inputs 5")
        if args.train_from_scratch:
            raise ValueError(
                "Phase 5 requires a pretrained Phase 4 LUT4 checkpoint"
            )
        if args.bireal_tune:
            raise ValueError("Phase 5 does not support --bireal-tune")
        if args.no_validation:
            raise ValueError(
                "Phase 5 candidate acceptance requires a validation split"
            )
        if args.ddp:
            raise ValueError(
                "Phase 5 verified LUT search currently supports one GPU per run"
            )
        if args.compile:
            raise ValueError(
                "Phase 5 verified LUT search requires compile to remain disabled"
            )
        if args.schedule not in ("constant", "cosine"):
            raise ValueError(
                "Phase 5 requires --schedule constant or cosine"
            )
        if args.lut_search_epochs <= 0:
            raise ValueError("Phase 5 requires --lut-search-epochs > 0")
        if args.lut_sign_flip_prob != 0.0:
            raise ValueError(
                "Phase 5 starts from exact duplicated LUT4 slices; "
                "set --lut-sign-flip-prob 0"
            )
        if args.phase5_max_immediate_val_drop < 0.0:
            raise ValueError(
                "phase5_max_immediate_val_drop must be non-negative"
            )

    phase3p5_qat_enabled = args.phase == 3.5
    if phase3p5_qat_enabled:
        if args.lut_inputs != 5:
            raise ValueError("Phase3.5 activation QAT requires --lut-inputs 5")
        if args.train_from_scratch:
            raise ValueError(
                "Phase3.5 activation QAT requires a pretrained Phase2 checkpoint; "
                "--train-from-scratch is not supported"
            )
        if args.bireal_tune:
            raise ValueError("Phase3.5 activation QAT does not support --bireal-tune")
        if args.schedule not in ("constant", "cosine"):
            raise ValueError(
                "Phase3.5 activation QAT requires --schedule constant or cosine; "
                "the default schedule amplifies the learning rate"
            )
        if args.phase3p5_warmup_epochs < 0:
            raise ValueError("phase3p5_warmup_epochs must be non-negative")
        if args.phase3p5_transition_epochs is None:
            args.phase3p5_transition_epochs = max(2, args.num_epochs // 2)
        if args.phase3p5_transition_epochs < 2:
            raise ValueError("phase3p5_transition_epochs must be at least 2")
        if (
            args.phase3p5_warmup_epochs + args.phase3p5_transition_epochs
            > args.num_epochs
        ):
            raise ValueError(
                "phase3p5 warmup + transition epochs must not exceed num_epochs"
            )
        if args.phase3p5_beta_start <= 0.0 or args.phase3p5_beta_end <= 0.0:
            raise ValueError("phase3p5 beta values must be positive")
        if args.phase3p5_beta_end < args.phase3p5_beta_start:
            raise ValueError("phase3p5_beta_end must be >= phase3p5_beta_start")

    phase3p6_qat_enabled = args.phase == 3.6
    if phase3p6_qat_enabled:
        if args.lut_inputs != 5:
            raise ValueError("Phase3.6 C8 QAT requires --lut-inputs 5")
        if args.train_from_scratch:
            raise ValueError(
                "Phase3.6 C8 QAT requires a pretrained LUT4 Phase3 checkpoint"
            )
        if args.bireal_tune:
            raise ValueError("Phase3.6 C8 QAT does not support --bireal-tune")
        if args.schedule not in ("constant", "cosine"):
            raise ValueError(
                "Phase3.6 C8 QAT requires --schedule constant or cosine"
            )
        if args.lut_sign_flip_prob != 0.0:
            raise ValueError(
                "Phase3.6 has no trainable LUT table; set LUT flip "
                "probability to 0"
            )
        if args.phase3p6_warmup_epochs < 0:
            raise ValueError("phase3p6_warmup_epochs must be non-negative")
        if args.phase3p6_transition_epochs is None:
            args.phase3p6_transition_epochs = max(2, args.num_epochs // 2)
        if args.phase3p6_transition_epochs < 2:
            raise ValueError("phase3p6_transition_epochs must be at least 2")
        if (
            args.phase3p6_warmup_epochs + args.phase3p6_transition_epochs
            > args.num_epochs
        ):
            raise ValueError(
                "phase3p6 warmup + transition epochs must not exceed num_epochs"
            )
        if args.phase3p6_beta_start <= 0.0 or args.phase3p6_beta_end <= 0.0:
            raise ValueError("phase3p6 beta values must be positive")
        if args.phase3p6_beta_end < args.phase3p6_beta_start:
            raise ValueError("phase3p6_beta_end must be >= phase3p6_beta_start")

    phase3_lut5_transition_enabled = args.phase == 3 and args.lut_inputs == 5
    if phase3_lut5_transition_enabled:
        if args.train_from_scratch:
            raise ValueError(
                "Phase3 LUT5 transition requires a LUT4 Phase3 checkpoint; "
                "--train-from-scratch is not supported"
            )
        if args.checkpoint is None:
            raise ValueError(
                "Phase3 LUT5 transition requires --checkpoint pointing to a "
                "LUT4 Bestmodel_phase3.pt"
            )
        if args.bireal_tune:
            raise ValueError(
                "Phase3 LUT5 transition does not support --bireal-tune"
            )
        if args.schedule not in ("constant", "cosine"):
            raise ValueError(
                "Phase3 LUT5 transition requires --schedule constant or cosine; "
                "the default schedule amplifies the learning rate"
            )
        if args.num_epochs < 2:
            raise ValueError("Phase3 LUT5 transition requires at least 2 epochs")
        if args.phase3_lut5_transition_epochs is None:
            args.phase3_lut5_transition_epochs = max(2, args.num_epochs // 2)
        if not 2 <= args.phase3_lut5_transition_epochs <= args.num_epochs:
            raise ValueError(
                "phase3_lut5_transition_epochs must be between 2 and num_epochs"
            )
    elif args.phase3_lut5_transition_epochs is not None:
        raise ValueError(
            "--phase3-lut5-transition-epochs is only valid with "
            "--phase 3 --lut-inputs 5"
        )

    if args.lut_hard_bn_only_epochs < 0:
        raise ValueError("lut_hard_bn_only_epochs must be non-negative")
    if args.lut_hard_bn_only_epochs > 0:
        if args.lut_soft_only_epochs > 0:
            raise ValueError("lut_hard_bn_only_epochs cannot be combined with lut_soft_only_epochs")
        args.lut_soft_only_epochs = args.lut_hard_bn_only_epochs
    if args.lut_soft_only_epochs < 0:
        raise ValueError("lut_soft_only_epochs must be non-negative")
    if args.lut_soft_only_epochs > 0:
        if args.phase != 4:
            raise ValueError("LUT soft-only staging is only available in Phase 4")
        if args.lut_soft_only_epochs >= args.num_epochs:
            raise ValueError(
                "lut_soft_only_epochs must leave at least one epoch for hard weight adaptation"
            )
        if args.lut_search_epochs > 0:
            raise ValueError(
                "lut_soft_only_epochs cannot be combined with lut_search_epochs"
            )
        if args.lut_hard_ste:
            raise ValueError(
                "lut_soft_only_epochs cannot be combined with lut_hard_ste"
            )
        if args.lut_hard_transition_epochs > 0:
            raise ValueError(
                "lut_soft_only staging uses one-time projection; set lut_hard_transition_epochs=0"
            )
    if args.lut_search_epochs < 0:
        raise ValueError("lut_search_epochs must be non-negative")
    if args.lut_schedule == "constant" and args.lut_lr is None:
        raise ValueError("lut_schedule=constant requires an explicit lut_lr")
    if args.lut_search_epochs > 0:
        if args.phase not in (4, 5):
            raise ValueError("LUT flip search is only available in Phase 4 or 5")
        if args.lut_search_epochs >= args.num_epochs:
            raise ValueError(
                "lut_search_epochs must leave at least one epoch for final hard-LUT weight adaptation"
            )
        if args.lut_search_weight_epochs_per_lut < 1:
            raise ValueError("lut_search_weight_epochs_per_lut must be at least 1")
        if args.lut_search_max_flips_per_commit < 1:
            raise ValueError("lut_search_max_flips_per_commit must be at least 1")
        if args.lut_search_max_flips_per_layer < 1:
            raise ValueError("lut_search_max_flips_per_layer must be at least 1")
        if args.lut_search_flip_cooldown < 0:
            raise ValueError("lut_search_flip_cooldown must be non-negative")
        if not 0.0 < args.lut_search_init_flip_prob < 0.5:
            raise ValueError("lut_search_init_flip_prob must be between 0 and 0.5")
        if args.lut_search_score_temperature <= 0.0:
            raise ValueError("lut_search_score_temperature must be positive")
        if not 0.5 < args.lut_search_commit_threshold < 1.0:
            raise ValueError("lut_search_commit_threshold must be between 0.5 and 1.0")
        if args.lut_search_sparsity < 0.0:
            raise ValueError("lut_search_sparsity must be non-negative")
        if phase5_enabled:
            cycle_epochs = args.lut_search_weight_epochs_per_lut + 1
            if args.lut_search_epochs % cycle_epochs != 0:
                raise ValueError(
                    "Phase 5 lut_search_epochs must be divisible by its "
                    "1+weight-recovery cycle length ({})".format(cycle_epochs)
                )

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
    summary.append("Phase2.1 Mode:           " + str(args.phase2p1_mode))
    summary.append("Phase3.1 Mode:           " + str(args.phase3p1_mode))
    summary.append("Train From Scratch:      " + str(args.train_from_scratch))
    summary.append("LUT Inputs:              " + str(args.lut_inputs))
    if learned_lut5_phase2p1:
        lut5_activation_description = (
            "hard sign_r/sign_i plus STE dominance; direct learned 5-to-2 LUT"
        )
    elif pure_mlp_phase3p1:
        lut5_activation_description = (
            "3-bit Gray-coded C8 phase; pure 5-to-hidden-to-2 MLP operation"
        )
    elif neural_lut5_phase3p1:
        lut5_activation_description = (
            "3-bit Gray-coded C8 phase; residual neural 5-to-2 LUT generator"
        )
    elif semantic_lut5_phase3p1:
        lut5_activation_description = (
            "direct hard [sign_r, sign_i, dominance] with end-to-end semantic STE"
        )
    elif learned_lut5_phase3p1:
        lut5_activation_description = (
            "3-bit Gray-coded C8 phase; direct learned 5-to-2 LUT"
        )
    elif fixed_c8_phase_enabled:
        lut5_activation_description = (
            "3-bit Gray-coded nearest C8 roots"
            if args.c8_codebook == "roots"
            else "3-bit sign-preserving octant code with dominance"
        )
    elif phase3p6_qat_enabled:
        lut5_activation_description = "3-bit Gray-coded nearest-C8 phase index"
    elif magnitude_lut5_enabled:
        lut5_activation_description = (
            "1[abs(BN_pre(x)) >= theta_c], hard forward + sigmoid STE"
        )
    elif args.lut_inputs == 5:
        lut5_activation_description = "1[abs(x_r) >= abs(x_i)]"
    else:
        lut5_activation_description = "disabled"
    summary.append(
        "LUT5 Activation Code:    " + lut5_activation_description
    )
    summary.append("LUT Extra Bit Mode:      " + str(args.lut_extra_bit))
    summary.append(
        "Magnitude Threshold Init/LR/Schedule: {}/{}/{}".format(
            args.magnitude_threshold_init,
            args.magnitude_threshold_lr,
            args.magnitude_threshold_schedule,
        )
    )
    summary.append(
        "Magnitude Beta/Shadow:  {}/{}".format(
            args.magnitude_bit_beta,
            args.magnitude_shadow_epsilon,
        )
    )
    summary.append("C8 Codebook:             " + str(args.c8_codebook))
    summary.append("C8 Gradient Mode:        " + str(args.c8_grad_mode))
    summary.append("C8 QAT Beta:             " + str(args.c8_beta))
    summary.append(
        "Neural LUT Hidden/Scale: {}/{}".format(
            args.neural_lut_hidden,
            args.neural_lut_residual_scale,
        )
    )
    summary.append(
        "Neural LUT Hard Eval:    every {} epoch(s)".format(
            args.neural_lut_hard_eval_interval
        )
    )
    summary.append(
        "Pure MLP LUT Annealing:  "
        + ("disabled" if pure_mlp_phase3p1 else "n/a")
    )
    summary.append(
        "C8 Local Compare:        "
        + ("enabled" if fixed_phase3p1 else "disabled")
    )
    summary.append(
        "Phase3.1 Padding Mode:   " + str(args.phase3p1_padding_mode)
    )
    summary.append(
        "Phase3.1 Transition:     "
        + ("enabled" if phase3p1_transition_enabled else "disabled")
    )
    summary.append(
        "Phase3.1 Transition Epochs/Schedule: {}/{}".format(
            args.phase3p1_transition_epochs,
            args.phase3p1_transition_schedule,
        )
    )
    summary.append(
        "Phase3 LUT5 Transition: "
        + ("enabled" if phase3_lut5_transition_enabled else "disabled")
    )
    summary.append(
        "Phase3 LUT5 Mix Epochs: "
        + str(args.phase3_lut5_transition_epochs)
    )
    summary.append(
        "Phase3.5 Activation QAT: "
        + ("enabled" if phase3p5_qat_enabled else "disabled")
    )
    summary.append("Phase3.5 Warmup Epochs: " + str(args.phase3p5_warmup_epochs))
    summary.append(
        "Phase3.5 Transition:    " + str(args.phase3p5_transition_epochs)
    )
    summary.append(
        "Phase3.5 Beta Start/End: {}/{}".format(
            args.phase3p5_beta_start,
            args.phase3p5_beta_end,
        )
    )
    summary.append(
        "Phase3.6 C8 QAT:         "
        + ("enabled" if phase3p6_qat_enabled else "disabled")
    )
    summary.append(
        "Phase3.6 Training Op:    analytic decode/multiply/local-compare"
        if phase3p6_qat_enabled else "Phase3.6 Training Op:    disabled"
    )
    summary.append("Phase3.6 Warmup Epochs: " + str(args.phase3p6_warmup_epochs))
    summary.append(
        "Phase3.6 Transition:    " + str(args.phase3p6_transition_epochs)
    )
    summary.append(
        "Phase3.6 Beta Start/End: {}/{}".format(
            args.phase3p6_beta_start,
            args.phase3p6_beta_end,
        )
    )
    summary.append("Phase3.6 Code Schedule: " + str(args.phase3p6_schedule))
    summary.append("LUT Logit Init:          " + str(args.lut_logit_init))
    summary.append("LUT Init Mode:           " + str(args.lut_init_mode))
    summary.append("LUT Sign Flip Prob:      " + str(args.lut_sign_flip_prob))
    summary.append("LUT Learning Rate:       " + str(args.lut_lr))
    summary.append("LUT LR Schedule:         " + str(args.lut_schedule))
    summary.append("LUT Tau Min/Max:         {}/{}".format(args.lut_tau_min, args.lut_tau_max))
    summary.append("LUT Soft Warmup Epochs:  " + str(args.lut_soft_warmup_epochs))
    summary.append("LUT Anneal Epochs:       " + str(args.lut_anneal_epochs))
    summary.append("LUT Hard STE:            " + str(args.lut_hard_ste))
    summary.append("LUT Hard Epoch Fraction: " + str(args.lut_hard_epoch_fraction))
    summary.append("LUT Hard Transition:     " + str(args.lut_hard_transition_epochs))
    summary.append("LUT Soft-Only Epochs:    " + str(args.lut_soft_only_epochs))
    summary.append("LUT Hard+BN Epochs:      " + str(args.lut_hard_bn_only_epochs))
    summary.append("LUT Search Epochs:       " + str(args.lut_search_epochs))
    summary.append("Weight Epochs/LUT Epoch: " + str(args.lut_search_weight_epochs_per_lut))
    summary.append("LUT Max Flips/Commit:    " + str(args.lut_search_max_flips_per_commit))
    summary.append("LUT Max Flips/Layer:     " + str(args.lut_search_max_flips_per_layer))
    summary.append("LUT Flip Cooldown:       " + str(args.lut_search_flip_cooldown))
    summary.append("LUT Search Init Prob:    " + str(args.lut_search_init_flip_prob))
    summary.append("LUT Score Temperature:   " + str(args.lut_search_score_temperature))
    summary.append("LUT Commit Threshold:    " + str(args.lut_search_commit_threshold))
    summary.append("LUT Search Sparsity:     " + str(args.lut_search_sparsity))
    summary.append("LUT Search Freeze BN:    " + str(args.lut_search_freeze_bn))
    summary.append("Phase5 Anchor Slice:     " + str(args.phase5_anchor_slice))
    summary.append("Phase5 Ties Only:        " + str(args.phase5_ties_only))
    summary.append(
        "Phase5 Immediate Drop:   "
        + str(args.phase5_max_immediate_val_drop)
    )
    summary.append(
        "Phase5 Min Cycle Gain:   "
        + str(args.phase5_min_cycle_val_gain)
    )
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
    summary.append("Min LR Factor:           " + str(args.min_lr_factor))
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
    initialization_checkpoint_path = None
    phase5_initial_duplication_stats = None

    if args.phase > 1 and args.train_from_scratch:
        if is_main:
            train_logger.info(
                "==> Training phase {} from scratch; skipping previous-phase checkpoint initialization.".format(args.phase)
            )

    if args.phase > 1 and not args.train_from_scratch:
        previous_phases = (
            (3,)
            if magnitude_lut5_enabled
            else phase_initialization_sources(
                args.phase,
                args.lut_inputs,
            )
        )
        expected_phase = (
            previous_phases[0] if len(previous_phases) == 1 else previous_phases
        )
        selected_previous_phase = None
        if args.checkpoint is not None:
            ckpt_path = args.checkpoint
            if not os.path.isfile(ckpt_path):
                raise FileNotFoundError("Checkpoint not found: {}".format(ckpt_path))
        else:
            ckpt_path = None
            candidate_paths = []
            for candidate_phase in previous_phases:
                candidate_path = phase_checkpoint_path(
                    args.workdir, candidate_phase
                )
                candidate_paths.append(candidate_path)
                if os.path.isfile(candidate_path):
                    ckpt_path = candidate_path
                    selected_previous_phase = candidate_phase
                    break

            legacy_ckpt_path = legacy_best_checkpoint_path(args.workdir)
            if ckpt_path is None and os.path.isfile(legacy_ckpt_path):
                ckpt_path = legacy_ckpt_path
                if is_main:
                    train_logger.info(
                        "[Compatibility] Phase-specific checkpoint missing; "
                        "using legacy {}.".format(ckpt_path)
                    )
            if ckpt_path is None:
                raise FileNotFoundError(
                    "Phase {} expects a predecessor checkpoint at one of: {}".format(
                        args.phase, candidate_paths
                    )
                )

        initialization_checkpoint_path = ckpt_path
        predecessor_text = (
            str(selected_previous_phase)
            if selected_previous_phase is not None
            else " or ".join(str(phase) for phase in previous_phases)
        )
        if is_main:
            if phase3_lut5_transition_enabled:
                train_logger.info(
                    "==> Loading LUT4 Phase 3 initialization for the "
                    "phase-conditioned LUT5 transition from {} ...".format(
                        ckpt_path
                    )
                )
            else:
                train_logger.info(
                    "==> Loading Phase {} initialization for Phase {} from {} ...".format(
                        predecessor_text, args.phase, ckpt_path
                    )
                )
        load_phase_checkpoint(
            model=unwrapped_model,
            checkpoint_path=ckpt_path,
            device=device,
            expected_phase=expected_phase,
            expected_lut_inputs=(
                5
                if args.phase == 3.1
                else (
                    4
                    if (
                        phase3_lut5_transition_enabled
                        or phase3p6_qat_enabled
                        or phase5_enabled
                        or magnitude_lut5_enabled
                    )
                    else None
                )
            ),
            expected_c8_codebook=(
                args.c8_codebook if args.phase == 3.1 else None
            ),
            expected_c8_grad_mode=(
                args.c8_grad_mode if args.phase == 3.1 else None
            ),
            expected_phase2p1_mode=(
                "c8" if args.phase == 3.1 else None
            ),
            duplicate_lut4_to_lut5=phase5_enabled,
        )
        if phase5_enabled:
            phase5_initial_duplication_stats = verify_lut5_slices_duplicated(
                unwrapped_model
            )
            if is_main:
                train_logger.info(
                    "[Phase 5 Initialization] Duplicated Phase 4 LUT4 into "
                    "{} identical LUT5 slice pairs across {} modules; "
                    "value/sign mismatches=0, max_abs_diff={:.1f}.".format(
                        phase5_initial_duplication_stats["pairs"],
                        phase5_initial_duplication_stats["modules"],
                        phase5_initial_duplication_stats["max_abs_diff"],
                    )
                )
        if is_main:
            if phase3_lut5_transition_enabled:
                train_logger.info(
                    "==> Successfully initialized Phase3 LUT5 transition "
                    "from the LUT4 Phase3 checkpoint."
                )
            else:
                train_logger.info("==> Successfully initialized phase {} model from previous best checkpoint.".format(args.phase))

    if fixed_c8_phase_enabled:
        fixed_c8_stats = set_fixed_c8_qat_state(
            unwrapped_model,
            beta=args.c8_beta,
        )
        if is_main:
            train_logger.info(
                "[Fixed C8 QAT] Applied beta={:.4f} after checkpoint loading "
                "to {} activation layer(s) and {} local comparator layer(s). "
                "The hard C8 activation code is active from epoch 1.".format(
                    args.c8_beta,
                    fixed_c8_stats["activations"],
                    fixed_c8_stats["local_comparators"],
                )
            )

    if fixed_phase3p1:
        phase3p1_current_state = phase3p1_local_transition_state_for_epoch(
            0,
            args.phase3p1_transition_epochs,
            args.phase3p1_transition_schedule,
        )
        phase3p1_module_count = set_phase3p1_local_compare_progress(
            unwrapped_model,
            phase3p1_current_state["local_progress"],
        )
        if is_main:
            train_logger.info(
                "[Phase3.1 Local Transition] Initialized {} layer(s) at "
                "rho={:.6f}; padding_mode={}; transition_epochs={}; "
                "schedule={}. Only rho=1 epochs may save "
                "Bestmodel_phase3p1.pt.".format(
                    phase3p1_module_count,
                    phase3p1_current_state["local_progress"],
                    args.phase3p1_padding_mode,
                    args.phase3p1_transition_epochs,
                    args.phase3p1_transition_schedule,
                )
            )

    if phase3_lut5_transition_enabled:
        transition_module_count = set_phase3_lut5_mix(unwrapped_model, 0.0)
        if is_main:
            train_logger.info(
                "[Phase3 LUT5 Transition] Initialized {} layers at phase_mix=0. "
                "Epoch 1 is exactly LUT4; phase_mix reaches 1 at epoch {} and "
                "only fully hard LUT5 epochs may save Bestmodel_phase3.pt.".format(
                    transition_module_count,
                    args.phase3_lut5_transition_epochs,
                )
            )

    if phase3p5_qat_enabled:
        qat_module_count = set_phase3p5_qat_state(
            unwrapped_model,
            strength=0.0,
            beta=args.phase3p5_beta_start,
        )
        if is_main:
            train_logger.info(
                "[Phase3.5 QAT] Initialized {} layers at strength=0 and beta={:.4f}. "
                "The forward is exactly Phase2 BNN at the start, uses a hard "
                "phase comparator with sigmoid backward, reaches deployable LUT5 "
                "after {} warmup + {} transition epoch(s), and only then may save "
                "Bestmodel_phase3p5.pt.".format(
                    qat_module_count,
                    args.phase3p5_beta_start,
                    args.phase3p5_warmup_epochs,
                    args.phase3p5_transition_epochs,
                )
            )

    if phase3p6_qat_enabled:
        qat_module_count = set_phase3p6_qat_state(
            unwrapped_model,
            progress=0.0,
            beta=args.phase3p6_beta_start,
        )
        if is_main:
            train_logger.info(
                "[Phase3.6 C8 QAT] Initialized {} layers with only the four "
                "pretrained diagonal C4 codes enabled (progress=0, beta={:.4f}). "
                "Training uses the fixed analytic C8 decode, binary complex "
                "multiply, and per-local real/imag >=0 before accumulation; "
                "no LUT table is materialized or trained. Full C8 is reached "
                "after {} warmup + {} transition epoch(s), and only then may "
                "save Bestmodel_phase3p6.pt.".format(
                    qat_module_count,
                    args.phase3p6_beta_start,
                    args.phase3p6_warmup_epochs,
                    args.phase3p6_transition_epochs,
                )
            )

    learned_lut5_reference = (
        capture_lut_reference(unwrapped_model)
        if learned_lut_training_enabled
        else None
    )
    learned_lut5_last_change = None
    learned_lut5_last_divergence = None
    learned_lut5_last_sensitivity = None
    neural_lut_last_hard_projection = None
    pure_mlp_init_stats = None
    magnitude_gradient_tracker = (
        MagnitudeThresholdGradientTracker(unwrapped_model)
        if magnitude_lut5_enabled
        else None
    )
    magnitude_last_gradient = None
    magnitude_last_thresholds = (
        analyze_magnitude_thresholds(unwrapped_model)
        if magnitude_lut5_enabled
        else None
    )
    if learned_lut_training_enabled and is_main:
        initial_divergence = analyze_lut5_slice_divergence(unwrapped_model)
        initial_change = analyze_lut_change(
            unwrapped_model,
            learned_lut5_reference,
        )
        learned_lut5_last_sensitivity = (
            analyze_lut_activation_bit_sensitivity(
                unwrapped_model,
                learned_lut_bit_names,
            )
        )
        if pure_mlp_phase3p1:
            pure_mlp_modules = [
                module
                for module in iter_lut_modules(unwrapped_model)
                if getattr(module, "lut_parameterization", None) == "mlp"
            ]
            pure_mlp_init_stats = {
                "modules": len(pure_mlp_modules),
                "mse_max": max(
                    module.mlp_init_mse for module in pure_mlp_modules
                ),
                "max_abs_error": max(
                    module.mlp_init_max_abs_error
                    for module in pure_mlp_modules
                ),
                "hard_mismatches": sum(
                    module.mlp_init_hard_mismatches
                    for module in pure_mlp_modules
                ),
            }
        train_logger.info(
            "[{}] Initialized trainable 5-to-2 operation with "
            "strategy={}, parameterization={}, init_mode={}; {} slice "
            "hard_diff={} / {}, soft_abs_diff_mean={:.6f}, "
            "near_zero(<0.05)={} / {}. {}".format(
                learned_lut_stage_label,
                (
                    "semantic_c8_product"
                    if semantic_lut5_phase3p1
                    else "c8_product"
                    if learned_lut5_phase3p1
                    else "duplicate_lut4"
                ),
                (
                    "pure_mlp_5x{}x2".format(
                        args.neural_lut_hidden
                    )
                    if pure_mlp_phase3p1
                    else (
                        "neural_residual_5x{}x2".format(
                            args.neural_lut_hidden
                        )
                        if neural_lut5_phase3p1
                        else "direct_entries"
                    )
                ),
                args.lut_init_mode,
                learned_lut_slice_label,
                initial_divergence["hard_slice_diff"],
                initial_divergence["pairs"],
                initial_divergence["soft_abs_diff_mean"],
                initial_change["near_zero"]["0.05"],
                initial_change["entries"],
                (
                    "The MLP is the only trainable local operation; its "
                    "teacher-fit initialization has mse_max={:.3e}, "
                    "max_abs_error={:.3e}, hard_mismatches={}. No LUT "
                    "parameters or tau schedule are present.".format(
                        pure_mlp_init_stats["mse_max"],
                        pure_mlp_init_stats["max_abs_error"],
                        pure_mlp_init_stats["hard_mismatches"],
                    )
                    if pure_mlp_phase3p1
                    else (
                        (
                            "The direct semantic table exactly represents the "
                            "C8 local product at tau_min. Its d=0/d=1 slices "
                            "are distinct from epoch 0, so dominance receives "
                            "task gradients without a backward-only shadow."
                            if semantic_lut5_phase3p1
                            else "The soft table exactly represents each valid "
                            "C8 local product at tau_min before annealing."
                        )
                        if learned_lut5_phase3p1
                        else (
                            (
                                "The physical magnitude slices initially "
                                "implement the same LUT4 operation. Their "
                                "hard forward is exact, while an asymmetric "
                                "same-sign shadow provides fifth-bit input "
                                "gradients only."
                            )
                            if magnitude_lut5_enabled
                            else (
                                "The two dominance slices initially implement "
                                "the same LUT4 operation and may separate during "
                                "training."
                            )
                        )
                    )
                ),
            )
        )
        train_logger.info(
            "[{} Bit Sensitivity] Epoch 0: {}.".format(
                learned_lut_stage_label,
                ", ".join(
                    "{}={}/{} ({:.6%})".format(
                        name,
                        learned_lut5_last_sensitivity["bits"][name][
                            "hard_diff"
                        ],
                        learned_lut5_last_sensitivity["bits"][name][
                            "pairs"
                        ],
                        learned_lut5_last_sensitivity["bits"][name][
                            "hard_diff_ratio"
                        ],
                    )
                    for name in learned_lut_bit_names
                ),
            )
        )

    lut_soft_then_weight_enabled = (
        args.phase == 4 and args.lut_soft_only_epochs > 0
    )
    lut_soft_reference = (
        capture_lut_reference(unwrapped_model)
        if lut_soft_then_weight_enabled
        else None
    )
    lut_soft_origin_reference = (
        capture_lut_unperturbed_reference(unwrapped_model)
        if lut_soft_then_weight_enabled
        else None
    )
    lut_soft_initial_perturbation_stats = (
        analyze_lut_change(unwrapped_model, lut_soft_origin_reference)
        if lut_soft_then_weight_enabled
        else None
    )
    lut_soft_projection_stats = None
    lut_soft_origin_stats = None
    lut_soft_projected = False
    if lut_soft_then_weight_enabled and is_main:
        train_logger.info(
            "[LUT Soft-Then-Weight] For the first {} epoch(s), optimize only "
            "continuous LUT parameters. Save LUTsoft_end_phase4.pt and inspect "
            "sign changes before one-time hard projection. If sign_diff=0, stop early; "
            "otherwise freeze hard LUT and reset the weight LR schedule for {} epoch(s).".format(
                args.lut_soft_only_epochs,
                args.num_epochs - args.lut_soft_only_epochs,
            )
        )
        train_logger.info(
            "[LUT Soft-Then-Weight] Initial perturbation vs unperturbed LUT: "
            "sign_diff={} / {} ({:.6f}).".format(
                lut_soft_initial_perturbation_stats["sign_diff"],
                lut_soft_initial_perturbation_stats["entries"],
                lut_soft_initial_perturbation_stats["sign_diff_ratio"],
            )
        )

    lut_search_enabled = args.phase in (4, 5) and args.lut_search_epochs > 0
    lut_search_microcommit_stats = None
    lut_search_microcommit_history = []
    lut_search_final_stats = None
    lut_score_collection_active = False
    lut_search_cycle_epochs = args.lut_search_weight_epochs_per_lut + 1
    phase5_cycle_history = []
    phase5_cycle_snapshot = None
    phase5_cycle_baseline_metrics = None
    phase5_cycle_selected_entries = []
    phase5_cycle_commit_history_index = None
    phase5_pending_cycle_record = None
    if lut_search_enabled:
        module_count, entry_count, eligible_entry_count = begin_lut_flip_search(
            unwrapped_model,
            args.lut_search_init_flip_prob,
            args.lut_search_score_temperature,
            anchor_slice=(args.phase5_anchor_slice if phase5_enabled else None),
            ties_only=(args.phase5_ties_only if phase5_enabled else False),
        )
        lut_score_collection_active = True
        if is_main:
            train_logger.info(
                "[LUT Search] Started hard-forward/soft-backward flip search for "
                "{} modules / {} real+imag entries ({} eligible). For {} epochs, each cycle uses "
                "one LUT score/micro-commit epoch followed by {} weight recovery epoch(s); "
                "each commit is capped at {} globally and {} per layer.".format(
                    module_count,
                    entry_count,
                    eligible_entry_count,
                    args.lut_search_epochs,
                    args.lut_search_weight_epochs_per_lut,
                    args.lut_search_max_flips_per_commit,
                    args.lut_search_max_flips_per_layer,
                )
            )

    if phase5_enabled:
        baseline_val_loss, baseline_val_acc = evaluate(
            model, val_loader, device
        )
        phase5_cycle_baseline_metrics = {
            "val_loss": float(baseline_val_loss),
            "val_acc": float(baseline_val_acc),
        }
        train_logger.info(
            "[Phase 5 Baseline] anchor d={}, trainable d={}, ties_only={}; "
            "duplicated hard LUT5 val_loss={:.6f}, val_acc={:.4f}.".format(
                args.phase5_anchor_slice,
                1 - args.phase5_anchor_slice,
                args.phase5_ties_only,
                baseline_val_loss,
                baseline_val_acc,
            )
        )

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
    pure_mlp_best_hard_acc = float("-inf")
    pure_mlp_best_hard_metrics = None
    phase_extra_bit_strategy = (
        args.lut_inputs == 5
        and (
            args.phase in (3, 3.5, 4, 5)
            or learned_lut5_phase2p1
            or semantic_lut5_phase3p1
        )
    )
    phase_bit_last_occupancy = None
    phase_bit_occupancy_tracker = None
    c8_last_occupancy = None
    c8_occupancy_tracker = None
    if phase_extra_bit_strategy and is_main:
        if learned_lut5_phase2p1:
            train_logger.info(
                "[LUT5 Phase Bit] Phase2.1 learned_lut5 trains the direct "
                "hardware map [sign_r, sign_i, dominance, weight_r, weight_i] "
                "to two local output bits. All three activation bits use "
                "hard forward values with STE/multilinear LUT gradients."
            )
        elif semantic_lut5_phase3p1:
            train_logger.info(
                "[LUT5 Dominance Bit] Direct semantic address "
                "[sign_r, sign_i, dominance, weight_r, weight_i] is active. "
                "The forward bits are hard; sign bits use Bi-Real STE and "
                "dominance uses sigmoid semantic STE. Distinct analytic C8 "
                "slices carry gradients through dominance into BN_pre and "
                "the preceding backbone from the first batch."
            )
        elif phase3p5_qat_enabled:
            train_logger.info(
                "[LUT5 Phase Bit] Phase3.5 uses d=1[abs(x_r)>=abs(x_i)] "
                "with a hard forward and sigmoid comparator backward. "
                "Activation reconstruction moves continuously from (1,1) "
                "component magnitudes to deployable dominant (2,0) magnitudes."
            )
        elif magnitude_lut5_enabled:
            train_logger.info(
                "[LUT5 Magnitude Bit] d=1[abs(BN_pre(x)) >= theta_c] "
                "uses a hard forward, sigmoid STE backward, and trainable "
                "per-input-channel theta_c. The physical LUT5 starts from "
                "two identical LUT4 slices; a same-sign asymmetric shadow "
                "table is used only for activation-input gradients so d, "
                "theta_c, and the backbone receive task gradients from the "
                "first batch without changing forward outputs."
            )
        else:
            train_logger.info(
                "[LUT5 Phase Bit] Enabled deterministic routing "
                "d=1[abs(x_r)>=abs(x_i)]; LUT5 tie entries use component "
                "dominance to resolve the local complex-product sign."
            )
        phase_bit_occupancy_tracker = PhaseOccupancyTracker(model)

    if (fixed_c8_phase_enabled or phase3p6_qat_enabled) and is_main:
        if fixed_c8_phase_enabled:
            if args.phase == 2.1:
                c8_operator = (
                    "ordinary complex convolution; no local comparator"
                )
            elif learned_lut5_phase3p1:
                c8_operator = (
                    (
                        "pure 5-to-hidden-to-2 MLP operation initialized "
                        "by fitting the analytic C8 local product"
                        if pure_mlp_phase3p1
                        else (
                            "residual neural-generated 5-to-2 LUT initialized "
                            "from the analytic C8 local product"
                            if neural_lut5_phase3p1
                            else (
                                "differentiable direct-entry 5-to-2 LUT "
                                "initialized from the analytic C8 local product"
                            )
                        )
                    )
                )
            else:
                c8_operator = (
                    "analytic binary complex multiply with local >=0 "
                    "before accumulation"
                )
            c8_geometry = (
                "unit roots at 0/45-degree increments"
                if args.c8_codebook == "roots"
                else (
                    "octant centers preserving real/imag signs and "
                    "adding abs(real)>=abs(imag) dominance"
                )
            )
            train_logger.info(
                "[C8 Phase Code] Hard nearest-C8 enabled from epoch 1; "
                "codebook={} ({}); gradient_mode={}; beta={:.4f}; "
                "operator={}.".format(
                    args.c8_codebook,
                    c8_geometry,
                    args.c8_grad_mode,
                    args.c8_beta,
                    c8_operator,
                )
            )
        else:
            train_logger.info(
                "[C8 Phase Code] Phase3.6 C4-to-C8 curriculum enabled; "
                "forward keeps the analytic local-comparator operator."
            )
        c8_occupancy_tracker = C8CodeOccupancyTracker(model)

    phase3_lut5_current_mix = None
    phase3p1_current_state = None
    phase3p5_current_state = None
    phase3p6_current_state = None
    for epoch in range(initial_epoch, args.num_epochs):
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)

        if fixed_phase3p1:
            phase3p1_current_state = (
                phase3p1_local_transition_state_for_epoch(
                    epoch,
                    args.phase3p1_transition_epochs,
                    args.phase3p1_transition_schedule,
                )
            )
            set_phase3p1_local_compare_progress(
                unwrapped_model,
                phase3p1_current_state["local_progress"],
            )
            if is_main and (
                epoch == 0
                or (epoch + 1) % 10 == 0
                or (
                    phase3p1_current_state["hard_local_compare_ready"]
                    and epoch + 1 == args.phase3p1_transition_epochs
                )
            ):
                train_logger.info(
                    "[Phase3.1 Local Transition] Epoch {} raw_progress={:.6f}, "
                    "rho={:.6f}, padding_mode={}, state={}.".format(
                        epoch + 1,
                        phase3p1_current_state["raw_progress"],
                        phase3p1_current_state["local_progress"],
                        args.phase3p1_padding_mode,
                        (
                            "deployable hard local comparator"
                            if phase3p1_current_state[
                                "hard_local_compare_ready"
                            ]
                            else "continuous Phase2.1-to-local transition"
                        ),
                    )
                )

        if phase3_lut5_transition_enabled:
            phase3_lut5_current_mix = phase3_lut5_mix_for_epoch(
                epoch,
                args.phase3_lut5_transition_epochs,
            )
            set_phase3_lut5_mix(
                unwrapped_model,
                phase3_lut5_current_mix,
            )
            if is_main and (
                epoch == 0
                or (epoch + 1) % 10 == 0
                or (
                    phase3_lut5_current_mix >= 1.0
                    and epoch + 1 == args.phase3_lut5_transition_epochs
                )
            ):
                train_logger.info(
                    "[Phase3 LUT5 Transition] Epoch {} phase_mix={:.6f} "
                    "({}).".format(
                        epoch + 1,
                        phase3_lut5_current_mix,
                        "hard LUT5" if phase3_lut5_current_mix >= 1.0 else "continuous transition",
                    )
                )

        if phase3p5_qat_enabled:
            phase3p5_current_state = phase3p5_qat_state_for_epoch(
                epoch,
                args.phase3p5_warmup_epochs,
                args.phase3p5_transition_epochs,
                args.phase3p5_beta_start,
                args.phase3p5_beta_end,
            )
            set_phase3p5_qat_state(
                unwrapped_model,
                phase3p5_current_state["strength"],
                phase3p5_current_state["beta"],
            )
            if is_main and (
                epoch == 0
                or (epoch + 1) % 10 == 0
                or phase3p5_current_state["hard_lut5_ready"]
                and epoch
                == args.phase3p5_warmup_epochs
                + args.phase3p5_transition_epochs
                - 1
            ):
                train_logger.info(
                    "[Phase3.5 QAT] Epoch {} progress={:.6f}, "
                    "strength={:.6f}, beta={:.6f}, state={}.".format(
                        epoch + 1,
                        phase3p5_current_state["progress"],
                        phase3p5_current_state["strength"],
                        phase3p5_current_state["beta"],
                        (
                            "deployable hard LUT5"
                            if phase3p5_current_state["hard_lut5_ready"]
                            else "activation-QAT transition"
                        ),
                    )
                )

        if phase3p6_qat_enabled:
            phase3p6_current_state = phase3p6_qat_state_for_epoch(
                epoch,
                args.phase3p6_warmup_epochs,
                args.phase3p6_transition_epochs,
                args.phase3p6_beta_start,
                args.phase3p6_beta_end,
                args.phase3p6_schedule,
            )
            set_phase3p6_qat_state(
                unwrapped_model,
                phase3p6_current_state["code_progress"],
                phase3p6_current_state["beta"],
            )
            if is_main and (
                epoch == 0
                or (epoch + 1) % 10 == 0
                or (
                    phase3p6_current_state["hard_c8_ready"]
                    and epoch + 1
                    == args.phase3p6_warmup_epochs
                    + args.phase3p6_transition_epochs
                )
            ):
                train_logger.info(
                    "[Phase3.6 C8 QAT] Epoch {} raw_progress={:.6f}, "
                    "code_progress={:.6f}, beta={:.6f}, state={}.".format(
                        epoch + 1,
                        phase3p6_current_state["raw_progress"],
                        phase3p6_current_state["code_progress"],
                        phase3p6_current_state["beta"],
                        (
                            "full C8 analytic LUT-aware operation"
                            if phase3p6_current_state["hard_c8_ready"]
                            else "nested C4-to-C8 transition"
                        ),
                    )
                )

        if lut_soft_then_weight_enabled:
            lut_soft_stage_active = epoch < args.lut_soft_only_epochs
            if lut_soft_stage_active:
                schedule_epoch = epoch
                schedule_epochs = args.lut_soft_only_epochs
            else:
                schedule_epoch = epoch - args.lut_soft_only_epochs
                schedule_epochs = args.num_epochs - args.lut_soft_only_epochs
            lr = get_lr_for_epoch(
                schedule_epoch,
                args,
                num_epochs=schedule_epochs,
            )
        else:
            lut_soft_stage_active = False
            lr = get_lr_for_epoch(epoch, args)

        if lut_soft_then_weight_enabled:
            lut_training_epoch = False
            if lut_soft_stage_active:
                update_lut_soft_only_stage(
                    unwrapped_model,
                    epoch,
                    args.lut_soft_only_epochs,
                    args.lut_tau_min,
                    args.lut_tau_max,
                    train_logger=train_logger if is_main else None,
                    hard_forward=args.lut_hard_bn_only_epochs > 0,
                )
            inactive_optimizer_groups = set_lut_soft_then_weight_stage_lr(
                optimizer,
                lr,
                lut_stage=lut_soft_stage_active,
                lut_lr=args.lut_lr,
                lut_schedule=args.lut_schedule,
                train_bn=args.lut_hard_bn_only_epochs > 0,
            )
            phase_can_save_best = not lut_soft_stage_active
            if is_main and (
                epoch == 0
                or epoch == args.lut_soft_only_epochs
                or (epoch + 1) % 10 == 0
                or epoch == args.num_epochs - 1
            ):
                active_lrs = {
                    group.get("name", ""): group["lr"]
                    for group in optimizer.param_groups
                    if group["lr"] > 0.0
                }
                active_stage = (
                    "soft LUT-only exploration"
                    if lut_soft_stage_active
                    else "fixed hard-LUT weight adaptation"
                )
                train_logger.info(
                    "[LUT Soft-Then-Weight] Epoch {} optimizes {}; "
                    "stage-local learning rates: {}".format(
                        epoch + 1,
                        active_stage,
                        active_lrs,
                    )
                )
        elif lut_search_enabled:
            if epoch == args.lut_search_epochs:
                if phase5_enabled and (
                    phase5_cycle_snapshot is not None
                    or phase5_pending_cycle_record is not None
                ):
                    raise RuntimeError("Phase 5 search ended with an unresolved cycle")
                lut_search_final_stats = finalize_lut_flip_search(
                    unwrapped_model,
                    args.lut_logit_init,
                )
                lut_score_collection_active = False
                reset_optimizer_state(optimizer)
                if is_main:
                    train_logger.info(
                        "[LUT Search] Finalized hard LUT after {} alternating epochs: "
                        "net flips={} / {} (real={}, imag={}, ratio={:.6f}).".format(
                            args.lut_search_epochs,
                            lut_search_final_stats["net_total_flips"],
                            lut_search_final_stats["entries"],
                            lut_search_final_stats["net_real_flips"],
                            lut_search_final_stats["net_imag_flips"],
                            lut_search_final_stats["net_flip_ratio"],
                        )
                    )

            lut_training_epoch = (
                lut_score_collection_active and epoch % lut_search_cycle_epochs == 0
            )
            inactive_optimizer_groups = set_lut_search_stage_lr(
                optimizer,
                lr,
                searching=lut_training_epoch,
                lut_lr=args.lut_lr,
                lut_schedule=args.lut_schedule,
            )
            phase_can_save_best = not lut_score_collection_active
            active_stage = (
                "LUT flip-score + micro-commit"
                if lut_training_epoch
                else (
                    "weight tracking"
                    if lut_score_collection_active
                    else "final hard LUT weight adaptation"
                )
            )
            if is_main and (
                lut_score_collection_active or epoch == args.lut_search_epochs
            ):
                active_lrs = {
                    group.get("name", ""): group["lr"]
                    for group in optimizer.param_groups
                    if group["lr"] > 0.0
                }
                train_logger.info(
                    "[LUT Alternating] Epoch {} optimizes {}; active learning rates: {}".format(
                        epoch + 1,
                        active_stage,
                        active_lrs,
                    )
                )
        else:
            inactive_optimizer_groups = set()
            lut_training_epoch = False
            set_optimizer_lr(
                optimizer,
                lr,
                lut_lr=args.lut_lr,
                lut_schedule=args.lut_schedule,
            )
            if (
                is_main
                and args.schedule in ["constant", "cosine"]
                and epoch in {0, args.num_epochs // 2, args.num_epochs - 1}
            ):
                group_lrs = {
                    group.get("name", ""): group["lr"]
                    for group in optimizer.param_groups
                }
                train_logger.info(
                    "Current optimizer group learning rates: {}".format(group_lrs)
                )
            if args.schedule in ["default", "bireal"]:
                if is_main:
                    if args.schedule == "default" and epoch in [0, 10, 100, 120, 150]:
                        train_logger.info("Current learning rate value is {}".format(lr))
                        group_lrs = {
                            group.get("name", ""): group["lr"]
                            for group in optimizer.param_groups
                        }
                        train_logger.info(
                            "Current optimizer group learning rates: {}".format(group_lrs)
                        )
                    if args.schedule == "bireal":
                        warmup_epochs = 5
                        t1 = int(args.num_epochs * 0.5)
                        t2 = int(args.num_epochs * 0.75)
                        t3 = int(args.num_epochs * 0.875)
                        if epoch in [0, warmup_epochs - 1, t1, t2, t3]:
                            train_logger.info("Current learning rate value is {}".format(lr))

            phase_can_save_best = True
            if args.phase == 4 or annealed_lut_training_enabled:
                phase_can_save_best = update_lut_annealing(
                    model=model,
                    epoch=epoch,
                    num_epochs=args.num_epochs,
                    phase=args.phase,
                    train_logger=train_logger if is_main else None,
                    is_main=is_main,
                    args=args
                )
        if phase3_lut5_transition_enabled:
            phase_can_save_best = (
                phase_can_save_best and phase3_lut5_current_mix >= 1.0
            )
        if fixed_phase3p1:
            phase_can_save_best = (
                phase_can_save_best
                and phase3p1_current_state["hard_local_compare_ready"]
            )
        if phase3p5_qat_enabled:
            phase_can_save_best = (
                phase_can_save_best
                and phase3p5_current_state["hard_lut5_ready"]
            )
        if phase3p6_qat_enabled:
            phase_can_save_best = (
                phase_can_save_best
                and phase3p6_current_state["hard_c8_ready"]
            )
        if phase5_enabled and lut_score_collection_active and lut_training_epoch:
            if phase5_cycle_snapshot is not None:
                raise RuntimeError("Phase 5 started a new cycle before resolving the previous one")
            phase5_cycle_snapshot = capture_lut_search_cycle_snapshot(
                unwrapped_model
            )
            phase5_cycle_selected_entries = []
            phase5_cycle_commit_history_index = None
            train_logger.info(
                "[Phase 5 Cycle {}] Captured baseline val_acc={:.4f} before "
                "LUT proposal scoring.".format(
                    epoch // lut_search_cycle_epochs + 1,
                    phase5_cycle_baseline_metrics["val_acc"],
                )
            )
        t0 = time.time()
        if magnitude_gradient_tracker is not None:
            magnitude_gradient_tracker.reset()
        train_loss_sum, train_correct, train_count = train_one_epoch(
            model,
            train_loader,
            optimizer,
            device,
            clipnorm=args.clipnorm,
            clipval=args.clipval,
            inactive_optimizer_groups=inactive_optimizer_groups,
            freeze_batchnorm_stats=(lut_training_epoch and args.lut_search_freeze_bn),
            lut_search_sparsity=(args.lut_search_sparsity if lut_training_epoch else 0.0),
            magnitude_gradient_tracker=magnitude_gradient_tracker,
        )
        if magnitude_gradient_tracker is not None:
            magnitude_last_gradient = magnitude_gradient_tracker.finish()
            magnitude_last_thresholds = analyze_magnitude_thresholds(
                unwrapped_model
            )
        elapsed = time.time() - t0

        if lut_score_collection_active and lut_training_epoch:
            lut_search_microcommit_stats = microcommit_lut_flip_search(
                unwrapped_model,
                optimizer,
                threshold=args.lut_search_commit_threshold,
                max_flips=args.lut_search_max_flips_per_commit,
                max_flips_per_layer=args.lut_search_max_flips_per_layer,
                cooldown_commits=args.lut_search_flip_cooldown,
            )
            lut_search_microcommit_stats["epoch"] = epoch + 1
            if phase5_enabled:
                phase5_cycle_selected_entries = [
                    dict(entry)
                    for entry in lut_search_microcommit_stats["selected_entries"]
                ]
                lut_search_microcommit_stats["phase5_cycle"] = (
                    epoch // lut_search_cycle_epochs + 1
                )
                lut_search_microcommit_stats["phase5_status"] = "pending"
            lut_search_microcommit_history.append(dict(lut_search_microcommit_stats))
            if phase5_enabled:
                phase5_cycle_commit_history_index = len(lut_search_microcommit_history) - 1
            if is_main:
                min_prob = lut_search_microcommit_stats["min_committed_prob"]
                min_prob_text = "none" if min_prob is None else "{:.6f}".format(min_prob)
                train_logger.info(
                    "[LUT Micro-Commit] Epoch {} flipped {} / {} entries "
                    "(real={}, imag={}, eligible={}, layer_limited={}, "
                    "mean_prob={:.6f}, max_prob={:.6f}, min_committed_prob={}).".format(
                        epoch + 1,
                        lut_search_microcommit_stats["total_flips"],
                        lut_search_microcommit_stats["entries"],
                        lut_search_microcommit_stats["real_flips"],
                        lut_search_microcommit_stats["imag_flips"],
                        lut_search_microcommit_stats["eligible_candidates"],
                        lut_search_microcommit_stats["layer_limited_candidates"],
                        lut_search_microcommit_stats["mean_flip_prob"],
                        lut_search_microcommit_stats["max_flip_prob"],
                        min_prob_text,
                    )
                )

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

        if phase5_enabled and lut_score_collection_active:
            cycle_number = epoch // lut_search_cycle_epochs + 1
            cycle_end = (
                not lut_training_epoch
                and epoch % lut_search_cycle_epochs == lut_search_cycle_epochs - 1
            )
            if lut_training_epoch:
                candidate_val_loss = float(val_loss)
                candidate_val_acc = float(val_acc)
                phase5_pending_cycle_record = {
                    "cycle": cycle_number,
                    "score_epoch": epoch + 1,
                    "baseline_val_loss": phase5_cycle_baseline_metrics["val_loss"],
                    "baseline_val_acc": phase5_cycle_baseline_metrics["val_acc"],
                    "proposed_flips": lut_search_microcommit_stats["total_flips"],
                    "selected_entries": [
                        dict(entry) for entry in phase5_cycle_selected_entries
                    ],
                    "candidate_val_loss": candidate_val_loss,
                    "candidate_val_acc": candidate_val_acc,
                }
                immediate_drop = (
                    phase5_cycle_baseline_metrics["val_acc"] - candidate_val_acc
                )
                if lut_search_microcommit_stats["total_flips"] == 0:
                    candidate_status = "no_candidate"
                    train_logger.info(
                        "[Phase 5 Cycle {}] No hard entry reached the commit "
                        "threshold; continuing with weight recovery.".format(
                            cycle_number
                        )
                    )
                elif immediate_drop > args.phase5_max_immediate_val_drop:
                    restore_lut_search_cycle_snapshot(
                        unwrapped_model, phase5_cycle_snapshot
                    )
                    cooldown_rejected_lut_entries(
                        unwrapped_model,
                        phase5_cycle_selected_entries,
                        args.lut_search_flip_cooldown,
                    )
                    reset_optimizer_state(optimizer)
                    val_loss, val_acc = evaluate(model, val_loader, device)
                    candidate_status = "rejected_immediate"
                    phase5_pending_cycle_record.update(
                        {
                            "immediate_drop": float(immediate_drop),
                            "restored_val_loss": float(val_loss),
                            "restored_val_acc": float(val_acc),
                        }
                    )
                    train_logger.info(
                        "[Phase 5 Cycle {}] Rejected {} hard flip(s) immediately: "
                        "candidate val_acc={:.4f}, baseline={:.4f}, restored={:.4f}.".format(
                            cycle_number,
                            lut_search_microcommit_stats["total_flips"],
                            candidate_val_acc,
                            phase5_cycle_baseline_metrics["val_acc"],
                            val_acc,
                        )
                    )
                else:
                    candidate_status = "provisional"
                    phase5_pending_cycle_record["immediate_drop"] = float(
                        immediate_drop
                    )
                    train_logger.info(
                        "[Phase 5 Cycle {}] Provisionally accepted {} hard flip(s): "
                        "val_acc={:.4f}, drop={:.4f}; starting {} recovery epoch(s).".format(
                            cycle_number,
                            lut_search_microcommit_stats["total_flips"],
                            val_acc,
                            immediate_drop,
                            args.lut_search_weight_epochs_per_lut,
                        )
                    )
                phase5_pending_cycle_record["candidate_status"] = candidate_status
                lut_search_microcommit_history[
                    phase5_cycle_commit_history_index
                ]["phase5_status"] = candidate_status

            elif cycle_end:
                if phase5_cycle_snapshot is None or phase5_pending_cycle_record is None:
                    raise RuntimeError(
                        "Phase 5 recovery ended without a captured candidate cycle"
                    )
                recovered_val_loss = float(val_loss)
                recovered_val_acc = float(val_acc)
                required_val_acc = (
                    phase5_cycle_baseline_metrics["val_acc"]
                    + args.phase5_min_cycle_val_gain
                )
                candidate_status = phase5_pending_cycle_record["candidate_status"]
                if recovered_val_acc >= required_val_acc:
                    if candidate_status == "provisional":
                        final_status = "accepted"
                    elif candidate_status == "rejected_immediate":
                        final_status = "candidate_rejected_weight_cycle_accepted"
                    else:
                        final_status = "no_candidate_weight_cycle_accepted"
                    phase5_cycle_baseline_metrics = {
                        "val_loss": recovered_val_loss,
                        "val_acc": recovered_val_acc,
                    }
                    train_logger.info(
                        "[Phase 5 Cycle {}] {} after recovery: val_acc={:.4f}, "
                        "required={:.4f}.".format(
                            cycle_number,
                            final_status,
                            recovered_val_acc,
                            required_val_acc,
                        )
                    )
                else:
                    restore_lut_search_cycle_snapshot(
                        unwrapped_model, phase5_cycle_snapshot
                    )
                    if phase5_cycle_selected_entries:
                        cooldown_rejected_lut_entries(
                            unwrapped_model,
                            phase5_cycle_selected_entries,
                            args.lut_search_flip_cooldown,
                        )
                    reset_optimizer_state(optimizer)
                    val_loss, val_acc = evaluate(model, val_loader, device)
                    final_status = "rolled_back_after_recovery"
                    train_logger.info(
                        "[Phase 5 Cycle {}] Rolled back complete cycle: "
                        "recovered val_acc={:.4f}, required={:.4f}, restored={:.4f}.".format(
                            cycle_number,
                            recovered_val_acc,
                            required_val_acc,
                            val_acc,
                        )
                    )
                phase5_pending_cycle_record.update(
                    {
                        "recovery_epoch": epoch + 1,
                        "recovered_val_loss": recovered_val_loss,
                        "recovered_val_acc": recovered_val_acc,
                        "required_val_acc": float(required_val_acc),
                        "final_val_loss": float(val_loss),
                        "final_val_acc": float(val_acc),
                        "status": final_status,
                    }
                )
                lut_search_microcommit_history[
                    phase5_cycle_commit_history_index
                ]["phase5_status"] = final_status
                phase5_cycle_history.append(dict(phase5_pending_cycle_record))
                phase5_cycle_snapshot = None
                phase5_pending_cycle_record = None
                phase5_cycle_selected_entries = []
                phase5_cycle_commit_history_index = None

        test_loss, test_acc = (0.0, 0.0)
        if test_loader is not None and is_main:
            if phase_bit_occupancy_tracker is not None:
                phase_bit_occupancy_tracker.start()
            if c8_occupancy_tracker is not None:
                c8_occupancy_tracker.start()
            try:
                test_loss, test_acc = evaluate(model, test_loader, device)
            finally:
                if phase_bit_occupancy_tracker is not None:
                    phase_bit_last_occupancy = (
                        phase_bit_occupancy_tracker.finish()
                    )
                if c8_occupancy_tracker is not None:
                    c8_last_occupancy = c8_occupancy_tracker.finish()

        hard_eval_interval = args.neural_lut_hard_eval_interval
        if (
            (neural_lut5_phase3p1 or pure_mlp_phase3p1)
            and is_main
            and (pure_mlp_phase3p1 or not phase_can_save_best)
            and hard_eval_interval > 0
            and (
                epoch == 0
                or (epoch + 1) % hard_eval_interval == 0
            )
        ):
            neural_lut_last_hard_projection = (
                evaluate_hard_lut_projection(
                    model,
                    val_loader,
                    test_loader,
                    device,
                )
            )
            neural_lut_last_hard_projection["epoch"] = epoch + 1
            train_logger.info(
                "[{} Hard Projection] Epoch {}: "
                "val_loss={:.6f}, val_acc={:.4f}, test_loss={:.6f}, "
                "test_acc={:.4f}. Soft training state restored.".format(
                    learned_lut_stage_label,
                    epoch + 1,
                    neural_lut_last_hard_projection["val_loss"],
                    neural_lut_last_hard_projection["val_acc"],
                    neural_lut_last_hard_projection["test_loss"],
                    neural_lut_last_hard_projection["test_acc"],
                )
            )
            if pure_mlp_phase3p1:
                hard_metric_acc = (
                    neural_lut_last_hard_projection["val_acc"]
                    if val_loader is not None
                    else neural_lut_last_hard_projection["test_acc"]
                )
                if hard_metric_acc > pure_mlp_best_hard_acc:
                    pure_mlp_best_hard_acc = hard_metric_acc
                    pure_mlp_best_hard_metrics = {
                        "phase": args.phase,
                        "phase_name": PHASE_DESCRIPTIONS[args.phase],
                        "epoch": epoch + 1,
                        "best_acc": float(hard_metric_acc),
                        "hard_mode": True,
                        "compiled_from_pure_mlp": True,
                        "source_checkpoint": initialization_checkpoint_path,
                        "input_bits": [
                            "c8_gray_bit_2",
                            "c8_gray_bit_1",
                            "c8_gray_bit_0",
                            "weight_r",
                            "weight_i",
                        ],
                        "output_bits": ["local_real", "local_imag"],
                        "truth_table_shape_per_output": [32],
                        "neural_hidden_width": args.neural_lut_hidden,
                        "hard_projection": dict(
                            neural_lut_last_hard_projection
                        ),
                    }
                    hard_model_state = {
                        name: value.detach().clone()
                        for name, value in unwrapped_model.state_dict().items()
                    }
                    for name, value in hard_model_state.items():
                        if name.endswith(".hard") and value.numel() == 1:
                            value.fill_(1.0)
                    hard_checkpoint = {
                        "phase": args.phase,
                        "model": hard_model_state,
                        "optimizer": optimizer.state_dict(),
                        "epoch": epoch + 1,
                        "args": vars(args),
                        "metrics": pure_mlp_best_hard_metrics,
                        "hard_lut_tables": capture_hard_lut_tables(
                            unwrapped_model
                        ),
                    }
                    hard_path = save_checkpoint(
                        hard_checkpoint,
                        args.workdir,
                        "BestHardmodel_phase3p1_mlp_lut5.pt",
                    )
                    train_logger.info(
                        "[Phase3.1 Pure MLP5] Saved best compiled hard "
                        "LUT5 checkpoint to {} at epoch {}.".format(
                            hard_path,
                            epoch + 1,
                        )
                    )

        train_loss_hist.append(train_loss)
        train_acc_hist.append(train_acc)
        val_loss_hist.append(val_loss)
        val_acc_hist.append(val_acc)
        test_loss_hist.append(test_loss)
        test_acc_hist.append(test_acc)

        if is_main:
            if phase_bit_last_occupancy is not None:
                layer_ratios = [
                    layer["one_ratio"]
                    for layer in phase_bit_last_occupancy["layers"]
                ]
                train_logger.info(
                    "[LUT5 {} Occupancy] Epoch {} test: "
                    "ones={}, zeros={}, one_ratio={:.6%}, "
                    "layer_ratio_range=[{:.6%}, {:.6%}], "
                    "mean_abs_margin={:.6f}, near_boundary_ratio={}.".format(
                        (
                            "Magnitude"
                            if magnitude_lut5_enabled
                            else "Phase"
                        ),
                        epoch + 1,
                        phase_bit_last_occupancy["ones"],
                        phase_bit_last_occupancy["zeros"],
                        phase_bit_last_occupancy["one_ratio"],
                        min(layer_ratios),
                        max(layer_ratios),
                        phase_bit_last_occupancy["mean_abs_margin"],
                        (
                            "n/a"
                            if phase_bit_last_occupancy["near_boundary_ratio"] is None
                            else "{:.6%}".format(
                                phase_bit_last_occupancy["near_boundary_ratio"]
                            )
                        ),
                    )
                )
            if magnitude_last_gradient is not None:
                train_logger.info(
                    "[Magnitude Fifth-Bit Gradient] Epoch {}: "
                    "nonzero_batches={}/{} ({:.6%}), "
                    "nonzero_elements={}/{} ({:.6%}), "
                    "mean_abs={:.6e}, max_abs={:.6e}; "
                    "physical_threshold[min/mean/max]="
                    "{:.6f}/{:.6f}/{:.6f}; shadow_epsilon={:.6f}.".format(
                        epoch + 1,
                        magnitude_last_gradient["batches_with_nonzero"],
                        magnitude_last_gradient["batches"],
                        magnitude_last_gradient["batch_nonzero_ratio"],
                        magnitude_last_gradient["nonzero_elements"],
                        magnitude_last_gradient["elements"],
                        magnitude_last_gradient["nonzero_ratio"],
                        magnitude_last_gradient["mean_abs"],
                        magnitude_last_gradient["max_abs"],
                        magnitude_last_thresholds["min"],
                        magnitude_last_thresholds["mean"],
                        magnitude_last_thresholds["max"],
                        args.magnitude_shadow_epsilon,
                    )
                )
            if c8_last_occupancy is not None:
                if c8_last_occupancy["codebook_mode"] == "roots":
                    secondary_ratio_name = "added_axis_ratio"
                    secondary_ratio = c8_last_occupancy[
                        "added_axis_ratio"
                    ]
                else:
                    secondary_ratio_name = "odd_octant_ratio"
                    secondary_ratio = c8_last_occupancy[
                        "odd_index_ratio"
                    ]
                train_logger.info(
                    "[C8 Code Occupancy] Epoch {} test: counts={}, "
                    "ratios={}, active_codes={}, {}={:.6%}, "
                    "normalized_entropy={:.6f}, mean_angle_error={:.4f}deg, "
                    "max_angle_error={:.4f}deg.".format(
                        epoch + 1,
                        c8_last_occupancy["counts"],
                        [round(value, 6) for value in c8_last_occupancy["ratios"]],
                        c8_last_occupancy["active_codes"],
                        secondary_ratio_name,
                        secondary_ratio,
                        c8_last_occupancy["normalized_entropy"],
                        c8_last_occupancy["mean_angle_error_deg"],
                        c8_last_occupancy["max_angle_error_deg"],
                    )
                )
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

            if learned_lut_training_enabled:
                learned_lut5_last_change = analyze_lut_change(
                    unwrapped_model,
                    learned_lut5_reference,
                )
                learned_lut5_last_divergence = (
                    analyze_lut5_slice_divergence(unwrapped_model)
                )
                learned_lut5_last_sensitivity = (
                    analyze_lut_activation_bit_sensitivity(
                        unwrapped_model,
                        learned_lut_bit_names,
                    )
                )
                if (
                    epoch == 0
                    or (epoch + 1) % 10 == 0
                    or epoch + 1 == args.num_epochs
                ):
                    train_logger.info(
                        "[{}] Epoch {}: "
                        "init_sign_diff={} / {} ({:.6%}), "
                        "{}_slice_hard_diff={} / {} ({:.6%}), "
                        "{}_slice_soft_abs_diff_mean={:.6f}, "
                        "max={:.6f}, "
                        "hard_checkpoint_eligible={}.".format(
                            learned_lut_stage_label,
                            epoch + 1,
                            learned_lut5_last_change["sign_diff"],
                            learned_lut5_last_change["entries"],
                            learned_lut5_last_change["sign_diff_ratio"],
                            learned_lut_slice_label,
                            learned_lut5_last_divergence[
                                "hard_slice_diff"
                            ],
                            learned_lut5_last_divergence["pairs"],
                            learned_lut5_last_divergence[
                                "hard_slice_diff_ratio"
                            ],
                            learned_lut_slice_label,
                            learned_lut5_last_divergence[
                                "soft_abs_diff_mean"
                            ],
                            learned_lut5_last_divergence[
                                "soft_abs_diff_max"
                            ],
                            phase_can_save_best,
                        )
                    )
                    train_logger.info(
                        "[{} Bit Sensitivity] Epoch {}: {}.".format(
                            learned_lut_stage_label,
                            epoch + 1,
                            ", ".join(
                                "{}={}/{} ({:.6%})".format(
                                    name,
                                    learned_lut5_last_sensitivity["bits"][
                                        name
                                    ]["hard_diff"],
                                    learned_lut5_last_sensitivity["bits"][
                                        name
                                    ]["pairs"],
                                    learned_lut5_last_sensitivity["bits"][
                                        name
                                    ]["hard_diff_ratio"],
                                )
                                for name in learned_lut_bit_names
                            ),
                        )
                    )


        lut_soft_early_stop = False
        if (
            lut_soft_then_weight_enabled
            and not lut_soft_projected
            and epoch + 1 == args.lut_soft_only_epochs
        ):
            lut_soft_projection_stats = analyze_lut_change(
                unwrapped_model,
                lut_soft_reference,
            )
            lut_soft_origin_stats = analyze_lut_change(
                unwrapped_model,
                lut_soft_origin_reference,
            )
            soft_stage_metrics = {
                "epoch": epoch + 1,
                "train_loss": float(train_loss),
                "train_acc": float(train_acc),
                "val_loss": float(val_loss),
                "val_acc": float(val_acc),
                "test_loss": float(test_loss),
                "test_acc": float(test_acc),
            }
            report = {
                "status": "soft_stage_complete",
                "soft_metrics": soft_stage_metrics,
                "lut_change": lut_soft_projection_stats,
                "lut_change_from_start": lut_soft_projection_stats,
                "lut_change_from_unperturbed_origin": lut_soft_origin_stats,
                "initial_perturbation": lut_soft_initial_perturbation_stats,
            }

            if is_main:
                soft_checkpoint = {
                    "phase": args.phase,
                    "stage": "lut_soft_end",
                    "model": unwrapped_model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "epoch": epoch + 1,
                    "args": vars(args),
                    "metrics": soft_stage_metrics,
                    "lut_change": lut_soft_projection_stats,
                    "lut_change_from_start": lut_soft_projection_stats,
                    "lut_change_from_unperturbed_origin": lut_soft_origin_stats,
                    "initial_perturbation": lut_soft_initial_perturbation_stats,
                }
                soft_path = save_checkpoint(
                    soft_checkpoint,
                    args.workdir,
                    "LUTsoft_end_phase4.pt",
                )
                train_logger.info(
                    "[LUT Soft Checkpoint] Saved {} after epoch {}.".format(
                        soft_path,
                        epoch + 1,
                    )
                )
                train_logger.info(
                    "[LUT Soft Checkpoint] sign_diff={} / {} ({:.6f}), "
                    "abs_delta_mean={:.6f}, abs_delta_max={:.6f}, "
                    "near_zero={}".format(
                        lut_soft_projection_stats["sign_diff"],
                        lut_soft_projection_stats["entries"],
                        lut_soft_projection_stats["sign_diff_ratio"],
                        lut_soft_projection_stats["abs_delta_mean"],
                        lut_soft_projection_stats["abs_delta_max"],
                        lut_soft_projection_stats["near_zero"],
                    )
                )
                train_logger.info(
                    "[LUT Soft Checkpoint] vs unperturbed origin: "
                    "sign_diff={} / {} ({:.6f}); initial perturbation was "
                    "{} / {} ({:.6f}).".format(
                        lut_soft_origin_stats["sign_diff"],
                        lut_soft_origin_stats["entries"],
                        lut_soft_origin_stats["sign_diff_ratio"],
                        lut_soft_initial_perturbation_stats["sign_diff"],
                        lut_soft_initial_perturbation_stats["entries"],
                        lut_soft_initial_perturbation_stats["sign_diff_ratio"],
                    )
                )
                for layer in lut_soft_projection_stats["layers"]:
                    train_logger.info(
                        "[LUT Soft Checkpoint][{}] sign_diff={} / {} "
                        "({:.6f}), abs_delta_mean={:.6f}, "
                        "abs_delta_max={:.6f}, near_zero={}".format(
                            layer["name"],
                            layer["sign_diff"],
                            layer["entries"],
                            layer["sign_diff_ratio"],
                            layer["abs_delta_mean"],
                            layer["abs_delta_max"],
                            layer["near_zero"],
                        )
                    )

            if ddp_enabled:
                dist.barrier()

            if lut_soft_projection_stats["sign_diff"] == 0:
                lut_soft_early_stop = True
                report["status"] = "stopped_no_lut_sign_change"
                if is_main:
                    train_logger.info(
                        "[LUT Soft-Then-Weight] sign_diff=0 after {} LUT-only "
                        "epoch(s). Stopping before hard projection and weight "
                        "adaptation because the truth table did not change.".format(
                            args.lut_soft_only_epochs
                        )
                    )
            else:
                materialize_hard_lut(
                    unwrapped_model,
                    args.lut_logit_init,
                )
                reset_optimizer_state(optimizer)
                lut_soft_projected = True
                report["status"] = "projected_hard_lut"
                hard_val_loss, hard_val_acc = (0.0, 0.0)
                if val_loader is not None and is_main:
                    hard_val_loss, hard_val_acc = evaluate(
                        model,
                        val_loader,
                        device,
                    )
                hard_test_loss, hard_test_acc = (0.0, 0.0)
                if test_loader is not None and is_main:
                    hard_test_loss, hard_test_acc = evaluate(
                        model,
                        test_loader,
                        device,
                    )
                if is_main:
                    report["projected_hard_metrics_stale_bn"] = {
                        "val_loss": float(hard_val_loss),
                        "val_acc": float(hard_val_acc),
                        "test_loss": float(hard_test_loss),
                        "test_acc": float(hard_test_acc),
                    }
                    train_logger.info(
                        "[LUT Soft-Then-Weight] One-time hard projection complete. "
                        "Before weight adaptation / with stale BN: "
                        "val_acc={:.4f}, test_acc={:.4f}.".format(
                            hard_val_acc,
                            hard_test_acc,
                        )
                    )

            if is_main:
                report_path = os.path.join(
                    args.workdir,
                    "lut_soft_stage_report.json",
                )
                with open(report_path, "w", encoding="utf-8") as f:
                    json.dump(report, f, indent=2, sort_keys=True)
                train_logger.info(
                    "[LUT Soft Checkpoint] Saved report to {}.".format(
                        report_path
                    )
                )

            if ddp_enabled:
                dist.barrier()
            if lut_soft_early_stop:
                break


        metric_loss, metric_acc = (val_loss, val_acc) if val_loader is not None else (test_loss, test_acc)
        if is_main and phase_can_save_best and metric_acc > best_acc:
            best_acc = metric_acc
            best_metrics = {
                "phase": args.phase,
                "phase_name": PHASE_DESCRIPTIONS[args.phase],
                "lut_inputs": args.lut_inputs,
                "epoch": epoch + 1,
                "best_acc": float(metric_acc),
                "best_loss": float(metric_loss),
                "val_acc": float(val_acc),
                "val_loss": float(val_loss),
                "test_acc": float(test_acc),
                "test_loss": float(test_loss),
                "train_acc": float(train_acc),
                "train_loss": float(train_loss),
                "hard_mode": bool(
                    not pure_mlp_phase3p1
                    and (
                        args.phase != 4
                        and not learned_lut_training_enabled
                        or phase_can_save_best
                    )
                ),
            }
            if learned_lut5_phase2p1:
                best_metrics["phase2p1_learned_lut5"] = {
                    "source_checkpoint": initialization_checkpoint_path,
                    "source_phase": 1,
                    "input_bits": [
                        "sign_r",
                        "sign_i",
                        "dominance",
                        "weight_r",
                        "weight_i",
                    ],
                    "output_bits": [
                        "local_real",
                        "local_imag",
                    ],
                    "truth_table_shape_per_output": [32],
                    "lut5_init_strategy": "duplicate_lut4",
                    "lut_init_mode": args.lut_init_mode,
                    "hard_forward_ready": bool(phase_can_save_best),
                    "lut_change_from_start": dict(
                        learned_lut5_last_change
                    ),
                    "d_slice_divergence": dict(
                        learned_lut5_last_divergence
                    ),
                }
            if learned_lut5_phase3p1:
                best_metrics["phase3p1_learned_lut5"] = {
                    "source_checkpoint": initialization_checkpoint_path,
                    "source_phase": 2.1,
                    "input_bits": (
                        ["sign_r", "sign_i", "dominance"]
                        if semantic_lut5_phase3p1
                        else [
                            "c8_gray_bit_2",
                            "c8_gray_bit_1",
                            "c8_gray_bit_0",
                        ]
                    ) + ["weight_r", "weight_i"],
                    "output_bits": [
                        "local_real",
                        "local_imag",
                    ],
                    "truth_table_shape_per_output": [32],
                    "lut5_init_strategy": (
                        "semantic_c8_product"
                        if semantic_lut5_phase3p1
                        else "c8_product"
                    ),
                    "activation_address": (
                        "direct_semantic" if semantic_lut5_phase3p1 else "gray"
                    ),
                    "parameterization": (
                        "pure_mlp"
                        if pure_mlp_phase3p1
                        else (
                            "neural_residual"
                            if neural_lut5_phase3p1
                            else "direct_entries"
                        )
                    ),
                    "lut_init_mode": args.lut_init_mode,
                    "lut_init_tau": args.lut_tau_min,
                    "neural_hidden_width": (
                        args.neural_lut_hidden
                        if neural_lut5_phase3p1 or pure_mlp_phase3p1
                        else None
                    ),
                    "neural_residual_scale": (
                        args.neural_lut_residual_scale
                        if neural_lut5_phase3p1
                        else None
                    ),
                    "pure_mlp_initialization": (
                        dict(pure_mlp_init_stats)
                        if pure_mlp_init_stats is not None
                        else None
                    ),
                    "last_soft_stage_hard_projection": (
                        dict(neural_lut_last_hard_projection)
                        if neural_lut_last_hard_projection is not None
                        else None
                    ),
                    "soft_warmup_epochs": args.lut_soft_warmup_epochs,
                    "anneal_epochs": args.lut_anneal_epochs,
                    "hard_transition_epochs": (
                        args.lut_hard_transition_epochs
                    ),
                    "hard_forward_ready": bool(
                        phase_can_save_best and not pure_mlp_phase3p1
                    ),
                    "training_forward_hard": bool(
                        phase_can_save_best and not pure_mlp_phase3p1
                    ),
                    "compiled_hard_table_available": bool(
                        pure_mlp_phase3p1 or phase_can_save_best
                    ),
                    "lut_change_from_start": dict(
                        learned_lut5_last_change
                    ),
                    (
                        "dominance_slice_divergence"
                        if semantic_lut5_phase3p1
                        else "c8_gray_bit_0_slice_divergence"
                    ): dict(
                        learned_lut5_last_divergence
                    ),
                    "activation_bit_sensitivity": dict(
                        learned_lut5_last_sensitivity
                    ),
                    "jointly_trains_spatial_weights_and_lut": (
                        not pure_mlp_phase3p1
                    ),
                    "jointly_trains_spatial_weights_and_mlp": (
                        pure_mlp_phase3p1
                    ),
                    "trainable_table_entries": not pure_mlp_phase3p1,
                }
            if phase_extra_bit_strategy:
                best_metrics["lut5_phase_bit"] = {
                    "definition": (
                        "1[abs(BN_pre(x)) >= theta_c]"
                        if magnitude_lut5_enabled
                        else "1[abs(x_r) >= abs(x_i)]"
                    ),
                    "semantic": (
                        "learned activation magnitude threshold"
                        if magnitude_lut5_enabled
                        else "activation phase/component dominance"
                    ),
                    "bit_occupancy": (
                        None
                        if phase_bit_last_occupancy is None
                        else dict(phase_bit_last_occupancy)
                    ),
                }
                if magnitude_lut5_enabled:
                    best_metrics["lut5_phase_bit"].update({
                        "mode": "magnitude_ste",
                        "threshold_granularity": "input_channel",
                        "thresholds": dict(magnitude_last_thresholds),
                        "threshold_gradient": (
                            None
                            if magnitude_last_gradient is None
                            else dict(magnitude_last_gradient)
                        ),
                        "sigmoid_beta": args.magnitude_bit_beta,
                        "shadow_epsilon": args.magnitude_shadow_epsilon,
                        "physical_forward_uses_shadow": False,
                        "input_gradient_uses_shadow": True,
                        "backbone_detached_from_fifth_bit": False,
                    })
            if phase3_lut5_transition_enabled:
                best_metrics["phase3_lut5_transition"] = {
                    "source_checkpoint": initialization_checkpoint_path,
                    "source_phase": 3,
                    "source_lut_inputs": 4,
                    "transition_epochs": args.phase3_lut5_transition_epochs,
                    "phase_mix": float(phase3_lut5_current_mix),
                    "fully_hard_lut5": bool(
                        phase3_lut5_current_mix >= 1.0
                    ),
                }
            if phase3p5_qat_enabled:
                best_metrics["phase3p5_activation_qat"] = {
                    "source_checkpoint": initialization_checkpoint_path,
                    "source_phase": 2,
                    "warmup_epochs": args.phase3p5_warmup_epochs,
                    "transition_epochs": args.phase3p5_transition_epochs,
                    "progress": float(phase3p5_current_state["progress"]),
                    "strength": float(phase3p5_current_state["strength"]),
                    "beta": float(phase3p5_current_state["beta"]),
                    "hard_forward_soft_backward": True,
                    "fully_hard_lut5": bool(
                        phase3p5_current_state["hard_lut5_ready"]
                    ),
                    "activation_magnitudes_at_deployment": [2.0, 0.0],
                }

            if fixed_c8_phase_enabled:
                local_compare = fixed_phase3p1
                learned_c8_lut = learned_lut5_phase3p1
                best_metrics["c8_branch"] = {
                    "stage": phase_tag(args.phase),
                    "source_checkpoint": initialization_checkpoint_path,
                    "source_phase": 2.1 if args.phase == 3.1 else 1,
                    "activation_code_bits": 3,
                    "weight_code_bits": 2,
                    "physical_lut_inputs": 5,
                    "codebook_mode": args.c8_codebook,
                    "gradient_mode": args.c8_grad_mode,
                    "codebook": (
                        "sign-preserving unit octant centers, direct "
                        "[sign_r, sign_i, dominance] encoded"
                        if semantic_lut5_phase3p1
                        else (
                            "nearest C8 unit roots, Gray encoded"
                            if args.c8_codebook == "roots"
                            else (
                                "sign-preserving unit octant centers with "
                                "dominance refinement, Gray encoded"
                            )
                        )
                    ),
                    "activation_address": (
                        "direct_semantic" if semantic_lut5_phase3p1 else "gray"
                    ),
                    "diagonal_phase_indices": (
                        [0, 2, 4, 6]
                        if args.c8_codebook == "roots"
                        else None
                    ),
                    "axis_phase_indices": (
                        [1, 3, 5, 7]
                        if args.c8_codebook == "roots"
                        else None
                    ),
                    "preserves_component_signs": (
                        args.c8_codebook == "octants"
                    ),
                    "dominance_refinement": (
                        args.c8_codebook == "octants"
                    ),
                    "beta": float(args.c8_beta),
                    "beta_active": args.c8_grad_mode in (
                        "softmax",
                        "semantic_ste",
                        "semantic_phase_ste",
                    ),
                    "semantic_bit_gradients": (
                        args.c8_grad_mode in (
                            "semantic_ste",
                            "semantic_phase_ste",
                        )
                    ),
                    "phase_normalized_dominance": (
                        args.c8_grad_mode == "semantic_phase_ste"
                    ),
                    "fully_hard_c8_forward": True,
                    "local_compare_padding_mode": (
                        args.phase3p1_padding_mode
                        if local_compare
                        else None
                    ),
                    "local_compare_transition_epochs": (
                        args.phase3p1_transition_epochs
                        if local_compare
                        else 0
                    ),
                    "local_compare_transition_schedule": (
                        args.phase3p1_transition_schedule
                        if local_compare
                        else None
                    ),
                    "local_compare_progress": (
                        float(phase3p1_current_state["local_progress"])
                        if local_compare
                        else 0.0
                    ),
                    "fully_hard_local_compare": (
                        bool(
                            phase3p1_current_state[
                                "hard_local_compare_ready"
                            ]
                        )
                        if local_compare
                        else False
                    ),
                    "hard_forward_surrogate_backward": True,
                    "local_truncate_before_accumulate": (
                        local_compare
                        or (
                            learned_c8_lut
                            and not pure_mlp_phase3p1
                        )
                    ),
                    "compiled_lut_local_truncate_before_accumulate": bool(
                        pure_mlp_phase3p1
                    ),
                    "training_operator": (
                        (
                            "pure continuous 5-to-hidden-to-2 MLP local "
                            "outputs, accumulate"
                            if pure_mlp_phase3p1
                            else (
                                "residual neural-generated C8 LUT5 local "
                                "outputs, accumulate"
                                if neural_lut5_phase3p1
                                else (
                                    "differentiable direct-entry C8 LUT5 "
                                    "local outputs, accumulate"
                                )
                            )
                        )
                        if learned_c8_lut
                        else
                        (
                            "binary complex multiply, local >=0, "
                            "skip out-of-bounds, accumulate"
                            if args.phase3p1_padding_mode == "zero"
                            else (
                                "binary complex multiply, local >=0, "
                                "low-code pad, accumulate"
                            )
                        )
                        if local_compare
                        else "ordinary complex convolution"
                    ),
                    "uses_lut_backend_during_training": learned_c8_lut,
                    "trainable_lut_parameters": (
                        learned_c8_lut and not pure_mlp_phase3p1
                    ),
                    "lut_parameterization": (
                        "pure_mlp"
                        if pure_mlp_phase3p1
                        else (
                            "neural_residual"
                            if neural_lut5_phase3p1
                            else (
                                "direct_entries"
                                if learned_c8_lut
                                else None
                            )
                        )
                    ),
                    "truth_table_materialized_during_training": (
                        learned_c8_lut
                    ),
                    "persistent_trainable_truth_table": (
                        learned_c8_lut and not pure_mlp_phase3p1
                    ),
                    "mlp_enumeration_is_execution_only": (
                        pure_mlp_phase3p1
                    ),
                    "code_occupancy": (
                        None
                        if c8_last_occupancy is None
                        else dict(c8_last_occupancy)
                    ),
                }

            if phase3p6_qat_enabled:
                best_metrics["phase3p6_c8_qat"] = {
                    "source_checkpoint": initialization_checkpoint_path,
                    "source_phase": 3,
                    "source_lut_inputs": 4,
                    "activation_code_bits": 3,
                    "weight_code_bits": 2,
                    "physical_lut_inputs": 5,
                    "codebook": "nearest C8 unit-circle, Gray encoded",
                    "old_c4_phase_indices": [0, 2, 4, 6],
                    "added_axis_phase_indices": [1, 3, 5, 7],
                    "warmup_epochs": args.phase3p6_warmup_epochs,
                    "transition_epochs": args.phase3p6_transition_epochs,
                    "transition_schedule": args.phase3p6_schedule,
                    "raw_progress": float(
                        phase3p6_current_state["raw_progress"]
                    ),
                    "code_progress": float(
                        phase3p6_current_state["code_progress"]
                    ),
                    "beta": float(phase3p6_current_state["beta"]),
                    "fully_hard_c8": bool(
                        phase3p6_current_state["hard_c8_ready"]
                    ),
                    "hard_forward_soft_backward": True,
                    "local_truncate_before_accumulate": True,
                    "training_operator": "analytic C8 decode, binary complex multiply, local >=0, accumulate",
                    "uses_lut_backend_during_training": False,
                    "trainable_lut_parameters": False,
                    "truth_table_materialized_during_training": False,
                    "code_occupancy": (
                        None
                        if c8_last_occupancy is None
                        else dict(c8_last_occupancy)
                    ),
                }

            if lut_soft_projection_stats is not None:
                best_metrics["lut_soft_stage"] = dict(lut_soft_projection_stats)
                best_metrics["lut_soft_stage_from_unperturbed_origin"] = dict(
                    lut_soft_origin_stats
                )
                best_metrics["lut_initial_perturbation"] = dict(
                    lut_soft_initial_perturbation_stats
                )
            if lut_search_final_stats is not None:
                best_metrics["lut_search_final"] = dict(lut_search_final_stats)
                best_metrics["lut_search_microcommit_history"] = [
                    dict(stats) for stats in lut_search_microcommit_history
                ]
            if phase5_enabled:
                best_metrics["phase5_anchor_search"] = {
                    "source_checkpoint": initialization_checkpoint_path,
                    "source_phase": 4,
                    "source_lut_inputs": 4,
                    "initial_duplication": dict(
                        phase5_initial_duplication_stats
                    ),
                    "anchor_slice": args.phase5_anchor_slice,
                    "trainable_slice": 1 - args.phase5_anchor_slice,
                    "ties_only": args.phase5_ties_only,
                    "max_immediate_val_drop": (
                        args.phase5_max_immediate_val_drop
                    ),
                    "min_cycle_val_gain": args.phase5_min_cycle_val_gain,
                    "cycle_history": [
                        dict(record) for record in phase5_cycle_history
                    ],
                    "final_cycle_baseline": dict(
                        phase5_cycle_baseline_metrics
                    ),
                    "hard_binary_checkpoint": True,
                }
            checkpoint = {
                "phase": args.phase,
                "model": unwrapped_model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "epoch": epoch + 1,
                "args": vars(args),
                "metrics": best_metrics,
            }
            if neural_lut5_phase3p1 or pure_mlp_phase3p1:
                checkpoint["hard_lut_tables"] = capture_hard_lut_tables(
                    unwrapped_model
                )
            best_path = save_checkpoint(checkpoint, args.workdir, phase_checkpoint_filename(args.phase))
            metrics_path = update_phase_metrics(args.workdir, args.phase, best_metrics)
            train_logger.info("Saved best model to {} at epoch {:5d}".format(best_path, epoch + 1))
            train_logger.info("Updated phase metrics at {}".format(metrics_path))

        if ddp_enabled:
            dist.barrier()

    if phase_bit_occupancy_tracker is not None:
        phase_bit_occupancy_tracker.close()
    if c8_occupancy_tracker is not None:
        c8_occupancy_tracker.close()

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
    parser.add_argument("--comp_init", default="complex_independent", type=str,
                        choices=["complex_independent", "bimodal"])
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
    parser.add_argument("--phase", default=1, type=parse_phase, choices=sorted(PHASE_DESCRIPTIONS),
                        help="Training phase: 1, 2, 2.1, 3, 3.1, 3.5, 3.6, 4, or 5")
    parser.add_argument("--checkpoint", default=None, type=str,
                        help="Explicit checkpoint path to initialize phases greater than 1")
    parser.add_argument("--train-from-scratch", action="store_true",
                        help="Skip previous-phase checkpoint initialization and train the requested phase from random initialization")
    parser.add_argument("--lut-sets", default=1, type=int,
                        help="Number of independently learnable LUT pairs per ComplexLUTConv2d layer")
    parser.add_argument("--lut-allocation", default="layer", choices=["layer", "channel"],
                        help="Allocate LUT sets from a layer-shared pool or from per-output-channel pools")
    parser.add_argument("--lut-sets-per-channel", default=1, type=int,
                        help="Number of independently learnable LUT pairs owned by each output channel when --lut-allocation=channel")
    parser.add_argument("--lut-inputs", default=4, type=int, choices=[4, 5],
                        help="Physical LUT inputs: 4 uses two activation sign bits; 5 uses phase dominance in legacy phases or a three-bit C8 code in Phase2.1/3.1/3.6")
    parser.add_argument(
        "--lut-extra-bit",
        default="phase",
        choices=["phase", "magnitude_ste"],
        help=(
            "Third LUT5 activation bit. phase preserves the existing "
            "component-dominance path; magnitude_ste uses a learned "
            "magnitude threshold and backward-only asymmetric LUT shadow"
        ),
    )
    parser.add_argument(
        "--magnitude-threshold-init",
        default=1.0,
        type=float,
        help="Initial per-input-channel threshold for magnitude_ste",
    )
    parser.add_argument(
        "--magnitude-bit-beta",
        default=2.0,
        type=float,
        help="Sigmoid inverse temperature used only by the magnitude-bit backward",
    )
    parser.add_argument(
        "--magnitude-shadow-epsilon",
        default=0.05,
        type=float,
        help=(
            "Backward-only LUT probability offset that breaks duplicated "
            "magnitude-slice symmetry without changing hard forward outputs"
        ),
    )
    parser.add_argument(
        "--magnitude-threshold-lr",
        default=1e-3,
        type=float,
        help="Learning rate for per-channel magnitude thresholds",
    )
    parser.add_argument(
        "--magnitude-threshold-schedule",
        default="follow_base",
        choices=["follow_base", "constant"],
        help="Schedule for the magnitude-threshold optimizer group",
    )
    parser.add_argument(
        "--phase2p1-mode",
        default="c8",
        choices=["c8", "learned_lut5"],
        help=(
            "Phase2.1 operator: c8 keeps the existing decoded C8 complex "
            "multiply; learned_lut5 directly trains a differentiable "
            "[sign_r, sign_i, dominance, weight_r, weight_i] -> "
            "[local_real_bit, local_imag_bit] LUT5"
        ),
    )
    parser.add_argument(
        "--phase3p1-mode",
        default="fixed",
        choices=[
            "fixed", "learned_lut5", "semantic_lut5",
            "neural_lut5", "pure_mlp",
        ],
        help=(
            "Phase3.1 operator: fixed keeps the analytic local comparator; "
            "learned_lut5 initializes a differentiable 5-to-2 LUT from the "
            "pretrained Phase2.1 C8 product and trains independent table "
            "entries; semantic_lut5 uses the direct [sign_r, sign_i, "
            "dominance] address and semantic-STE gradients; neural_lut5 "
            "uses a residual 5-to-hidden-to-2 MLP to "
            "generate the same deployable truth table; pure_mlp removes "
            "trainable LUT entries and tau annealing, trains a continuous "
            "5-to-hidden-to-2 MLP operation, then enumerates it into LUT5"
        ),
    )
    parser.add_argument(
        "--neural-lut-hidden",
        default=8,
        type=int,
        help=(
            "Hidden width of the Phase3.1 residual neural LUT5 generator "
            "or pure MLP operation"
        ),
    )
    parser.add_argument(
        "--neural-lut-residual-scale",
        default=1.0,
        type=float,
        help="Scale applied to the neural LUT5 residual logits",
    )
    parser.add_argument(
        "--neural-lut-hard-eval-interval",
        default=10,
        type=int,
        help=(
            "Evaluate a temporary hard projection every N epochs during "
            "neural LUT5 or pure MLP soft training; 0 disables the extra "
            "evaluation"
        ),
    )
    parser.add_argument(
        "--c8-beta",
        default=2.0,
        type=float,
        help=(
            "Inverse temperature for soft-codebook or semantic-dominance "
            "C8 backward paths"
        ),
    )
    parser.add_argument(
        "--c8-codebook",
        default="roots",
        choices=["roots", "octants"],
        help=(
            "C8 decode geometry: roots uses 0/45-degree unit roots; "
            "octants strictly refines component signs with a dominance bit"
        ),
    )
    parser.add_argument(
        "--c8-grad-mode",
        default="softmax",
        choices=[
            "softmax",
            "bireal_ste",
            "semantic_ste",
            "semantic_phase_ste",
        ],
        help=(
            "C8 activation backward surrogate; semantic_ste gives direct "
            "sign-real, sign-imag, and dominance gradients for octants; "
            "semantic_phase_ste additionally normalizes only the dominance "
            "score by complex magnitude; "
            "the forward remains hard for every mode"
        ),
    )
    parser.add_argument(
        "--phase3p1-padding-mode",
        default="low_code",
        choices=["low_code", "zero"],
        help=(
            "Phase3.1 boundary semantics: low_code preserves the legacy "
            "physical-low C8 code; zero skips out-of-bounds local products"
        ),
    )
    parser.add_argument(
        "--phase3p1-transition-epochs",
        default=0,
        type=int,
        help=(
            "Epochs used to move continuously from the exact Phase2.1 "
            "complex convolution (rho=0) to the hard local comparator "
            "(rho=1); 0 preserves the immediate-hard legacy behavior"
        ),
    )
    parser.add_argument(
        "--phase3p1-transition-schedule",
        default="cosine",
        choices=["linear", "cosine"],
        help="Progress schedule for Phase3.1 local-comparator strength",
    )
    parser.add_argument(
        "--phase3-lut5-transition-epochs",
        default=None,
        type=int,
        help=(
            "Epochs used to linearly mix a LUT4 Phase3 checkpoint into the "
            "hard phase-conditioned LUT5 operation; defaults to half the run"
        ),
    )
    parser.add_argument(
        "--phase3p5-warmup-epochs",
        default=5,
        type=int,
        help=(
            "Phase3.5 epochs held at the exact Phase2 BNN endpoint before "
            "activation-QAT transition"
        ),
    )
    parser.add_argument(
        "--phase3p5-transition-epochs",
        default=None,
        type=int,
        help=(
            "Phase3.5 epochs used for cosine activation reconstruction from "
            "Phase2 magnitudes (1,1) to deployable LUT5 magnitudes (2,0); "
            "defaults to half the run"
        ),
    )
    parser.add_argument(
        "--phase3p5-beta-start",
        default=1.0,
        type=float,
        help="Initial sigmoid inverse temperature for the Phase3.5 comparator backward",
    )
    parser.add_argument(
        "--phase3p5-beta-end",
        default=8.0,
        type=float,
        help="Final sigmoid inverse temperature for the Phase3.5 comparator backward",
    )
    parser.add_argument(
        "--phase3p6-warmup-epochs",
        default=10,
        type=int,
        help="Epochs held at the exact four-code Phase3 activation endpoint",
    )
    parser.add_argument(
        "--phase3p6-transition-epochs",
        default=None,
        type=int,
        help="Epochs used to progressively enable the four additional C8 axis codes",
    )
    parser.add_argument(
        "--phase3p6-beta-start",
        default=2.0,
        type=float,
        help="Initial soft-codebook inverse temperature for the C8 STE backward",
    )
    parser.add_argument(
        "--phase3p6-beta-end",
        default=12.0,
        type=float,
        help="Final soft-codebook inverse temperature for the C8 STE backward",
    )
    parser.add_argument(
        "--phase3p6-schedule",
        default="cosine",
        choices=["linear", "cosine"],
        help="Progress schedule for enabling the four additional C8 codes",
    )
    parser.add_argument("--lut-logit-init", default=5.0, type=float,
                        help="Initial absolute LUT logit value; smaller values avoid tanh saturation")
    parser.add_argument("--lut-init-mode", default="binary", choices=["binary", "raw", "random"],
                        help="LUT initialization: binary keeps the current hard truth table; raw keeps zero-valued entries at logit 0; random samples each entry sign independently")
    parser.add_argument("--lut-sign-flip-prob", default=0.0, type=float,
                        help="Probability of flipping each initialized LUT logit's sign before Phase 4 training")
    parser.add_argument("--lut-lr", default=None, type=float,
                        help="Optional learning rate for lut_r/lut_i parameters")
    parser.add_argument("--lut-schedule", default="follow_base", choices=["follow_base", "constant"],
                        help="Make LUT LR follow the base schedule or remain fixed at --lut-lr")
    parser.add_argument("--lut-tau-min", default=1.0, type=float,
                        help="Initial tau for LUT annealing")
    parser.add_argument("--lut-tau-max", default=10.0, type=float,
                        help="Maximum tau for LUT annealing before hard mode")
    parser.add_argument("--lut-anneal-epochs", default=None, type=int,
                        help="Optional explicit number of Phase 4 epochs spent in soft LUT annealing before hard mode")
    parser.add_argument(
        "--lut-soft-warmup-epochs",
        default=0,
        type=int,
        help=(
            "Initial epochs held at lut_tau_min; these epochs are included "
            "inside --lut-anneal-epochs"
        ),
    )
    parser.add_argument("--lut-hard-ste", action="store_true",
                        help="Use hard LUT forward with STE gradients for all Phase 4 epochs")
    parser.add_argument("--lut-hard-epoch-fraction", default=0.9, type=float,
                        help="Fraction of Phase 4 epochs before switching to hard LUT mode when --lut-anneal-epochs is not set")
    parser.add_argument("--lut-hard-transition-epochs", default=0, type=int,
                        help="Number of epochs to linearly mix soft LUT outputs into hard LUT outputs after annealing")
    parser.add_argument("--lut-soft-only-epochs", default=0, type=int,
                        help="Train only soft LUT parameters for N epochs, save diagnostics, then project once and train only hard-LUT weights")
    parser.add_argument("--lut-hard-bn-only-epochs", default=0, type=int,
                        help="Use hard LUT forward with STE and train only LUT plus BatchNorm for N epochs, then freeze LUT and train weights plus BatchNorm")
    parser.add_argument("--lut-search-epochs", default=0, type=int,
                        help="Total LUT/weight alternating epochs before final hard-LUT weight adaptation")
    parser.add_argument("--lut-search-weight-epochs-per-lut", default=1, type=int,
                        help="Number of weight recovery epochs after every LUT score/micro-commit epoch")
    parser.add_argument("--lut-search-max-flips-per-commit", default=8, type=int,
                        help="Global maximum number of hard entry flips after each LUT epoch")
    parser.add_argument("--lut-search-max-flips-per-layer", default=1, type=int,
                        help="Maximum entries contributed by one LUT layer to each micro-commit")
    parser.add_argument("--lut-search-flip-cooldown", default=2, type=int,
                        help="Number of later LUT commit opportunities that cannot immediately flip an entry back")
    parser.add_argument("--lut-search-init-flip-prob", default=0.05, type=float,
                        help="Initial probability of flipping each entry during staged LUT search")
    parser.add_argument("--lut-search-score-temperature", default=0.25, type=float,
                        help="Temperature used only to parameterize flip scores; lower values make probabilities move faster")
    parser.add_argument("--lut-search-commit-threshold", default=0.8, type=float,
                        help="Commit an entry flip only when its learned probability reaches this threshold")
    parser.add_argument("--lut-search-sparsity", default=0.0, type=float,
                        help="Optional mean flip-probability penalty during LUT-only search")
    parser.add_argument("--lut-search-freeze-bn", action="store_true",
                        help="Keep BatchNorm running statistics fixed during LUT-only search")
    parser.add_argument(
        "--phase5-anchor-slice",
        default=0,
        type=int,
        choices=[0, 1],
        help="Frozen LUT5 phase-bit slice; only the opposite copied slice may change",
    )
    parser.add_argument(
        "--phase5-ties-only",
        action="store_true",
        help="Restrict Phase 5 candidates to local complex-product tie addresses",
    )
    parser.add_argument(
        "--phase5-max-immediate-val-drop",
        default=0.01,
        type=float,
        help="Maximum validation-accuracy drop allowed immediately after a hard candidate commit",
    )
    parser.add_argument(
        "--phase5-min-cycle-val-gain",
        default=0.0,
        type=float,
        help="Minimum validation-accuracy gain required after weight recovery",
    )

    opt = parser.add_argument_group("Optimizers")
    opt.add_argument("--optimizer", "--opt", default="sgd", type=str, choices=["sgd", "nag", "adam", "adamw", "rmsprop"])
    opt.add_argument("--clipnorm", "--cn", default=1.0, type=float)
    opt.add_argument("--clipval", "--cv", default=1.0, type=float)
    opt.add_argument("--l1", default=0.0, type=float)
    opt.add_argument("--l2", default=0.0, type=float)
    opt.add_argument("--lr", default=1e-3, type=float)
    opt.add_argument("--momentum", "--mom", default=0.9, type=float)
    opt.add_argument("--decay", default=0.0, type=float)
    opt.add_argument("--schedule", default="default", type=str,
                     choices=["default", "constant", "cosine", "bireal"])
    opt.add_argument("--min-lr-factor", default=0.1, type=float,
                     help="Minimum LR as a fraction of --lr for cosine schedule")

    opt = parser.add_argument_group("Adam")
    opt.add_argument("--beta1", default=0.9, type=float)
    opt.add_argument("--beta2", default=0.999, type=float)

    return parser.parse_args(argv)


def main(argv):
    args = parse_args(argv[1:])
    train(args)


if __name__ == "__main__":
    main(sys.argv) 

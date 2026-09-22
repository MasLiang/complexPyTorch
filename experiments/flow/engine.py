"""Dataset-agnostic staged training engine."""

import json
import logging
import math
import random
import sys
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from complexPyTorch.complexLayers import PairLUT4ComplexConv2d
from experiments.flow.checkpoints import (
    compile_lut4_from_bireal,
    expand_lut6_from_lut4,
    load_matching_stage,
)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _confusion(targets, predictions, classes):
    flat = targets.to(torch.long) * classes + predictions.to(torch.long)
    return torch.bincount(flat, minlength=classes * classes).reshape(classes, classes)


def _metrics(matrix, loss_sum, count):
    support = matrix.sum(dim=1)
    predicted = matrix.sum(dim=0)
    valid = support > 0
    per_class = torch.zeros(matrix.size(0), dtype=torch.float64)
    precision = torch.zeros(matrix.size(0), dtype=torch.float64)
    f1 = torch.zeros(matrix.size(0), dtype=torch.float64)
    per_class[valid] = matrix.diag()[valid].double() / support[valid].double()
    predicted_valid = predicted > 0
    precision[predicted_valid] = matrix.diag()[predicted_valid].double() / predicted[predicted_valid].double()
    denominator = precision + per_class
    f1[denominator > 0] = 2.0 * precision[denominator > 0] * per_class[denominator > 0] / denominator[denominator > 0]
    return {
        "loss": loss_sum / max(count, 1),
        "oa": matrix.diag().sum().item() / max(count, 1),
        "aa": per_class[valid].mean().item() if valid.any() else 0.0,
        "macro_f1": f1[valid].mean().item() if valid.any() else 0.0,
        "per_class_accuracy": per_class.tolist(),
        "per_class_precision": precision.tolist(),
        "per_class_f1": f1.tolist(),
        "support": support.tolist(),
        "confusion_matrix": matrix.tolist(),
    }


def run_epoch(model, loader, criterion, device, classes, optimizer=None, max_batches=None):
    training = optimizer is not None
    model.train(training)
    loss_sum = 0.0
    count = 0
    matrix = torch.zeros(classes, classes, dtype=torch.long)
    context = torch.enable_grad() if training else torch.no_grad()
    with context:
        for batch_index, (inputs, targets) in enumerate(loader):
            inputs = inputs.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            if training:
                optimizer.zero_grad(set_to_none=True)
            logits = model(inputs)
            loss = criterion(logits, targets)
            if not torch.isfinite(loss):
                raise FloatingPointError("Non-finite loss encountered")
            if training:
                loss.backward()
                optimizer.step()
            predictions = logits.argmax(dim=1)
            matrix += _confusion(targets.detach().cpu(), predictions.detach().cpu(), classes)
            loss_sum += loss.item() * targets.size(0)
            count += targets.size(0)
            if max_batches is not None and batch_index + 1 >= max_batches:
                break
    return _metrics(matrix, loss_sum, count)


def learning_rate(epoch, stage):
    if stage.schedule == "constant":
        return stage.lr
    if stage.schedule == "cosine":
        progress = epoch / float(max(stage.epochs - 1, 1))
        minimum = stage.lr * stage.min_lr_factor
        return minimum + 0.5 * (stage.lr - minimum) * (1.0 + math.cos(math.pi * progress))
    if stage.schedule == "linear":
        return stage.lr * max(1.0 - epoch / float(max(stage.epochs, 1)), 0.0)
    if stage.schedule == "bireal":
        warmup = min(5, stage.epochs)
        if epoch < warmup:
            return stage.lr * (epoch + 1) / float(warmup)
        if epoch < int(stage.epochs * 0.5):
            return stage.lr
        if epoch < int(stage.epochs * 0.75):
            return stage.lr * 0.1
        if epoch < int(stage.epochs * 0.875):
            return stage.lr * 0.01
        return stage.lr * 0.001
    if stage.schedule == "multistep":
        return stage.lr * (0.1 ** sum(epoch >= point for point in (90, 140, 180, 220)))
    raise ValueError("Unknown schedule: {}".format(stage.schedule))


def _residual_alpha(epoch, stage):
    if stage.residual_ramp_epochs <= 1:
        return stage.residual_alpha_end
    progress = min(epoch / float(stage.residual_ramp_epochs - 1), 1.0)
    return stage.residual_alpha_start + progress * (stage.residual_alpha_end - stage.residual_alpha_start)


def configure_lut(model, epoch, stage):
    modules = [
        module for module in model.modules()
        if isinstance(module, PairLUT4ComplexConv2d)
        and module.parameterization == "categorical_residual"
    ]
    if not modules:
        return None
    alpha = _residual_alpha(epoch, stage)
    for module in modules:
        module.set_dominance_residual_alpha(alpha)
        module.set_dominance_base_alpha(0.0)
    return alpha


def build_optimizer(model, stage):
    lut_parameters = []
    frozen_base = []
    for module in model.modules():
        if not isinstance(module, PairLUT4ComplexConv2d):
            continue
        if stage.lut_lr is not None:
            lut_parameters.append(module.weight)
        if module.parameterization == "categorical_residual":
            frozen_base.append(module.dominance_base_correction)
    special = {id(parameter) for parameter in lut_parameters + frozen_base}
    network = [
        parameter for parameter in model.parameters()
        if parameter.requires_grad and id(parameter) not in special
    ]
    groups = [{"params": network, "lr_scale": 1.0, "weight_decay": stage.weight_decay}]
    if lut_parameters:
        groups.append({"params": lut_parameters, "lr_scale": stage.lut_lr / stage.lr, "weight_decay": 0.0})
    if stage.optimizer == "adam":
        return torch.optim.Adam(groups, lr=stage.lr)
    if stage.optimizer == "sgd":
        return torch.optim.SGD(groups, lr=stage.lr, momentum=0.9)
    raise ValueError("Unsupported optimizer: {}".format(stage.optimizer))


def _initialize(model, stage, previous_checkpoint, device):
    if stage.initialization == "scratch":
        return
    if previous_checkpoint is None:
        raise ValueError("{} requires a parent checkpoint".format(stage.name))
    if stage.initialization == "matching":
        load_matching_stage(model, previous_checkpoint, stage.parent, device=device)
    elif stage.initialization == "compile_lut4":
        compile_lut4_from_bireal(model, previous_checkpoint, device=device)
    elif stage.initialization == "expand_lut6":
        expand_lut6_from_lut4(model, previous_checkpoint, device=device)
    else:
        raise ValueError("Unknown initialization: {}".format(stage.initialization))


def _save_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")



def _write_metrics(stage_dir, metrics):
    """Persist paper-facing summaries without duplicating evaluation work."""
    _save_json(stage_dir / "test_metrics.json", metrics)
    _save_json(stage_dir / "metrics.json", metrics)
    _save_json(stage_dir / "confusion_matrix.json", metrics["confusion_matrix"])
    _save_json(stage_dir / "per_class_metrics.json", {
        "support": metrics["support"],
        "accuracy": metrics["per_class_accuracy"],
        "precision": metrics["per_class_precision"],
        "f1": metrics["per_class_f1"],
    })

def _resolve_stage_order(stage_specs, selected_stages, skip_parent_stages=False):
    """Return requested stages, including parents unless explicitly reused."""
    ordered = list(stage_specs)
    requested = ordered if not selected_stages else list(selected_stages)
    unknown = [stage for stage in requested if stage not in stage_specs]
    if unknown:
        raise ValueError("Unknown stages: {}".format(unknown))
    if skip_parent_stages:
        return [name for name in ordered if name in requested]
    required = set()

    def include(name):
        if name in required:
            return
        parent = stage_specs[name].parent
        if parent is not None:
            include(parent)
        required.add(name)

    for name in requested:
        include(name)
    return [name for name in ordered if name in required]


def run_flow(protocol, args, selected_stages=None):
    set_seed(args.seed)
    device = torch.device(args.device if args.device != "cuda" or torch.cuda.is_available() else "cpu")
    bundle = protocol.build_bundle(args)
    stage_specs = {stage.name: stage for stage in protocol.stages(args)}
    if args.skip_parent_stages and not args.parent_workdir:
        raise ValueError("--skip-parent-stages requires --parent-workdir")
    requested = _resolve_stage_order(
        stage_specs,
        selected_stages,
        skip_parent_stages=args.skip_parent_stages,
    )

    root = Path(args.workdir)
    root.mkdir(parents=True, exist_ok=True)
    _save_json(root / "protocol.json", {"dataset": protocol.name, "num_classes": bundle.num_classes, **bundle.metadata, "args": vars(args)})
    (root / "command.txt").write_text(" ".join(sys.argv) + "\n", encoding="utf-8")
    logger = logging.getLogger("flow.{}".format(protocol.name))
    results = {}
    for name in requested:
        stage = stage_specs[name]
        stage_dir = root / stage.name
        stage_dir.mkdir(parents=True, exist_ok=True)
        _save_json(stage_dir / "config.json", {"dataset": protocol.name, "stage": asdict(stage), "args": vars(args), "metadata": bundle.metadata})
        log_handler = logging.FileHandler(stage_dir / "training.log", encoding="utf-8")
        log_handler.setFormatter(logging.Formatter("[%(asctime)s %(levelname)s] %(message)s"))
        logger.addHandler(log_handler)
        parent_checkpoint = None
        if stage.parent:
            parent_root = (
                Path(args.parent_workdir)
                if args.parent_workdir is not None
                else root
            )
            parent_checkpoint = parent_root / stage.parent / "best.pt"
        model = protocol.build_model(stage, args, bundle.num_classes).to(device)
        _initialize(model, stage, parent_checkpoint, device)
        alpha = configure_lut(model, 0, stage)
        criterion = nn.CrossEntropyLoss(label_smoothing=stage.label_smoothing)
        optimizer = build_optimizer(model, stage)
        loader_kwargs = {"batch_size": args.batch_size, "num_workers": args.num_workers, "pin_memory": device.type == "cuda"}
        train_loader = DataLoader(bundle.train, shuffle=True, **loader_kwargs)
        validation_loader = None if bundle.validation is None else DataLoader(bundle.validation, shuffle=False, **loader_kwargs)
        test_loader = DataLoader(bundle.test, shuffle=False, **loader_kwargs)
        if args.smoke_test:
            metrics = run_epoch(model, train_loader, criterion, device, bundle.num_classes, optimizer, max_batches=1)
            payload = {"stage": stage.name, "model": model.state_dict(), "optimizer": optimizer.state_dict(), "epoch": 1, "selection": metrics, "args": vars(args)}
            torch.save(payload, stage_dir / "best.pt")
            torch.save(payload, stage_dir / "best_aa.pt")
            torch.save(payload, stage_dir / "last.pt")
            _write_metrics(stage_dir, metrics)
            results[stage.name] = metrics
            logger.info("Smoke stage=%s train_OA=%.4f", stage.name, metrics["oa"])
            logger.removeHandler(log_handler)
            log_handler.close()
            continue
        best_oa = -math.inf
        best_aa = -math.inf
        history = []
        for epoch in range(stage.epochs):
            lr = learning_rate(epoch, stage)
            for group in optimizer.param_groups:
                group["lr"] = lr * group.get("lr_scale", 1.0)
            alpha = configure_lut(model, epoch, stage)
            train_metrics = run_epoch(model, train_loader, criterion, device, bundle.num_classes, optimizer)
            validation_metrics = None if validation_loader is None else run_epoch(model, validation_loader, criterion, device, bundle.num_classes)
            selection_metrics = validation_metrics or run_epoch(model, test_loader, criterion, device, bundle.num_classes)
            selection_eligible = (
                args.select_any_residual_alpha
                or alpha is None
                or abs(alpha - stage.residual_alpha_end) <= 1e-8
            )
            record = {
                "epoch": epoch + 1,
                "lr": lr,
                "residual_alpha": alpha,
                "selection_eligible": selection_eligible,
                "train": train_metrics,
                "validation": validation_metrics,
                "selection": selection_metrics,
            }
            history.append(record)
            logger.info("dataset=%s stage=%s epoch=%03d train_OA=%.4f selection_OA=%.4f selection_AA=%.4f", protocol.name, stage.name, epoch + 1, train_metrics["oa"], selection_metrics["oa"], selection_metrics["aa"])
            payload = {
                "stage": stage.name,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "epoch": epoch + 1,
                "residual_alpha": alpha,
                "selection_eligible": selection_eligible,
                "selection": selection_metrics,
                "args": vars(args),
            }
            torch.save(payload, stage_dir / "last.pt")
            if selection_eligible and selection_metrics["oa"] > best_oa:
                best_oa = selection_metrics["oa"]
                torch.save(payload, stage_dir / "best.pt")
            if selection_eligible and selection_metrics["aa"] > best_aa:
                best_aa = selection_metrics["aa"]
                torch.save(payload, stage_dir / "best_aa.pt")
            _save_json(stage_dir / "history.json", history)
        model.load_state_dict(torch.load(stage_dir / "best.pt", map_location=device)["model"])
        test_metrics = run_epoch(model, test_loader, criterion, device, bundle.num_classes)
        _write_metrics(stage_dir, test_metrics)
        results[stage.name] = test_metrics
        logger.info("finished dataset=%s stage=%s test_OA=%.4f test_AA=%.4f test_macro_f1=%.4f", protocol.name, stage.name, test_metrics["oa"], test_metrics["aa"], test_metrics["macro_f1"])
        logger.removeHandler(log_handler)
        log_handler.close()
    _save_json(root / "flow_results.json", results)
    return results

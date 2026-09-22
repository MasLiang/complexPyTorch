"""Shared training utilities for San Francisco classification stages."""

import random

import numpy as np
import torch


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def confusion_matrix(targets, predictions, num_classes):
    matrix = torch.zeros(num_classes, num_classes, dtype=torch.long)
    flat = targets.to(torch.long) * num_classes + predictions.to(torch.long)
    matrix += torch.bincount(flat, minlength=num_classes * num_classes).reshape(
        num_classes, num_classes
    )
    return matrix


def metrics_from_confusion(matrix):
    total = matrix.sum().item()
    correct = matrix.diag().sum().item()
    oa = correct / total if total else 0.0
    support = matrix.sum(dim=1)
    valid = support > 0
    per_class = torch.zeros(matrix.size(0), dtype=torch.float64)
    per_class[valid] = matrix.diag()[valid].double() / support[valid].double()
    aa = per_class[valid].mean().item() if valid.any() else 0.0
    return oa, aa, per_class.tolist()


def run_epoch(model, loader, criterion, device, optimizer=None, max_batches=None):
    training = optimizer is not None
    model.train(training)
    loss_sum = 0.0
    count = 0
    matrix = None
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
            batch_matrix = confusion_matrix(
                targets.detach().cpu(), predictions.detach().cpu(), logits.size(1)
            )
            matrix = batch_matrix if matrix is None else matrix + batch_matrix
            loss_sum += loss.item() * targets.size(0)
            count += targets.size(0)
            if max_batches is not None and batch_index + 1 >= max_batches:
                break
    oa, aa, per_class = metrics_from_confusion(matrix)
    return {
        "loss": loss_sum / max(count, 1),
        "oa": oa,
        "aa": aa,
        "per_class_accuracy": per_class,
        "confusion_matrix": matrix.tolist(),
    }

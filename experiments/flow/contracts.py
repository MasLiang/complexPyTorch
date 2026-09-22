"""Contracts shared by all dataset-specific training protocols."""

from dataclasses import dataclass
from typing import Callable, Optional


@dataclass(frozen=True)
class DatasetBundle:
    train: object
    validation: Optional[object]
    test: object
    num_classes: int
    metadata: dict


@dataclass(frozen=True)
class StageSpec:
    name: str
    model_key: str
    epochs: int
    lr: float
    optimizer: str
    schedule: str
    parent: Optional[str] = None
    initialization: str = "scratch"
    weight_decay: float = 0.0
    label_smoothing: float = 0.0
    min_lr_factor: float = 0.01
    lut_lr: Optional[float] = None
    residual_alpha_start: float = 0.0
    residual_alpha_end: float = 1.0
    residual_ramp_epochs: int = 1


class DatasetProtocol:
    """Dataset-owned construction hooks consumed by the shared stage engine."""

    name: str

    def add_arguments(self, parser):
        raise NotImplementedError

    def build_bundle(self, args) -> DatasetBundle:
        raise NotImplementedError

    def build_model(self, stage: StageSpec, args, num_classes):
        raise NotImplementedError

    def stages(self, args):
        raise NotImplementedError

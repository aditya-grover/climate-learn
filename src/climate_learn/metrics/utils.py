# Standard library
from dataclasses import dataclass
from typing import List

# Local application
import numpy.typing as npt
import torch
import torch.nn as nn


@dataclass
class MetricsMetaInfo:
    in_vars: List[str]
    out_vars: List[str]
    lat: npt.ArrayLike
    lon: npt.ArrayLike
    climatology: torch.Tensor
    denormalization: nn.Module


METRICS_REGISTRY = {}
def register(name):
    def decorator(metric_class):
        metric_class.name = name
        return metric_class
    return decorator
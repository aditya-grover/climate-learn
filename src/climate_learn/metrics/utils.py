# Standard library
from dataclasses import dataclass
from typing import List

# Local application
import numpy.typing as npt
import torch


@dataclass
class MetricsMetaInfo:
    in_vars: List[str]
    out_vars: List[str]
    lat: npt.ArrayLike
    lon: npt.ArrayLike
    climatology: torch.Tensor


METRICS_REGISTRY = {}
def register(name):
    def decorator(metric_class):
        METRICS_REGISTRY[name] = metric_class
        metric_class.name = name
        return metric_class
    return decorator
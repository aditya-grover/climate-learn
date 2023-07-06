# Standard library
from dataclasses import dataclass
from functools import wraps
from typing import List, Union

# Third party
import numpy.typing as npt
import torch

Pred = Union[torch.FloatTensor, torch.DoubleTensor, torch.distributions.Normal]


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


def handles_probabilistic(metric):
    @wraps(metric)
    def wrapper(pred: Pred, *args, **kwargs):
        if isinstance(pred, torch.distributions.Normal):
            pred = pred.loc
        return metric(pred, *args, **kwargs)

    return wrapper

# Standard library
from typing import Callable, Optional, Union

# Local application
from .metrics import *

# Third party
import torch
import torch.nn as nn


def mse(
    pred: Union[torch.FloatTensor, torch.DoubleTensor],
    target: Union[torch.FloatTensor, torch.DoubleTensor],
    aggregate_only: bool = False,
    lat_weighted: bool = False,
    metainfo: Optional[MetricsMetaInfo] = None
) -> Union[torch.FloatTensor, torch.DoubleTensor]:    
    if lat_weighted:
        metric = LatWeightedMSE(aggregate_only, metainfo=metainfo)
    else:
        metric = MSE(aggregate_only, metainfo=metainfo)
    return metric(pred, target)

def rmse(
    pred: Union[torch.FloatTensor, torch.DoubleTensor],
    target: Union[torch.FloatTensor, torch.DoubleTensor],
    aggregate_only: bool = False,
    lat_weighted: bool = False,
    metainfo: Optional[MetricsMetaInfo] = None
) -> Union[torch.FloatTensor, torch.DoubleTensor]:
    if lat_weighted:
        metric = LatWeightedRMSE(aggregate_only, metainfo=metainfo)
    else:
        metric = RMSE(aggregate_only, metainfo=metainfo)
    return metric(pred, target)

def acc(
    pred: Union[torch.FloatTensor, torch.DoubleTensor],
    target: Union[torch.FloatTensor, torch.DoubleTensor],
    aggregate_only: bool = False,
    lat_weighted: bool = False,
    metainfo: Optional[MetricsMetaInfo] = None
) -> Union[torch.FloatTensor, torch.DoubleTensor]:
    if lat_weighted:
        metric = LatWeightedACC(aggregate_only, metainfo=metainfo)
    else:
        metric = ACC(aggregate_only, metainfo=metainfo)
    return metric(pred, target)

def pearson(
    pred: Union[torch.FloatTensor, torch.DoubleTensor],
    target: Union[torch.FloatTensor, torch.DoubleTensor]
) -> Union[torch.FloatTensor, torch.DoubleTensor]:
    metric = Pearson()
    return metric(pred, target)

def mean_bias(
    pred: Union[torch.FloatTensor, torch.DoubleTensor],
    target: Union[torch.FloatTensor, torch.DoubleTensor]
) -> Union[torch.FloatTensor, torch.DoubleTensor]:
    metric = MeanBias()
    return metric(pred, target)

def denormalized(
    denorm: nn.Module,
    metric: Callable[[torch.Tensor], torch.Tensor],
    pred: Union[torch.FloatTensor, torch.DoubleTensor],
    target: Union[torch.FloatTensor, torch.DoubleTensor]
) -> Union[torch.FloatTensor, torch.DoubleTensor]:
    metric = Denormalized(denorm, metric)
    return metric(pred, target)
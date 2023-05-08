# Standard library
from typing import Optional, Union

# Local application
from .metrics import *

# Third party
import torch


def mse(
    pred: Union[torch.FloatTensor, torch.DoubleTensor],
    target: Union[torch.FloatTensor, torch.DoubleTensor],
    lat_weighted: bool = False,
    metainfo: Optional[MetricsMetaInfo] = None
) -> Union[torch.FloatTensor, torch.DoubleTensor]:
    metric = MSE(lat_weighted, metainfo=metainfo)
    return metric(pred, target)

def rmse(
    pred: Union[torch.FloatTensor, torch.DoubleTensor],
    target: Union[torch.FloatTensor, torch.DoubleTensor],
    lat_weighted: bool = False,
    metainfo: Optional[MetricsMetaInfo] = None
) -> Union[torch.FloatTensor, torch.DoubleTensor]:
    metric = RMSE(lat_weighted, metainfo=metainfo)
    return metric(pred, target)

def acc(
    pred: Union[torch.FloatTensor, torch.DoubleTensor],
    target: Union[torch.FloatTensor, torch.DoubleTensor],
    lat_weighted: bool = False,
    metainfo: Optional[MetricsMetaInfo] = None
) -> Union[torch.FloatTensor, torch.DoubleTensor]:
    metric = ACC(lat_weighted, metainfo=metainfo)
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
from dataclasses import dataclass
from typing import List, Optional, Union
import numpy.typing as npt
import torch
import torch.nn as nn
import numpy as np


@dataclass
class MetricsMetaInfo:
    in_vars: List[str]
    out_vars: List[str]
    lat: npt.ArrayLike
    lon: npt.ArrayLike
    climatology: torch.Tensor


class Metric(nn.Module):
    """Parent class for all ClimateLearn metrics."""
    def __init__(self,
                 aggregate_only: bool = False,
                 metainfo: Optional[MetricsMetaInfo] = None
                ):
        r"""
        .. highlight:: python

        :param aggregate_only: If false, returns both the aggregate and
            per-channel metrics. Otherwise, returns only the aggregate metric.
            Default if `False`.
        :type aggregate_only: bool
        :param metainfo: Optional meta-information used by some metrics.
        :type metainfo: MetricsMetaInfo|None
        """
        super().__init__()
        self.aggregate_only = aggregate_only
        self.metainfo = metainfo

    def forward(self,
                pred: torch.Tensor,
                target: torch.Tensor
                ) -> torch.Tensor:
        """
        :param pred: The predicted value(s).
        :type pred: torch.Tensor
        :param target: The ground truth target value(s).
        :type target: torch.Tensor        

        :return: A tensor. See child classes for specifics.
        :rtype: torch.Tensor
        """
        raise NotImplementedError()
    

class MeanSquaredError(Metric):
    """Computes standard mean-squared error."""
    def forward(self,
                pred: Union[torch.FloatTensor, torch.DoubleTensor],
                target: Union[torch.FloatTensor, torch.DoubleTensor]
                ) -> Union[torch.FloatTensor, torch.DoubleTensor]:
        r"""
        .. highlight:: python

        :param pred: The predicted values of shape [B,C,H,W].
        :type pred: torch.FloatTensor|torch.DoubleTensor
        :param target: The ground truth target values of shape [B,C,H,W].
        :type target: torch.FloatTensor|torch.DoubleTensor

        :return: A singleton tensor if `self.aggregate_only` is `True`. Else, a
            tensor of shape [C+1], where the last element is the aggregate
            MSE, and the preceding elements are the channel-wise MSEs.
        :rtype: torch.FloatTensor|torch.DoubleTensor
        """
        error = (pred - target).square()
        loss = error.mean()
        if not self.aggregate_only:
            per_channel_losses = error.mean([0,2,3])
            loss = loss.unsqueeze(0)
            loss = torch.cat((per_channel_losses, loss))
        return loss
    

class MSE(MeanSquaredError):
    r"""
    .. highlight:: python

    An alias for `MeanSquaredError`.
    """
    pass


class LatitudeWeightedMetric(Metric):
    """Parent class for all latitude-weighted metrics."""
    def __init__(self,
                aggregate_only: bool = False,
                metainfo: Optional[MetricsMetaInfo] = None
            ):
        super().__init__(aggregate_only, metainfo)
        lat_weights = np.cos(np.deg2rad(self.metainfo.lat))
        lat_weights = lat_weights / lat_weights.mean()
        lat_weights = torch.from_numpy(lat_weights).view(1, 1, -1, 1)
        self.lat_weights = lat_weights


class LatitudeWeightedMSE(LatitudeWeightedMetric):
    """
    Compute latitude-weighted mean-squared error such that errors near the
    top and bottom edges of the prediction are penalized more heavily.
    """
    def forward(self,
                pred: Union[torch.FloatTensor, torch.DoubleTensor],
                target: Union[torch.FloatTensor, torch.DoubleTensor]
            ) -> Union[torch.FloatTensor, torch.DoubleTensor]:
        r"""
        .. highlight:: python

        :param pred: The predicted values of shape [B,C,H,W].
        :type pred: torch.FloatTensor|torch.DoubleTensor
        :param target: The ground truth target values of shape [B,C,H,W].
        :type target: torch.FloatTensor|torch.DoubleTensor

        :return: A singleton tensor if `self.aggregate_only` is `True`. Else, a
            tensor of shape [C+1], where the last element is the aggregate
            MSE, and the preceding elements are the channel-wise MSEs.
        :rtype: torch.FloatTensor|torch.DoubleTensor
        """
        self.lat_weights.to(pred.device)
        error = (pred - target).square() * self.lat_weights
        loss = error.mean()
        if not self.aggregate_only:
            per_channel_losses = error.mean([0,2,3])
            loss = loss.unsqueeze(0)
            loss = torch.cat((per_channel_losses, loss))
        return loss
    

class LatMSE(Metric):
    r"""
    .. highlight:: python

    An alias for `LatitudeWeightedMSE`.
    """
    pass


class LatitudeWeightedRMSE(LatitudeWeightedMetric):
    """
    Compute latitude-weighted root-mean-squared error such that errors near the
    top and bottom edges of the prediction are penalized more heavily.
    """
    
    def forward(self,
                pred: Union[torch.FloatTensor, torch.DoubleTensor],
                target: Union[torch.FloatTensor, torch.DoubleTensor]
            ) -> Union[torch.FloatTensor, torch.DoubleTensor]:
        r"""
        .. highlight:: python

        :param pred: The predicted values of shape [B,C,H,W].
        :type pred: torch.FloatTensor|torch.DoubleTensor
        :param target: The ground truth target values of shape [B,C,H,W].
        :type target: torch.FloatTensor|torch.DoubleTensor

        :return: A singleton tensor if `self.aggregate_only` is `True`. Else, a
            tensor of shape [C+1], where the last element is the aggregate
            MSE, and the preceding elements are the channel-wise MSEs.
        :rtype: torch.FloatTensor|torch.DoubleTensor
        """
        self.lat_weights.to(pred.device)
        error = (pred - target).square() * self.lat_weights
        loss = error.mean().sqrt()
        if not self.aggregate_only:
            per_channel_losses = error.mean([2,3]).sqrt().mean()
            loss = loss.unsqueeze(0)
            loss = torch.cat((per_channel_losses, loss))
        return loss
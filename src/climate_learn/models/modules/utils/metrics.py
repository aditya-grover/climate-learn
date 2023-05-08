# Standard Library
from dataclasses import dataclass
from typing import List, Optional, Union

# Third party
import numpy as np
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


class Metric(nn.Module):
    """Parent class for all ClimateLearn metrics."""
    def __init__(
        self,
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

    def forward(
        self,
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


class LatitudeWeightedMetric(Metric):
    """Parent class for metrics that have a latitude-weighted version."""
    def __init__(self, lat_weighted: bool = False, *args, **kwargs):
        """
        :param lat_weighted: Whether to use latitude-weighting.
        :type lat_weighted: bool
        """
        super().__init__(args, kwargs)
        self.lat_weighted = lat_weighted
        if self.lat_weighted:
            lat_weights = np.cos(np.deg2rad(self.metainfo.lat))
            lat_weights = lat_weights / lat_weights.mean()
            lat_weights = torch.from_numpy(lat_weights).view(1, 1, -1, 1)
            self.lat_weights = lat_weights

    def forward(
        self,
        pred: Union[torch.FloatTensor, torch.DoubleTensor],
        target: Union[torch.FloatTensor, torch.DoubleTensor]
    ) -> None:
        r"""
        .. highlight:: python
        
        Casts latitude weights to the same device as `pred`.
        """
        self.lat_weights.to(device=pred.device)


class ClimatologyBasedMetric(Metric):
    """Parent class for metrics that use climatology."""
    def __init__(self, *args, **kwargs):
        super().__init__(args, kwargs)
        climatology = self.metainfo.climatology
        climatology = climatology.unsqueeze(0)
        self.climatology = climatology

    def forward(
        self,
        pred: Union[torch.FloatTensor, torch.DoubleTensor],
        target: Union[torch.FloatTensor, torch.DoubleTensor]
    ) -> None:
        r"""
        .. highlight:: python
        
        Casts climatology to the same device as `pred`.
        """
        self.climatology.to(device=pred.device)    
    

class MSE(LatitudeWeightedMetric):
    """Computes mean-squared error, with optional latitude-weighting."""
    def forward(
        self,
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
        super().forward(pred, target)
        error = (pred - target).square()
        if self.lat_weighted:            
            error = error * self.lat_weights
        loss = error.mean()
        if not self.aggregate_only:
            per_channel_losses = error.mean([0,2,3])
            loss = loss.unsqueeze(0)
            loss = torch.cat((per_channel_losses, loss))
        return loss


class RMSE(LatitudeWeightedMetric):
    """Computes root mean-squared error, with optional latitude-weighting."""
    def forward(
        self,
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
            RMSE, and the preceding elements are the channel-wise RMSEs.
        :rtype: torch.FloatTensor|torch.DoubleTensor
        """
        super().forward(pred, target)
        error = (pred - target).square()
        if self.lat_weighted:
            error = error * self.lat_weights
        loss = error.mean().sqrt()
        if not self.aggregate_only:
            per_channel_losses = error.mean([2,3]).sqrt().mean(0)
            loss = loss.unsqueeze(0)
            loss = torch.cat((per_channel_losses, loss))
        return loss


class ACC(LatitudeWeightedMetric, ClimatologyBasedMetric):
    """
    Computes the anomaly correlation coefficient, with optional
    latitude-weighting.
    """
    def __init__(self, *args, **kwargs):
        LatitudeWeightedMetric.__init__(self, *args, **kwargs)
        ClimatologyBasedMetric.__init__(self, *args, **kwargs)

    def forward(
        self,
        pred: Union[torch.FloatTensor, torch.DoubleTensor],
        target: Union[torch.FloatTensor, torch.DoubleTensor]
    ) -> Union[torch.FloatTensor, torch.DoubleTensor]:
        r"""
        .. highlight:: python

        :param pred: The predicted values of shape [B,C,H,W]. These should be
            denormalized.
        :type pred: torch.FloatTensor|torch.DoubleTensor
        :param target: The ground truth target values of shape [B,C,H,W]. These
            should be denormalized.
        :type target: torch.FloatTensor|torch.DoubleTensor

        :return: A singleton tensor if `self.aggregate_only` is `True`. Else, a
            tensor of shape [C+1], where the last element is the aggregate
            ACC, and the preceding elements are the channel-wise ACCs.
        :rtype: torch.FloatTensor|torch.DoubleTensor
        """
        LatitudeWeightedMetric.forward(self, pred, target)
        ClimatologyBasedMetric.forward(self, pred, target)
        pred = pred - self.climatology
        target = target - self.climatology
        pred_prime = pred - pred.mean([0,2,3])
        target_prime = target - target.mean([0,2,3])
        if self.lat_weighted:
            numer = (self.lat_weights * pred_prime * target_prime).sum()
            denom1 = (self.lat_weights * pred_prime.square()).sum()
            denom2 = (self.lat_weights * target_prime.square()).sum()
        else:
            numer = (pred_prime * target_prime).sum()
            denom1 = pred_prime.square().sum()
            denom2 = target_prime.square().sum()
        per_channel_losses = numer / (denom1 * denom2).sqrt()
        loss = per_channel_losses.mean()
        if not self.aggregate_only:
            loss = loss.unsqueeze(0)
            loss = torch.cat((per_channel_losses, loss))
        return loss
    

class Pearson(Metric):
    """
    Computes the Pearson correlation coefficient, based on
    https://discuss.pytorch.org/t/use-pearson-correlation-coefficient-as-cost-function/8739/10
    """
    def __init__(self, *args, **kwargs):
        super().__init__(args, kwargs)
        self.cos = nn.CosineSimilarity(dim=1, eps=1e-6)

    def forward(
        self,
        pred: Union[torch.FloatTensor, torch.DoubleTensor],
        target: Union[torch.FloatTensor, torch.DoubleTensor]
    ) -> Union[torch.FloatTensor, torch.DoubleTensor]:
        r"""
        .. highlight:: python

        :param pred: The predicted values of shape [B,C,H,W]. These should be
            denormalized.
        :type pred: torch.FloatTensor|torch.DoubleTensor
        :param target: The ground truth target values of shape [B,C,H,W]. These
            should be denormalized.
        :type target: torch.FloatTensor|torch.DoubleTensor

        :return: A singleton tensor if `self.aggregate_only` is `True`. Else, a
            tensor of shape [C+1], where the last element is the aggregate
            Pearson correlation coefficient, and the preceding elements are the
            channel-wise Pearson correlation coefficients.
        :rtype: torch.FloatTensor|torch.DoubleTensor
        """
        def flatten_channel_wise(x: torch.Tensor) -> torch.Tensor:
            """
            :param x: A tensor of shape [B,C,H,W].
            :type x: torch.Tensor

            :return: A tensor of shape [C,B*H*W].
            :rtype: torch.Tensor
            """
            return torch.stack(
                [xi.flatten() for xi in torch.tensor_split(x, 2, 1)]
            )
        pred = flatten_channel_wise(pred)
        target = flatten_channel_wise(target)
        pred = pred - pred.mean(1, keepdims=True)
        target = target - target.mean(1, keepdims=True)
        per_channel_coeffs = self.cos(pred, target)
        coeff = torch.mean(per_channel_coeffs)
        if not self.aggregate_only:
            coeff = coeff.unsqueeze(0)
            coeff = torch.cat((per_channel_coeffs, coeff))
        return coeff
    

class MeanBias(Metric):
    """Computes the standard mean bias."""
    def forward(
        self,
        pred: Union[torch.FloatTensor, torch.DoubleTensor],
        target: Union[torch.FloatTensor, torch.DoubleTensor]
    ) -> Union[torch.FloatTensor, torch.DoubleTensor]:
        r"""
        .. highlight:: python

        :param pred: The predicted values of shape [B,C,H,W]. These should be
            denormalized.
        :type pred: torch.FloatTensor|torch.DoubleTensor
        :param target: The ground truth target values of shape [B,C,H,W]. These
            should be denormalized.
        :type target: torch.FloatTensor|torch.DoubleTensor

        :return: A singleton tensor if `self.aggregate_only` is `True`. Else, a
            tensor of shape [C+1], where the last element is the aggregate mean
            bias, and the preceding elements are the channel-wise mean bias.
        :rtype: torch.FloatTensor|torch.DoubleTensor
        """
        mean_bias = target.mean() - pred.mean()
        if not self.aggregate_only:            
            per_channel_mean_bias = target.mean([0,2,3]) - pred.mean([0,2,3])
            mean_bias = mean_bias.unsqueeze(0)
            mean_bias = torch.cat((per_channel_mean_bias, mean_bias))
        return mean_bias

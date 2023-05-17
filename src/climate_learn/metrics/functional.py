# Standard library
from typing import Optional, Union

# Third party
import torch
import torch.nn.functional as F


def mse(
    pred: Union[torch.FloatTensor, torch.DoubleTensor],
    target: Union[torch.FloatTensor, torch.DoubleTensor],
    aggregate_only: bool = False,
    lat_weights: Optional[Union[torch.FloatTensor, torch.DoubleTensor]] = None
) -> Union[torch.FloatTensor, torch.DoubleTensor]:
    error = (pred - target).square()
    if lat_weights is not None:
        error = error * lat_weights
    loss = error.mean()
    if not aggregate_only:
        per_channel_losses = error.mean([0, 2, 3])
        loss = loss.unsqueeze(0)
        loss = torch.cat((per_channel_losses, loss))
    return loss


def rmse(
    pred: Union[torch.FloatTensor, torch.DoubleTensor],
    target: Union[torch.FloatTensor, torch.DoubleTensor],
    aggregate_only: bool = False,
    lat_weights: Optional[Union[torch.FloatTensor, torch.DoubleTensor]] = None
) -> Union[torch.FloatTensor, torch.DoubleTensor]:
    error = (pred - target).square()
    if lat_weights is not None:
        error = error * lat_weights
    loss = error.mean().sqrt()
    if not aggregate_only:
        per_channel_losses = error.mean([0, 2, 3]).sqrt()
        loss = loss.unsqueeze(0)
        loss = torch.cat((per_channel_losses, loss))
    return loss


def acc(
    pred: Union[torch.FloatTensor, torch.DoubleTensor],
    target: Union[torch.FloatTensor, torch.DoubleTensor],
    climatology: Optional[Union[torch.FloatTensor, torch.DoubleTensor]],
    aggregate_only: bool = False,
    lat_weights: Optional[Union[torch.FloatTensor, torch.DoubleTensor]] = None
) -> Union[torch.FloatTensor, torch.DoubleTensor]:
    pred = pred - climatology
    target = target - climatology
    pred_prime = pred - pred.mean([0, 2, 3], keepdims=True)
    target_prime = target - target.mean([0, 2, 3], keepdims=True)
    if lat_weights is not None:
        numer = (lat_weights * pred_prime * target_prime).sum([0, 2, 3])
        denom1 = (lat_weights * pred_prime.square()).sum([0, 2, 3])
        denom2 = (lat_weights * target_prime.square()).sum([0, 2, 3])
    else:
        numer = (pred_prime * target_prime).sum([0, 2, 3])
        denom1 = pred_prime.square().sum([0, 2, 3])
        denom2 = target_prime.square().sum([0, 2, 3])
    per_channel_losses = numer / (denom1 * denom2).sqrt()
    loss = per_channel_losses.mean()
    if not aggregate_only:
        loss = loss.unsqueeze(0)
        loss = torch.cat((per_channel_losses, loss))
    return loss


def pearson(
    pred: Union[torch.FloatTensor, torch.DoubleTensor],
    target: Union[torch.FloatTensor, torch.DoubleTensor],
    aggregate_only: bool = False
) -> Union[torch.FloatTensor, torch.DoubleTensor]:
    pred = _flatten_channel_wise(pred)
    target = _flatten_channel_wise(target)
    pred = pred - pred.mean(1, keepdims=True)
    target = target - target.mean(1, keepdims=True)
    per_channel_coeffs = F.cosine_similarity(pred, target)
    coeff = torch.mean(per_channel_coeffs)
    if not aggregate_only:
        coeff = coeff.unsqueeze(0)
        coeff = torch.cat((per_channel_coeffs, coeff))
    return coeff


def mean_bias(
    pred: Union[torch.FloatTensor, torch.DoubleTensor],
    target: Union[torch.FloatTensor, torch.DoubleTensor],
    aggregate_only: bool = False
) -> Union[torch.FloatTensor, torch.DoubleTensor]:
    result = target.mean() - pred.mean()
    if not aggregate_only:
        per_channel_mean_bias = target.mean([0, 2, 3]) - pred.mean([0, 2, 3])
        result = result.unsqueeze(0)
        result = torch.cat((per_channel_mean_bias, result))
    return result


def _flatten_channel_wise(x: torch.Tensor) -> torch.Tensor:
    """
    :param x: A tensor of shape [B,C,H,W].
    :type x: torch.Tensor

    :return: A tensor of shape [C,B*H*W].
    :rtype: torch.Tensor
    """
    return torch.stack([xi.flatten() for xi in torch.tensor_split(x, 2, 1)])
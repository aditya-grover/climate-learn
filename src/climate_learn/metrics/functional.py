# Standard library
from typing import Optional, Union

# Third party
import torch
import torch.nn.functional as F
from torch.distributions.normal import Normal


def mse(
    pred: Union[torch.FloatTensor, torch.DoubleTensor, Normal],
    target: Union[torch.FloatTensor, torch.DoubleTensor],
    aggregate_only: bool = False,
    lat_weights: Optional[Union[torch.FloatTensor, torch.DoubleTensor]] = None,
) -> Union[torch.FloatTensor, torch.DoubleTensor]:
    if isinstance(pred, Normal):
        pred = pred.loc
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
    pred: Union[torch.FloatTensor, torch.DoubleTensor, Normal],
    target: Union[torch.FloatTensor, torch.DoubleTensor],
    aggregate_only: bool = False,
    lat_weights: Optional[Union[torch.FloatTensor, torch.DoubleTensor]] = None,
) -> Union[torch.FloatTensor, torch.DoubleTensor]:
    if isinstance(pred, Normal):
        pred = pred.loc
    error = (pred - target).square()
    if lat_weights is not None:
        error = error * lat_weights
    loss = error.mean().sqrt()
    if not aggregate_only:
        per_channel_losses = error.mean([0, 2, 3]).sqrt()
        loss = loss.unsqueeze(0)
        loss = torch.cat((per_channel_losses, loss))
    return loss


def spread(
    pred: Normal,
    target: Union[torch.FloatTensor, torch.DoubleTensor],
    aggregate_only: bool = False,
    lat_weights: Optional[Union[torch.FloatTensor, torch.DoubleTensor]] = None,
) -> Union[torch.FloatTensor, torch.DoubleTensor]:
    std_pred = pred.scale
    var_pred = std_pred**2
    if lat_weights is not None:
        var_pred = var_pred * lat_weights
    spread_score = var_pred.mean().sqrt()
    if not aggregate_only:
        per_channel_spreads = var_pred.mean([0, 2, 3]).sqrt()
        spread_score = spread_score.unsqueeze(0)
        spread_score = torch.cat((per_channel_spreads, spread_score))
    return spread_score


def spread_skill(
    pred: Normal,
    target: Union[torch.FloatTensor, torch.DoubleTensor],
    aggregate_only: bool = False,
    lat_weights: Optional[Union[torch.FloatTensor, torch.DoubleTensor]] = None,
) -> Union[torch.FloatTensor, torch.DoubleTensor]:
    return spread(pred, target, aggregate_only, lat_weights) / rmse(pred, target, aggregate_only, lat_weights)


def crps_gaussian(
    pred: Normal,
    target: Union[torch.FloatTensor, torch.DoubleTensor],
    aggregate_only: bool = False,
    lat_weights: Optional[Union[torch.FloatTensor, torch.DoubleTensor]] = None,
) -> Union[torch.FloatTensor, torch.DoubleTensor]:
    mean, std = pred.loc, pred.scale
    s = (target - mean) / std

    standard_normal = Normal(torch.zeros_like(target), torch.ones_like(target))
    cdf = standard_normal.cdf(s)
    pdf = torch.exp(standard_normal.log_prob(s))

    crps = std * (s * (2 * cdf - 1) + 2 * pdf - 1 / torch.pi)

    if lat_weights is not None:
        crps = crps * lat_weights
    crps_score = crps.mean()
    if not aggregate_only:
        per_channel_crps = crps.mean([0, 2, 3])
        crps_score = crps_score.unsqueeze(0)
        crps_score = torch.cat((per_channel_crps, crps_score))
    return crps_score


def acc(
    pred: Union[torch.FloatTensor, torch.DoubleTensor, Normal],
    target: Union[torch.FloatTensor, torch.DoubleTensor],
    climatology: Optional[Union[torch.FloatTensor, torch.DoubleTensor]],
    aggregate_only: bool = False,
    lat_weights: Optional[Union[torch.FloatTensor, torch.DoubleTensor]] = None,
) -> Union[torch.FloatTensor, torch.DoubleTensor]:
    if isinstance(pred, Normal):
        pred = pred.loc
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
    pred: Union[torch.FloatTensor, torch.DoubleTensor, Normal],
    target: Union[torch.FloatTensor, torch.DoubleTensor],
    aggregate_only: bool = False,
) -> Union[torch.FloatTensor, torch.DoubleTensor]:
    if isinstance(pred, Normal):
        pred = pred.loc
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
    pred: Union[torch.FloatTensor, torch.DoubleTensor, Normal],
    target: Union[torch.FloatTensor, torch.DoubleTensor],
    aggregate_only: bool = False,
) -> Union[torch.FloatTensor, torch.DoubleTensor]:
    if isinstance(pred, Normal):
        pred = pred.loc
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

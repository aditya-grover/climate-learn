import numpy as np
import torch
from scipy import stats
from torch.distributions.normal import Normal


### Training loss


def mse(pred, y, vars, lat=None, log_postfix=""):
    """
    y: [B, C, H, W]
    pred: [B, C, H, W]
    vars: list of variable names
    """
    error = (pred - y) ** 2

    loss_dict = {}

    with torch.no_grad():
        for i, var in enumerate(vars):
            loss_dict[f"mse_{var}_{log_postfix}"] = error[:, i].mean()
    loss_dict["loss"] = error.mean()

    return loss_dict


def lat_weighted_mse(pred, y, vars, lat, log_postfix=""):
    """
    y: [B, C, H, W]
    pred: [C, C, H, W]
    vars: list of variable names
    lat: H
    """
    error = (pred - y) ** 2

    # lattitude weights
    w_lat = np.cos(np.deg2rad(lat))
    w_lat = w_lat / w_lat.mean()  # (H, )
    w_lat = (
        torch.from_numpy(w_lat).unsqueeze(0).unsqueeze(-1).to(error.device)
    )  # (1, H, 1)

    loss_dict = {}
    with torch.no_grad():
        for i, var in enumerate(vars):
            loss_dict[f"w_mse_{var}_{log_postfix}"] = (error[:, i] * w_lat).mean()

    loss_dict["loss"] = torch.mean(error * w_lat.unsqueeze(1))
    return loss_dict


### Forecasting metrics


def lat_weighted_mse_val(pred, y, transform, vars, lat, clim, log_postfix):
    """
    y: [B, C, H, W]
    pred: [B, C, H, W]
    vars: list of variable names
    lat: H
    """
    error = (pred - y) ** 2  # [B, C, H, W]

    # lattitude weights
    w_lat = np.cos(np.deg2rad(lat))
    w_lat = w_lat / w_lat.mean()  # (H, )
    w_lat = (
        torch.from_numpy(w_lat).unsqueeze(0).unsqueeze(-1).to(error.device)
    )  # (1, H, 1)

    loss_dict = {}
    with torch.no_grad():
        for i, var in enumerate(vars):
            loss_dict[f"w_mse_{var}_{log_postfix}"] = (error[:, i] * w_lat).mean()

    loss_dict["w_mse"] = np.mean([loss_dict[k].cpu() for k in loss_dict.keys()])

    return loss_dict


def lat_weighted_rmse(pred, y, transform, vars, lat, clim, log_postfix, transform_pred=True):
    """
    y: [B, C, H, W]
    pred: [B, C, H, W]
    vars: list of variable names
    lat: H
    """
    if transform_pred:
        pred = transform(pred)
    y = transform(y)

    error = (pred - y) ** 2  # [B, C, H, W]

    # lattitude weights
    w_lat = np.cos(np.deg2rad(lat))
    w_lat = w_lat / w_lat.mean()  # (H, )
    w_lat = (
        torch.from_numpy(w_lat)
        .unsqueeze(0)
        .unsqueeze(-1)
        .to(dtype=error.dtype, device=error.device)
    )

    loss_dict = {}
    with torch.no_grad():
        for i, var in enumerate(vars):
            loss_dict[f"w_rmse_{var}_{log_postfix}"] = torch.mean(
                torch.sqrt(torch.mean(error[:, i] * w_lat, dim=(-2, -1)))
            )

    loss_dict["w_rmse"] = np.mean([loss_dict[k].cpu() for k in loss_dict.keys()])

    return loss_dict


def lat_weighted_acc(pred, y, transform, vars, lat, clim, log_postfix):
    """
    y: [B, C, H, W]
    pred: [B C, H, W]
    vars: list of variable names
    lat: H
    """

    pred = transform(pred)
    y = transform(y)

    # lattitude weights
    w_lat = np.cos(np.deg2rad(lat))
    w_lat = w_lat / w_lat.mean()  # (H, )
    w_lat = (
        torch.from_numpy(w_lat)
        .unsqueeze(0)
        .unsqueeze(-1)
        .to(dtype=pred.dtype, device=pred.device)
    )  # [1, H, 1]

    # clim = torch.mean(y, dim=(0, 1), keepdim=True)
    clim = clim.to(device=y.device).unsqueeze(0)
    pred = pred - clim
    y = y - clim
    loss_dict = {}

    with torch.no_grad():
        for i, var in enumerate(vars):
            pred_prime = pred[:, i] - torch.mean(pred[:, i])
            y_prime = y[:, i] - torch.mean(y[:, i])
            loss_dict[f"acc_{var}_{log_postfix}"] = torch.sum(
                w_lat * pred_prime * y_prime
            ) / torch.sqrt(
                torch.sum(w_lat * pred_prime**2) * torch.sum(w_lat * y_prime**2)
            )

    loss_dict["acc"] = np.mean([loss_dict[k].cpu() for k in loss_dict.keys()])

    return loss_dict

### uncertainty metrics


def lat_weighted_nll(pred, y, transform, vars, lat, clim, log_postfix):
    """
    y: [B, C, H, W]
    pred: Normal
    vars: list of variable names
    lat: H
    """
    assert type(pred) == Normal

    error = - pred.log_prob(y)

    # lattitude weights
    w_lat = np.cos(np.deg2rad(lat))
    w_lat = w_lat / w_lat.mean()  # (H, )
    w_lat = (
        torch.from_numpy(w_lat)
        .unsqueeze(0)
        .unsqueeze(-1)
        .to(dtype=error.dtype, device=error.device)
    )

    loss_dict = {}
    with torch.no_grad():
        for i, var in enumerate(vars):
            loss_dict[f"w_nll_{var}_{log_postfix}"] = torch.mean(error[:, i] * w_lat)

    loss_dict["w_nll"] = np.mean([loss_dict[k].cpu() for k in loss_dict.keys()])

    return loss_dict


def crps_gaussian(pred, y, transform, vars, lat, clim, log_postfix):
    """
    y: [B, C, H, W]
    pred: Normal
    vars: list of variable names
    lat: H
    """
    mean, std = pred.loc, pred.scale
    assert std is not None

    s = (y - mean) / std

    standard_normal = Normal(torch.zeros_like(y), torch.ones_like(y))
    cdf = standard_normal.cdf(s)
    pdf = torch.exp(standard_normal.log_prob(s))

    crps = std * (s * (2*cdf - 1) + 2*pdf - 1/torch.sqrt(torch.pi))

    # lattitude weights
    w_lat = np.cos(np.deg2rad(lat))
    w_lat = w_lat / w_lat.mean()  # (H, )
    w_lat = (
        torch.from_numpy(w_lat)
        .unsqueeze(0)
        .unsqueeze(-1)
        .to(dtype=crps.dtype, device=crps.device)
    )

    loss_dict = {}
    with torch.no_grad():
        for i, var in enumerate(vars):
            loss_dict[f"crps_{var}_{log_postfix}"] = torch.mean(crps[:, i] * w_lat)

    loss_dict["w_crps"] = np.mean([loss_dict[k].cpu() for k in loss_dict.keys()])

    return loss_dict


def lat_weighted_spread_skill_ratio(pred, y, transform, vars, lat, clim, log_postfix):
    """
    y: [B, C, H, W]
    pred: Normal
    vars: list of variable names
    lat: H
    """
    assert type(pred) == Normal

    mean, variance = pred.loc, torch.square(pred.scale)

    error = (mean - y) ** 2

    # lattitude weights
    w_lat = np.cos(np.deg2rad(lat))
    w_lat = w_lat / w_lat.mean()  # (H, )
    w_lat = (
        torch.from_numpy(w_lat)
        .unsqueeze(0)
        .unsqueeze(-1)
        .to(dtype=crps.dtype, device=crps.device)
    )

    loss_dict = {}
    with torch.no_grad():
        for i, var in enumerate(vars):
            spread = torch.mean(torch.sqrt(torch.mean(variance[:, i] * w_lat, dim=(-2, -1))))
            rmse = torch.mean(torch.sqrt(torch.mean(error[:, i] * w_lat, dim=(-2, -1))))
            loss_dict[f"spread_{var}_{log_postfix}"] = spread / rmse

    loss_dict["w_spread"] = np.mean([loss_dict[k].cpu() for k in loss_dict.keys()])

    return loss_dict


def categorical_loss(pred, y, transform, vars, lat, clim, log_postfix):
    """
    y: [B, 50, C, H, W]
    pred: [B, 50, C, H, W]
    vars: list of variable names
    lat: H
    """
    loss = torch.nn.CrossEntropyLoss(reduction='none')
    # get the labels [128, 1, 32, 64]
    _, labels = y.max(dim=1) # y.shape = pred.shape = [128, 50, 1, 32, 64] 
    error = loss(pred, labels.to(pred.device)) # error.shape [128, 1, 32, 64]
    print(error.shape)

    # lattitude weights
    w_lat = np.cos(np.deg2rad(lat))
    w_lat = w_lat / w_lat.mean()  # (H, )
    w_lat = (
        torch.from_numpy(w_lat)
        .unsqueeze(0)
        .unsqueeze(-1)
        .to(dtype=error.dtype, device=error.device)
    )

    print(w_lat.shape)

    loss_dict = {}
    with torch.no_grad():
        for i, var in enumerate(vars):
            loss_dict[f"categorical_{var}_{log_postfix}"] = torch.mean(error[:, i] * w_lat)

    loss_dict["w_categorical"] = np.mean([loss_dict[k].cpu() for k in loss_dict.keys()])

    return loss_dict

### Downscaling metrics



def mse_val(pred, y, transform, vars, lat, clim, log_postfix):
    """
    y: [B, C, H, W]
    pred: [B, C, H, W]
    vars: list of variable names
    """
    error = (pred - y) ** 2  # [B, C, H, W]

    loss_dict = {}

    with torch.no_grad():
        for i, var in enumerate(vars):
            loss_dict[f"mse_{var}"] = error[:, i].mean()

    loss_dict["mse"] = np.mean([loss_dict[k].cpu() for k in loss_dict.keys()])

    return loss_dict


def rmse(pred, y, transform, vars, lat, clim, log_postfix):
    """
    y: [B, C, H, W]
    pred: [B, C, H, W]
    vars: list of variable names
    """
    pred = transform(pred)
    y = transform(y)
    pred = pred.to(torch.float32)
    y = y.to(torch.float32)

    error = (pred - y) ** 2  # [N, C, H, W]

    loss_dict = {}
    with torch.no_grad():
        for i, var in enumerate(vars):
            loss_dict[f"rmse_{var}"] = torch.mean(
                torch.sqrt(torch.mean(error[:, i], dim=(-2, -1)))
            )

    loss_dict["rmse"] = np.mean([loss_dict[k].cpu() for k in loss_dict.keys()])

    return loss_dict


def pearson(pred, y, transform, vars, lat, clim, log_postfix):
    """
    y: [B, C, H, W]
    pred: [B, C, H, W]
    vars: list of variable names
    """
    pred = transform(pred)
    y = transform(y)
    pred = pred.to(torch.float32)
    y = y.to(torch.float32)

    loss_dict = {}
    with torch.no_grad():
        for i, var in enumerate(vars):
            loss_dict[f"pearsonr_{var}"] = stats.pearsonr(
                pred[:, i].flatten().cpu().numpy(), y[:, i].flatten().cpu().numpy()
            )[0]

    loss_dict["pearson"] = np.mean([loss_dict[k] for k in loss_dict.keys()])

    return loss_dict


def mean_bias(pred, y, transform, vars, lat, clim, log_postfix):
    """
    y: [B, C, H, W]
    pred: [B, C, H, W]
    vars: list of variable names
    """
    pred = transform(pred)
    y = transform(y)
    pred = pred.to(torch.float32)
    y = y.to(torch.float32)

    loss_dict = {}
    with torch.no_grad():
        for i, var in enumerate(vars):
            loss_dict[f"mean_bias_{var}"] = y.mean() - pred.mean()

    loss_dict["mean_bias"] = np.mean([loss_dict[k].cpu() for k in loss_dict.keys()])

    return loss_dict

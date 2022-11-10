import numpy as np
import torch
from scipy import stats
from torch.distributions.normal import Normal


def mse(pred, y, vars, lat=None, mask=None):
    """
    y: [N, 3, H, W]
    pred: [N, L, p*p*3]
    vars: list of variable names
    """
    loss = (pred - y) ** 2

    if mask is None:
        mask = torch.ones_like(loss)[:, 0]

    loss_dict = {}

    with torch.no_grad():
        for i, var in enumerate(vars):
            loss_dict[var] = (loss[:, i] * mask).sum() / mask.sum()
    loss_dict["loss"] = (loss.mean(dim=1) * mask).sum() / mask.sum()

    return loss_dict


def lat_weighted_mse(pred, y, vars, lat, mask=None):
    """
    y: [N, C, H, W]
    pred: [N, C, H, W]
    vars: list of variable names
    lat: H
    """

    error = (pred - y) ** 2  # [N, C, H, W]

    # lattitude weights
    w_lat = np.cos(np.deg2rad(lat))
    w_lat = w_lat / w_lat.mean()  # (H, )
    w_lat = torch.from_numpy(w_lat).unsqueeze(0).unsqueeze(-1).to(error.device)  # (1, H, 1)

    if mask is None:
        mask = torch.ones_like(error)[:, 0]

    loss_dict = {}
    with torch.no_grad():
        for i, var in enumerate(vars):
            loss_dict[var] = (error[:, i] * w_lat * mask).sum() / mask.sum()

    loss_dict["loss"] = ((error * w_lat.unsqueeze(1)).mean(dim=1) * mask).sum() / mask.sum()
    return loss_dict


def lat_weighted_mse_val(pred, y, clim, transform, vars, lat, log_steps, log_days):
    """
    y: [N, T, 3, H, W]
    pred: [N, T, 3, H, W]
    vars: list of variable names
    lat: H
    """

    error = (pred - y) ** 2  # [N, T, C, H, W]

    # lattitude weights
    w_lat = np.cos(np.deg2rad(lat))
    w_lat = w_lat / w_lat.mean()  # (H, )
    w_lat = torch.from_numpy(w_lat).unsqueeze(0).unsqueeze(-1).to(error.device)  # (1, H, 1)

    loss_dict = {}
    with torch.no_grad():
        for i, var in enumerate(vars):
            for day, step in zip(log_days, log_steps):
                loss_dict[f"w_mse_{var}_day_{day}"] = (error[:, step - 1, i] * w_lat).mean()

    # loss_dict["w_mse"] = np.mean([loss_dict[k].cpu() for k in loss_dict.keys()])

    return loss_dict


def lat_weighted_nll(pred: Normal, y, vars, lat, mask=None):
    """
    y: [N, C, H, W]
    pred: [N, C, H, W]
    vars: list of variable names
    """
    assert type(pred) == Normal

    error = - pred.log_prob(y)  # [N, C, H, W]

    # lattitude weights
    w_lat = np.cos(np.deg2rad(lat))
    w_lat = w_lat / w_lat.mean()  # (H, )
    w_lat = torch.from_numpy(w_lat).unsqueeze(0).unsqueeze(-1).to(error.device)  # (1, H, 1)

    loss_dict = {}
    with torch.no_grad():
        for i, var in enumerate(vars):
            loss_dict[f"w_nll_{var}"] = torch.mean(error[:, i] * w_lat)

    loss_dict["loss"] = torch.mean((error * w_lat.unsqueeze(1)).mean(dim=1))
    return loss_dict


def crps_gaussian(pred: Normal, y, vars, lat, mask=None):
    """
    y: [N, C, H, W]
    pred: [N, C, H, W]
    vars: list of variable names
    """
    mean, std = pred.loc, pred.scale # N, C, H, W
    assert std is not None

    s = (y - mean) / std

    standard_normal = Normal(torch.zeros_like(y), torch.ones_like(y))
    cdf = standard_normal.cdf(s)
    pdf = torch.exp(standard_normal.log_prob(s))

    crps = std * (s * (2*cdf - 1) + 2*pdf - 1/torch.pi)

    # lattitude weights
    w_lat = np.cos(np.deg2rad(lat))
    w_lat = w_lat / w_lat.mean()  # (H, )
    w_lat = torch.from_numpy(w_lat).unsqueeze(0).unsqueeze(-1).to(crps.device)  # (1, H, 1)

    loss_dict = {}
    for i, var in enumerate(vars):
        loss_dict[f"crps_{var}"] = torch.mean(crps[:, i] * w_lat)
    loss_dict["loss"] = torch.mean((crps * w_lat.unsqueeze(1)).mean(dim=1))

    return loss_dict


def crps_gaussian_val(pred, y, clim, vars, lat, log_day):
    """
    y: [N, C, H, W]
    pred: [N, C, H, W]
    vars: list of variable names
    """
    mean, std = pred.loc, pred.scale # N, C, H, W
    assert std is not None

    s = (y - mean) / std

    standard_normal = Normal(torch.zeros_like(y), torch.ones_like(y))
    cdf = standard_normal.cdf(s)
    pdf = torch.exp(standard_normal.log_prob(s))

    crps = std * (s * (2*cdf - 1) + 2*pdf - 1/torch.pi)

    # lattitude weights
    w_lat = np.cos(np.deg2rad(lat))
    w_lat = w_lat / w_lat.mean()  # (H, )
    w_lat = torch.from_numpy(w_lat).unsqueeze(0).unsqueeze(-1).to(crps.device)  # (1, H, 1)

    loss_dict = {}
    for i, var in enumerate(vars):
        loss_dict[f"crps_{var}_day_{log_day}"] = torch.mean(crps[:, i] * w_lat)

    return loss_dict
### Forecasting metrics


def lat_weighted_rmse(pred, y, clim, transform, vars, lat, log_steps, log_days, transform_pred=True):
    """
    y: [N, T, 3, H, W]
    pred: [N, T, 3, H, W]
    vars: list of variable names
    lat: H
    """
    if transform_pred:
        pred = transform(pred)
    y = transform(y)
    pred = pred.to(torch.float32)
    y = y.to(torch.float32)

    error = (pred - y) ** 2  # [N, T, 3, H, W]

    # lattitude weights
    w_lat = np.cos(np.deg2rad(lat))
    w_lat = w_lat / w_lat.mean()  # (H, )
    w_lat = torch.from_numpy(w_lat).unsqueeze(0).unsqueeze(-1).to(error.device)

    loss_dict = {}
    with torch.no_grad():
        for i, var in enumerate(vars):
            for day, step in zip(log_days, log_steps):
                loss_dict[f"w_rmse_{var}_day_{day}"] = torch.mean(
                    torch.sqrt(torch.mean(error[:, step - 1, i] * w_lat, dim=(-2, -1)))
                )

    # loss_dict["w_rmse"] = np.mean([loss_dict[k].cpu() for k in loss_dict.keys()])
    return loss_dict


def lat_weighted_acc(pred, y, clim, transform, vars, lat, log_steps, log_days, transform_pred=True):
    """
    y: [N, T, 3, H, W]
    pred: [N, T, 3, H, W]
    vars: list of variable names
    lat: H
    TODO: subtract the climatology
    """
    if transform_pred:
        pred = transform(pred)
    y = transform(y)
    pred = pred.to(torch.float32)
    y = y.to(torch.float32)

    # lattitude weights
    w_lat = np.cos(np.deg2rad(lat))
    w_lat = w_lat / w_lat.mean()  # (H, )
    w_lat = torch.from_numpy(w_lat).unsqueeze(0).unsqueeze(-1).to(pred.device)  # [1, H, 1]

    # clim = torch.mean(y, dim=(0, 1), keepdim=True)
    clim = clim.to(pred.device)
    pred = pred - clim
    y = y - clim
    loss_dict = {}

    with torch.no_grad():
        for i, var in enumerate(vars):
            for day, step in zip(log_days, log_steps):
                pred_prime = pred[:, step - 1, i] - torch.mean(pred[:, step - 1, i])
                y_prime = y[:, step - 1, i] - torch.mean(y[:, step - 1, i])
                loss_dict[f"acc_{var}_day_{day}"] = torch.sum(w_lat * pred_prime * y_prime) / torch.sqrt(
                    torch.sum(w_lat * pred_prime**2) * torch.sum(w_lat * y_prime**2)
                )

    # loss_dict["acc"] = np.mean([loss_dict[k].cpu() for k in loss_dict.keys()])
    return loss_dict

def categorical_loss(pred, y, clim, transform, vars, lat, log_steps, log_days, transform_pred=True):
    # """
    # y: [N, T, 3, H, W]
    # pred: [N, T, 3, H, W]
    # vars: list of variable names
    # lat: H
    # TODO: subtract the climatology
    # """
    # if transform_pred:
    #     pred = transform(pred)
    # y = transform(y)
    # pred = pred.to(torch.float32)
    # y = y.to(torch.float32)

    # # lattitude weights
    # w_lat = np.cos(np.deg2rad(lat))
    # w_lat = w_lat / w_lat.mean()  # (H, )
    # w_lat = torch.from_numpy(w_lat).unsqueeze(0).unsqueeze(-1).to(pred.device)  # [1, H, 1]

    # # clim = torch.mean(y, dim=(0, 1), keepdim=True)
    # clim = clim.to(pred.device)
    # pred = pred - clim
    # y = y - clim
    loss_dict = {}

    # with torch.no_grad():
    #     for i, var in enumerate(vars):
    #         for day, step in zip(log_days, log_steps):
    #             pred_prime = pred[:, step - 1, i] - torch.mean(pred[:, step - 1, i])
    #             y_prime = y[:, step - 1, i] - torch.mean(y[:, step - 1, i])
    #             loss_dict[f"acc_{var}_day_{day}"] = torch.sum(w_lat * pred_prime * y_prime) / torch.sqrt(
    #                 torch.sum(w_lat * pred_prime**2) * torch.sum(w_lat * y_prime**2)
    #             )

    # # loss_dict["acc"] = np.mean([loss_dict[k].cpu() for k in loss_dict.keys()])
    return loss_dict

### Downscaling metrics
def rmse(pred, y, transform, vars):
    """
    y: [N, C, H, W]
    pred: [N, C, H, W]
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

    return loss_dict

def pearson(pred, y, transform, vars):
    """
    y: [N, C, H, W]
    pred: [N, C, H, W]
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
                pred[:, i].flatten().cpu().numpy(),
                y[:, i].flatten().cpu().numpy()
            )[0]

    return loss_dict

def mean_bias(pred, y, transform, vars):
    """
    y: [N, C, H, W]
    pred: [N, C, H, W]
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

    return loss_dict
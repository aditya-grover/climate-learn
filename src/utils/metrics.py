import numpy as np
import torch


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


def lat_weighted_mse_val(pred, y, transform, vars, lat, log_steps, log_days):
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

    loss_dict["w_mse"] = np.mean([loss_dict[k].cpu() for k in loss_dict.keys()])

    return loss_dict


def lat_weighted_rmse(pred, y, transform, vars, lat, log_steps, log_days):
    """
    y: [N, T, 3, H, W]
    pred: [N, T, 3, H, W]
    vars: list of variable names
    lat: H
    """
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

    loss_dict["w_rmse"] = np.mean([loss_dict[k].cpu() for k in loss_dict.keys()])
    return loss_dict


def lat_weighted_acc(pred, y, transform, vars, lat, log_steps, log_days):
    """
    y: [N, T, 3, H, W]
    pred: [N, T, 3, H, W]
    vars: list of variable names
    lat: H
    TODO: subtract the climatology
    """
    pred = transform(pred)
    y = transform(y)
    pred = pred.to(torch.float32)
    y = y.to(torch.float32)

    # lattitude weights
    w_lat = np.cos(np.deg2rad(lat))
    w_lat = w_lat / w_lat.mean()  # (H, )
    w_lat = torch.from_numpy(w_lat).unsqueeze(0).unsqueeze(-1).to(pred.device)  # [1, H, 1]

    clim = torch.mean(y, dim=(0, 1), keepdim=True)
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

    loss_dict["acc"] = np.mean([loss_dict[k].cpu() for k in loss_dict.keys()])
    return loss_dict


# def compute_weighted_acc(da_fc, da_true, mean_dims):
#     """
#     Compute the ACC with latitude weighting from two xr.DataArrays.
#     WARNING: Does not work if datasets contain NaNs
#     Args:
#         da_fc (xr.DataArray): Forecast. Time coordinate must be validation time.
#         da_true (xr.DataArray): Truth.
#         mean_dims: dimensions over which to average score
#     Returns:
#         acc: Latitude weighted acc
#     """

#     clim = da_true.mean("time")
#     try:
#         t = np.intersect1d(da_fc.time, da_true.time)
#         fa = da_fc.sel(time=t) - clim
#     except AttributeError:
#         t = da_true.time.values
#         fa = da_fc - clim
#     a = da_true.sel(time=t) - clim

#     weights_lat = np.cos(np.deg2rad(da_fc.lat))
#     weights_lat /= weights_lat.mean()
#     w = weights_lat

#     fa_prime = fa - fa.mean()
#     a_prime = a - a.mean()

#     acc = np.sum(w * fa_prime * a_prime) / np.sqrt(
#         np.sum(w * fa_prime ** 2) * np.sum(w * a_prime ** 2)
#     )
#     return acc


# pred = torch.randn(2, 4, 3, 128, 256).cuda()
# y = torch.randn(2, 4, 3, 128, 256).cuda()
# vars = ["x", "y", "z"]
# print(lat_weighted_rmse(pred, y, vars))
# print(lat_weighted_acc(pred, y, vars))

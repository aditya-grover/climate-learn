import torch
import numpy as np
from torch.distributions.normal import Normal
from torch.nn.functional import one_hot
import properscoring as ps
from climate_learn.models.modules.utils.metrics import (
    lat_weighted_categorical_loss,
    lat_weighted_spread_skill_ratio,
    lat_weighted_crps_gaussian,
    lat_weighted_nll,
)


def test_lat_weighted_nll():
    # create test input tensors
    batch_size = 128
    num_channels = 3
    height = 32
    width = 64
    loc = torch.randn(batch_size, num_channels, height, width)
    scale = torch.ones(batch_size, num_channels, height, width)
    pred = Normal(loc, scale)
    y = torch.randn(batch_size, num_channels, height, width)

    # call the lat_weighted_nll function
    transform = None
    vars = ["var1", "var2", "var3"]
    lat = np.random.rand(height)
    clim = None
    log_postfix = "test"
    loss_dict = lat_weighted_nll(pred, y, transform, vars, lat, clim, log_postfix)

    # check the shape of the output dictionary
    assert len(loss_dict) == len(vars) + 1  # +1 for "w_nll" key

    # check the shape and type of each loss
    for var in vars:
        loss_key = f"w_nll_{var}_{log_postfix}"
        assert loss_key in loss_dict
        assert isinstance(loss_dict[loss_key], torch.Tensor)

    assert isinstance(loss_dict["w_nll"], np.float32)

    print("\nnll:\n", loss_dict)


def test_crps_gaussian():
    # create test input tensors
    batch_size = 128
    num_channels = 3
    height = 32
    width = 64
    loc = torch.randn(batch_size, num_channels, height, width)
    scale = torch.ones(batch_size, num_channels, height, width)
    pred = Normal(loc, scale)
    y = torch.randn(batch_size, num_channels, height, width)

    # call the lat_weighted_crps_gaussian function
    transform = None
    vars = ["var1", "var2", "var3"]
    lat = np.random.rand(height)
    clim = None
    log_postfix = "test"
    loss_dict = lat_weighted_crps_gaussian(
        pred, y, transform, vars, lat, clim, log_postfix
    )

    # check the shape of the output dictionary
    assert len(loss_dict) == len(vars) + 1  # +1 for "w_crps" key

    # checking result with crps from properscoring
    crps_score = torch.from_numpy(ps.crps_gaussian(y, loc, scale))

    w_lat = np.cos(np.deg2rad(lat))
    w_lat = w_lat / w_lat.mean()
    w_lat = (
        torch.from_numpy(w_lat)
        .unsqueeze(0)
        .unsqueeze(-1)
        .to(dtype=crps_score.dtype, device=crps_score.device)
    )

    loss_dict_ps = {}
    with torch.no_grad():
        for i, var in enumerate(vars):
            loss_dict_ps[f"w_crps_{var}_{log_postfix}"] = torch.mean(
                crps_score[:, i] * w_lat
            )

    loss_dict_ps["w_crps"] = np.mean(
        [loss_dict_ps[k].cpu() for k in loss_dict_ps.keys()]
    )

    # check the shape and type of each loss, check value with ps.crps_gaussian
    for var in vars:
        loss_key = f"w_crps_{var}_{log_postfix}"
        assert loss_key in loss_dict
        assert isinstance(loss_dict[loss_key], torch.Tensor)
        assert abs(loss_dict[loss_key].item() - loss_dict_ps[loss_key].item()) < 1e-6

    assert isinstance(loss_dict["w_crps"], np.float32)
    assert (loss_dict["w_crps"].item() - loss_dict_ps["w_crps"].item()) < 1e-6

    print("\ncrps:\n", loss_dict)


def test_spread_skill_ratio():
    # create test input tensors
    batch_size = 128
    num_channels = 3
    height = 32
    width = 64
    loc = torch.randn(batch_size, num_channels, height, width)
    scale = torch.ones(batch_size, num_channels, height, width)
    pred = Normal(loc, scale)
    y = torch.randn(batch_size, num_channels, height, width)

    # call the lat_weighted_spread_skill_ratio function
    transform = None
    vars = ["var1", "var2", "var3"]
    lat = np.random.rand(height)
    clim = None
    log_postfix = "test"
    loss_dict = lat_weighted_spread_skill_ratio(
        pred, y, transform, vars, lat, clim, log_postfix
    )

    # check the shape of the output dictionary
    assert len(loss_dict) == len(vars) + 1  # +1 for "w_spread" key

    # check the shape and type of each loss
    for var in vars:
        loss_key = f"w_spread_{var}_{log_postfix}"
        assert loss_key in loss_dict
        assert isinstance(loss_dict[loss_key], torch.Tensor)

    assert isinstance(loss_dict["w_spread"], np.float32)

    print("\nspread:\n", loss_dict)


def test_categorical_loss():
    # create test input tensors
    batch_size = 128
    num_bins = 50
    num_channels = 3
    height = 32
    width = 64
    pred = torch.randn(batch_size, num_bins, num_channels, height, width)
    y = one_hot(
        torch.randn(batch_size, num_channels, height, width, num_bins).argmax(dim=4),
        num_bins,
    )

    # call the lat_weighted_categorical_loss function
    transform = None
    vars = ["var1", "var2", "var3"]
    lat = np.random.rand(height)
    clim = None
    log_postfix = "test"
    loss_dict = lat_weighted_categorical_loss(
        pred, y, transform, vars, lat, clim, log_postfix
    )

    # check the shape of the output dictionary
    assert len(loss_dict) == len(vars) + 1  # +1 for "w_categorical" key

    # check the shape and type of each loss
    for var in vars:
        loss_key = f"w_categorical_{var}_{log_postfix}"
        assert loss_key in loss_dict
        assert isinstance(loss_dict[loss_key], torch.Tensor)

    assert isinstance(loss_dict["w_categorical"], np.float32)

    print("\ncategorical:\n", loss_dict)

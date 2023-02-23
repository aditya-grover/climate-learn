import torch
import numpy as np
from torch.distributions.normal import Normal
from src.climate_learn.models.modules.utils.metrics import (
    lat_weighted_categorical_loss,
    lat_weighted_spread_skill_ratio,
    lat_weighted_crps_gaussian,
    lat_weighted_nll
)

def test_categorical_loss():
    # create test input tensors
    batch_size = 128
    num_bins = 50
    num_channels = 2
    height = 32
    width = 64
    pred = torch.randn(batch_size, num_bins, num_channels, height, width)
    y = torch.randint(low=0, high=num_bins, size=(batch_size, num_bins, num_channels, height, width))
        
    # call the lat_weighted_categorical_loss function
    transform = None
    vars = ["var1", "var2"]
    lat = np.random.rand(height)
    clim = None
    log_postfix = "test"
    loss_dict = lat_weighted_categorical_loss(pred, y, transform, vars, lat, clim, log_postfix)
        
    # check the shape of the output dictionary
    assert len(loss_dict) == len(vars) + 1 # +1 for "w_categorical" key
        
    # check the shape and type of each loss
    for var in vars:
        loss_key = f"w_categorical_{var}_{log_postfix}"
        assert loss_key in loss_dict
        assert isinstance(loss_dict[loss_key], torch.Tensor)
    
    assert isinstance(loss_dict["w_categorical"], np.float32)
    
    print("\ncategorical:\n", loss_dict)


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
    loss_dict = lat_weighted_spread_skill_ratio(pred, y, transform, vars, lat, clim, log_postfix)

    # check the shape of the output dictionary
    assert len(loss_dict) == len(vars) + 1 # +1 for "w_spread" key
        
    # check the shape and type of each loss
    for var in vars:
        loss_key = f"w_spread_{var}_{log_postfix}"
        assert loss_key in loss_dict
        assert isinstance(loss_dict[loss_key], torch.Tensor)
        
    assert isinstance(loss_dict["w_spread"], np.float32)
    
    print("\nspread:\n", loss_dict)


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
    loss_dict = lat_weighted_crps_gaussian(pred, y, transform, vars, lat, clim, log_postfix)

    # check the shape of the output dictionary
    assert len(loss_dict) == len(vars) + 1 # +1 for "w_crps" key
        
    # check the shape and type of each loss
    for var in vars:
        loss_key = f"w_crps_{var}_{log_postfix}"
        assert loss_key in loss_dict
        assert isinstance(loss_dict[loss_key], torch.Tensor)
        
    assert isinstance(loss_dict["w_crps"], np.float32)
    
    print("\ncrps:\n", loss_dict)


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
    assert len(loss_dict) == len(vars) + 1 # +1 for "w_nll" key

    # get the golden loss to verify
    golden_loss = torch.nn.GaussianNLLLoss(reduction='none')
    loss = golden_loss(loc, y, scale)

    w_lat = np.cos(np.deg2rad(lat))
    w_lat = w_lat / w_lat.mean()  # (H, )
    w_lat = (torch.from_numpy(w_lat).unsqueeze(0).unsqueeze(-1).to(dtype=loss.dtype, device=loss.device))

    golden_loss_dict = {}
    with torch.no_grad():
        for i, var in enumerate(vars):
            golden_loss_dict[f"w_nll_{var}_{log_postfix}"] = torch.mean(loss[:, i] * w_lat)

    golden_loss_dict["w_nll"] = np.mean([golden_loss_dict[k].cpu() for k in golden_loss_dict.keys()])
    print('\ngolden_nll:\n',golden_loss_dict)
    print("\nnll:\n", loss_dict)
        
    # check the shape and type of each loss, and check with golden loss
    for var in vars:
        loss_key = f"w_nll_{var}_{log_postfix}"
        assert loss_key in loss_dict
        assert isinstance(loss_dict[loss_key], torch.Tensor)
        # assert loss_dict[loss_key] == golden_loss_dict[loss_key]
        
    assert isinstance(loss_dict["w_nll"], np.float32)
    # assert loss_dict['w_nll'] == golden_loss_dict['w_nll']
    
    
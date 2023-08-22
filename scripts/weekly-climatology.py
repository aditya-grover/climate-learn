import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import os

def load_test_data(path, var, years=slice('2017', '2018')):
    """
    Load the test dataset. If z return z500, if t return t850.
    Args:
        path: Path to nc files
        var: variable. Geopotential = 'z', Temperature = 't'
        years: slice for time window

    Returns:
        dataset: Concatenated dataset for 2017 and 2018
    """
    ds = xr.open_mfdataset(path, combine='by_coords')[var]
    if var in ['z', 't']:
        if len(ds["level"].dims) > 0:
            # try:
            #     ds = ds.sel(level=500 if var == 'z' else 850) .drop('level')
            # except ValueError:
            ds = ds.drop('level')
        else:
            assert ds["level"].values == 500 if var == 'z' else ds["level"].values == 850
    return ds.sel(time=years)

def compute_weighted_rmse(da_fc, da_true, mean_dims=xr.ALL_DIMS):
    """
    Compute the RMSE with latitude weighting from two xr.DataArrays.

    Args:
        da_fc (xr.DataArray): Forecast. Time coordinate must be validation time.
        da_true (xr.DataArray): Truth.
        mean_dims: dimensions over which to average score
    Returns:
        rmse: Latitude weighted root mean squared error
    """
    error = da_fc - da_true
    weights_lat = np.cos(np.deg2rad(error.lat))
    weights_lat /= weights_lat.mean()
    rmse = np.sqrt(((error)**2 * weights_lat).mean(mean_dims))
    return rmse

def compute_weighted_acc(da_fc, da_true, mean_dims=xr.ALL_DIMS):
    """
    Compute the ACC with latitude weighting from two xr.DataArrays.
    WARNING: Does not work if datasets contain NaNs

    Args:
        da_fc (xr.DataArray): Forecast. Time coordinate must be validation time.
        da_true (xr.DataArray): Truth.
        mean_dims: dimensions over which to average score
    Returns:
        acc: Latitude weighted acc
    """

    clim = da_true.mean('time')
    try:
        t = np.intersect1d(da_fc.time, da_true.time)
        fa = da_fc.sel(time=t) - clim
    except AttributeError:
        t = da_true.time.values
        fa = da_fc - clim
    a = da_true.sel(time=t) - clim

    weights_lat = np.cos(np.deg2rad(da_fc.lat))
    weights_lat /= weights_lat.mean()
    w = weights_lat

    fa_prime = fa - fa.mean()
    a_prime = a - a.mean()

    acc = (
            np.sum(w * fa_prime * a_prime) /
            np.sqrt(
                np.sum(w * fa_prime ** 2) * np.sum(w * a_prime ** 2)
            )
    )
    return acc


data_dir = '/data0/datasets/weatherbench/data/weatherbench/era5/5.625deg'

# Load the entire dataset for the relevant variables
z500 = xr.open_mfdataset(os.path.join(data_dir, 'geopotential_500/*.nc'), combine='by_coords').z.drop('level')
t850 = xr.open_mfdataset(os.path.join(data_dir, 'temperature_850/*.nc'), combine='by_coords').t.drop('level')
t2m = xr.open_mfdataset(os.path.join(data_dir, '2m_temperature/*.nc'), combine='by_coords').t2m
data = xr.merge([z500, t850, t2m])

# Load the training data
train_data = data.sel(time=slice(None, '2016'))
valid_data = data.sel(time=slice('2017', '2018'))
# # Load the validation subset of the data: 2017 and 2018
# z500_valid = load_test_data(os.path.join(data_dir, 'geopotential_500/*.nc'), 'z')
# t850_valid = load_test_data(os.path.join(data_dir, 'temperature_850/*.nc'), 't')
# t2m_valid = load_test_data(os.path.join(data_dir, '2m_temperature/*.nc'), 't2m')
# valid_data = xr.merge([z500_valid, t850_valid, t2m_valid])

def create_weekly_climatology_forecast(ds_train, valid_time):
    ds_train['week'] = ds_train['time.week']
    weekly_averages = ds_train.groupby('week').mean('time')
    valid_time['week'] = valid_time['time.week']
    fc_list = []
    for t in valid_time:
        fc_list.append(weekly_averages.sel(week=t.week))
    return xr.concat(fc_list, dim=valid_time)

weekly_climatology = create_weekly_climatology_forecast(train_data, valid_data.time)

print (compute_weighted_rmse(weekly_climatology.z, valid_data.z).to_numpy())
print (compute_weighted_rmse(weekly_climatology.t, valid_data.t).to_numpy())
print (compute_weighted_rmse(weekly_climatology.t2m, valid_data.t2m).to_numpy())
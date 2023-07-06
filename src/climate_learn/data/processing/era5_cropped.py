# Standard library
from argparse import ArgumentParser
import glob
import os

# Third party
import numpy as np
import xarray as xr


parser = ArgumentParser(
    description="Crops ERA5 data for ERA5 to PRISM downscaling experiments."
)
parser.add_argument(
    "source", help="The local directory containing raw ERA5 2.8125 degree files."
)
parser.add_argument(
    "destination", help="The destination directory for the processed files."
)
parser.add_argument(
    "--train_end", default=2015, type=int, help="The last year of training data."
)
parser.add_argument(
    "--val_end", default=2016, type=int, help="The last year of validation data."
)
parser.add_argument(
    "--test_end", default=2018, type=int, help="The last year of testing data."
)
args = parser.parse_args()

# Concatenate all 2m temperature xarray files
filelist = glob.glob(os.path.join(args.source, "2m_temperature", "*.nc"))
filelist = sorted(filelist)
xarr = None
for fi in filelist:
    if xarr is None:
        xarr = xr.open_dataset(fi)
    else:
        xarr = xr.concat((xarr, xr.open_dataset(fi)), dim="time")
lats = xarr.lat.data
lons = xarr.lon.data

# PRISM spatial bounds
bottom = 24.10
top = 49.94
left = 234.98
right = 293.48

# Get train data
prism_start_date = "1981-01-01"
train_data = xarr.sel(
    {
        "time": slice(prism_start_date, f"{args.train_end}-12-31"),
        "lat": slice(bottom, top),
        "lon": slice(left, right),
    }
)
train_data = train_data.resample(time="1D").max(dim="time")
train_mean = train_data.mean(dim="time")["t2m"].data
train_std = train_data.std(dim="time")["t2m"].data
train_narr = train_data["t2m"].data
with open(os.path.join(args.destination, "train.npz"), "wb") as f:
    np.savez(f, data=train_narr, mean=train_mean, std=train_std)

# Get validation data
val_data = xarr.sel(
    {
        "time": slice(f"{args.train_end+1}-01-01", f"{args.val_end}-12-31"),
        "lat": slice(bottom, top),
        "lon": slice(left, right),
    }
)
val_data = val_data.resample(time="1D").max(dim="time")
val_mean = val_data.mean(dim="time")["t2m"].data
val_std = val_data.std(dim="time")["t2m"].data
val_narr = val_data["t2m"].data
with open(os.path.join(args.destination, "val.npz"), "wb") as f:
    np.savez(f, data=val_narr, mean=val_mean, std=val_std)

# Get test data
test_data = xarr.sel(
    {
        "time": slice(f"{args.val_end+1}-01-01", f"{args.test_end}-12-31"),
        "lat": slice(bottom, top),
        "lon": slice(left, right),
    }
)
test_data = test_data.resample(time="1D").max(dim="time")
test_mean = test_data.mean(dim="time")["t2m"].data
test_std = test_data.std(dim="time")["t2m"].data
test_narr = test_data["t2m"].data
with open(os.path.join(args.destination, "test.npz"), "wb") as f:
    np.savez(f, data=test_narr, mean=test_mean, std=test_std)

# Save latitude and longitude
cropped_lats = train_data.lat.data
cropped_lons = train_data.lon.data
with open(os.path.join(args.destination, "coords.npz"), "wb") as f:
    np.savez(f, lat=cropped_lats, lon=cropped_lons)

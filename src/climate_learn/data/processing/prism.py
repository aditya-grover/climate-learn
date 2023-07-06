# Standard library
from argparse import ArgumentParser
import glob
import os

# Third party
import numpy as np
import rasterio as rio
from tqdm import tqdm
import xesmf as xe


parser = ArgumentParser(description="Processes PRISM data.")
parser.add_argument(
    "source", help="The local directory containing raw PRISM files. See download.py."
)
parser.add_argument(
    "destination", help="The destination directory for the processed files."
)
parser.add_argument(
    "--target_res",
    type=float,
    default=0.75,
    help="The desired target resolution in degrees.",
)
parser.add_argument(
    "--train_end", default="2016", help="The first year of validation data."
)
parser.add_argument("--val_end", default="2017", help="The first year of testing data.")
parser.add_argument("--test_end", default="2018", help="The last year of testing data.")
args = parser.parse_args()

root = args.source
subdirs = sorted(os.listdir(root))

# Build regridder
dataset = rio.open(glob.glob(os.path.join(root, subdirs[0], "*.bil"))[0])
lats = np.empty(dataset.height, dtype=float)
lons = np.empty(dataset.width, dtype=float)
for i in range(dataset.height):
    lats[i] = (dataset.transform * (dataset.width // 2, i))[1]
for i in range(dataset.width):
    lons[i] = (dataset.transform * (i, dataset.height // 2))[0] % 360

target_res = args.target_res
scaling_factor = 0.032 / target_res
target_width = round(dataset.width * scaling_factor)
target_height = round(dataset.height * scaling_factor)
grid_in = {"lon": lons, "lat": lats}
grid_out = {
    "lon": np.linspace(lons.min(), lons.max(), target_width),
    "lat": np.linspace(lats.min(), lats.max(), target_height),
}
regridder = xe.Regridder(grid_in, grid_out, "bilinear")

# Get mask
arr = dataset.read(1)
mask = (arr != -9999).astype(int)

# Define function to fix border
masked_arr = np.where(mask, arr, np.nan)
arr_out = regridder(masked_arr)
first_row = np.empty(arr_out.shape[1])
first_row[:] = np.nan


def fix(arr):
    return np.vstack((first_row, arr[1:]))


# Process PRISM data
all_prism_data = []
for sd in tqdm(subdirs):
    dataset = rio.open(glob.glob(os.path.join(root, sd, "*.bil"))[0])
    arr = dataset.read(1)
    masked_arr = np.where(mask, arr, np.nan)
    arr_out = regridder(masked_arr)
    fixed_arr = fix(arr_out)
    all_prism_data.append(fixed_arr)
all_prism_data = np.stack(all_prism_data, 0)

# Build train/val/test splits
train_end, val_end, test_end = None, None, None
for i, sd in enumerate(subdirs):
    if train_end is None and sd.startswith(args.train_end):
        train_end = i
    if val_end is None and sd.startswith(args.val_end):
        val_end = i
    if sd.startswith(args.test_end):
        test_end = i

train = all_prism_data[:train_end]
train_mean = train.mean(axis=0)
train_std = train.std(axis=0)

val = all_prism_data[train_end:val_end]
val_mean = val.mean(axis=0)
val_std = val.std(axis=0)

test = all_prism_data[val_end:test_end]
test_mean = test.mean(axis=0)
test_std = test.std(axis=0)

regridded_mask = np.where(np.isnan(train[0]), 0, 1)

# Save outputs
dest = args.destination
with open(f"{args.destination}/train.npz", "wb") as f:
    np.savez(f, data=train, mean=train_mean, std=train_std)

with open(f"{args.destination}/val.npz", "wb") as f:
    np.savez(f, data=val, mean=val_mean, std=val_std)

with open(f"{args.destination}/test.npz", "wb") as f:
    np.savez(f, data=test, mean=test_mean, std=test_std)

with open(f"{args.destination}/coords.npz", "wb") as f:
    np.savez(f, lat=grid_out["lat"], lon=grid_out["lon"])

with open(f"{args.destination}/mask.npy", "wb") as f:
    np.save(f, regridded_mask)

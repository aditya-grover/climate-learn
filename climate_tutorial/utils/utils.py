import os
import xarray as xr

def load_dataset(dir):
    return xr.open_mfdataset(os.path.join(dir, "*.nc"))

def view(dataset):
    print(dataset.t2m)
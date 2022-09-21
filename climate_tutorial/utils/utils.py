import os
import xarray as xr
from IPython import display

def load_dataset(dir):
    return xr.open_mfdataset(os.path.join(dir, "*.nc"))

def view(dataset):
    display(dataset.t2m)
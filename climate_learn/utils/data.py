import os
import xarray as xr
from IPython.display import display

def load_dataset(dir):
    """
    Loads a dataset from a directory of NetCDF files.

    :param dir: the directory to open
    :type dir: str
    :return: an xarray dataset object
    :rtype: xarray.Dataset
    """
    return xr.open_mfdataset(os.path.join(dir, "*.nc"))

def view(dataset):
    """
    Displays the given dataset in the current IPython notebook.
    
    :param dataset: the dataset to show
    :type dataset: xarray.Dataset
    """
    display(dataset)

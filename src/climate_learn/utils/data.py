# Standard library
import os

# Third party
from IPython.display import display
import xarray as xr


def load_dataset(dir):
    """
    Loads a dataset from a directory of NetCDF files.

    :param dir: The directory to open.
    :type dir: str
    :return: An xarray dataset object.
    :rtype: xarray.Dataset
    """
    return xr.open_mfdataset(os.path.join(dir, "*.nc"))


def view(dataset):
    """
    Displays the given dataset in the current IPython notebook.

    :param dataset: The dataset to show.
    :type dataset: xarray.Dataset
    """
    display(dataset)

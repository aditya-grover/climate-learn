import os
import glob
import xarray as xr
import numpy as np

from tqdm import tqdm
from typing import Callable, Dict, Iterable, Sequence
from climate_learn.data.climate_dataset import ClimateDataset
from climate_learn.data.climate_dataset.args import ERA5Args
from ..constants import (
    NAME_TO_VAR,
    DEFAULT_PRESSURE_LEVELS,
    CONSTANTS,
    SINGLE_LEVEL_VARS,
    PRESSURE_LEVEL_VARS,
)


class ERA5(ClimateDataset):
    args_class: Callable[..., ERA5Args] = ERA5Args

    def __init__(self, data_args: ERA5Args) -> None:
        super().__init__(data_args)
        self.root_dir: str = data_args.root_dir
        self.years: Iterable[int] = data_args.years

    def setup(self) -> None:
        self.constant_names: Sequence[str] = []
        self.data_dict: Dict[str, xr.core.dataarray.DataArray] = self.load_from_nc(
            self.root_dir
        )
        self.set_lat_lon()

    def load_from_nc(self, data_dir: str) -> Dict[str, xr.core.dataarray.DataArray]:
        self.constant_names = [
            name for name in self.variables if NAME_TO_VAR[name] in CONSTANTS
        ]
        self.constants: Dict[str, xr.core.dataarray.DataArray] = {}
        if len(self.constant_names) > 0:
            ps = glob.glob(os.path.join(data_dir, "constants", "*.nc"))
            all_constants = xr.open_mfdataset(ps, combine="by_coords")
            for name in self.constant_names:
                self.constants[name] = all_constants[NAME_TO_VAR[name]]

        non_const_names = [
            name for name in self.variables if name not in self.constant_names
        ]
        data_dict = {}
        for name in non_const_names:
            if name in SINGLE_LEVEL_VARS:
                data_dict[name] = []
            elif name in PRESSURE_LEVEL_VARS:
                for level in DEFAULT_PRESSURE_LEVELS:
                    data_dict[f"{name}_{level}"] = []
            else:
                raise NotImplementedError(
                    f"{name} is not either in single-level or pressure-level dict"
                )

        for year in tqdm(self.years):
            for var in non_const_names:
                dir_var = os.path.join(data_dir, var)
                ps = glob.glob(os.path.join(dir_var, f"*{year}*.nc"))
                xr_data = xr.open_mfdataset(ps, combine="by_coords")
                xr_data = xr_data[NAME_TO_VAR[var]]
                # np_data = xr_data.to_numpy()
                if len(xr_data.shape) == 3:  # 8760, 32, 64
                    xr_data = xr_data.expand_dims(dim="level", axis=1)
                    data_dict[var].append(xr_data)
                else:  # pressure level
                    for level in DEFAULT_PRESSURE_LEVELS:
                        xr_data_level = xr_data.sel(level=[level])
                        data_dict[f"{var}_{level}"].append(xr_data_level)

        data_dict = {k: xr.concat(data_dict[k], dim="time") for k in data_dict.keys()}
        # precipitation and solar radiation miss a few data points in the beginning
        len_min = min([data_dict[k].shape[0] for k in data_dict.keys()])
        data_dict = {k: data_dict[k][-len_min:] for k in data_dict.keys()}

        # remove constants from variables
        self.variables = list(data_dict.keys())
        return data_dict

    def set_lat_lon(self) -> None:
        # lat lon is stored in each of the nc files, just need to load one and extract
        dir_var = os.path.join(self.root_dir, self.variables[0])
        year = self.years[0]
        ps = glob.glob(os.path.join(dir_var, f"*{year}*.nc"))
        xr_data = xr.open_mfdataset(ps, combine="by_coords")
        self.lat: np.ndarray = xr_data["lat"].to_numpy()
        self.lon: np.ndarray = xr_data["lon"].to_numpy()


ERA5Args._data_class = ERA5

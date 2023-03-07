import os
import glob
import xarray as xr
import torch

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
    _args_class: Callable[..., ERA5Args] = ERA5Args

    def __init__(self, data_args: ERA5Args) -> None:
        super().__init__(data_args)
        self.root_dir: str = data_args.root_dir
        self.years: Iterable[int] = data_args.years

    def set_lat_lon(self) -> None:
        # lat lon is stored in each of the nc files, just need to load one and extract
        dir_var = os.path.join(self.root_dir, self.variables[0])
        year = self.years[0]
        ps = glob.glob(os.path.join(dir_var, f"*{year}*.nc"))
        xr_data = xr.open_mfdataset(ps, combine="by_coords")
        self.lat: torch.tensor = torch.tensor(xr_data["lat"].values)
        self.lon: torch.tensor = torch.tensor(xr_data["lon"].values)

    def setup_metadata(self):
        print("Setting up Meta Data")
        self.set_lat_lon()

    def setup_constants(self, data_dir: str) -> None:
        print("Loading Constants Data from Disk")
        self.constant_names: Sequence[str] = [
            name for name in self.variables if NAME_TO_VAR[name] in CONSTANTS
        ]
        self.constants: Dict[str, torch.tensor] = {}
        if len(self.constant_names) > 0:
            ps = glob.glob(os.path.join(data_dir, "constants", "*.nc"))
            all_constants = xr.open_mfdataset(ps, combine="by_coords")
            for name in self.constant_names:
                self.constants[name] = torch.tensor(
                    all_constants[NAME_TO_VAR[name]].values
                )

        # remove constants from variables
        self.variables = [
            name for name in self.variables if name not in self.constant_names
        ]

    def initialize_data_dict(self) -> None:
        self.data_dict: Dict[str, torch.tensor] = {}
        variables_to_update = {}
        for name in self.variables:
            if name in SINGLE_LEVEL_VARS:
                self.data_dict[name] = []
            elif name in PRESSURE_LEVEL_VARS:
                variables_to_add = []
                for level in DEFAULT_PRESSURE_LEVELS:
                    variables_to_add.append(f"{name}_{level}")
                    self.data_dict[f"{name}_{level}"] = []
                variables_to_update[name] = variables_to_add
            else:
                raise NotImplementedError(
                    f"{name} is not either in single-level or pressure-level dict"
                )
        return variables_to_update

    def load_from_nc(self, data_dir: str) -> Dict[str, torch.tensor]:
        print("Loading variables from disk")
        for year in tqdm(self.years):
            for var in self.variables:
                dir_var = os.path.join(data_dir, var)
                ps = glob.glob(os.path.join(dir_var, f"*{year}*.nc"))
                xr_data = xr.open_mfdataset(ps, combine="by_coords")
                xr_data = xr_data[NAME_TO_VAR[var]]
                # np_data = xr_data.to_numpy()
                if len(xr_data.shape) == 3:  # 8760, 32, 64
                    self.data_dict[var].append(xr_data)
                else:  # pressure level
                    for level in DEFAULT_PRESSURE_LEVELS:
                        xr_data_level = (xr_data.sel(level=[level])).squeeze(axis=1)
                        self.data_dict[f"{var}_{level}"].append(xr_data_level)

        print("Concatenating data from different years")
        self.data_dict = {
            k: xr.concat(self.data_dict[k], dim="time") for k in self.data_dict.keys()
        }
        # precipitation and solar radiation miss a few data points in the beginning
        len_min = min([self.data_dict[k].shape[0] for k in self.data_dict.keys()])
        self.data_dict = {
            k: self.data_dict[k][-len_min:] for k in self.data_dict.keys()
        }
        self.time = self.data_dict[self.variables[0]].time.values
        print("Converting data from xarray to torch tensor")
        self.data_dict = {
            k: torch.tensor(self.data_dict[k].values) for k in self.data_dict.keys()
        }
        self.variables = list(self.data_dict.keys())

    def setup_map(self):
        self.load_from_nc(self.root_dir)
        return len(self.data_dict[self.variables[0]])

    def setup_iter(self):
        pass

    def setup(self, style="map") -> None:
        self.setup_metadata()
        self.setup_constants(self.root_dir)
        variables_to_update = self.initialize_data_dict()

        if style == "map":
            return self.setup_map(), variables_to_update
        elif style == "iter":
            return self.setup_iter(), variables_to_update
        else:
            raise ValueError

    def get_item(self, index):  # Dict where each value is a torch tensor shape 32*64
        return {k: self.data_dict[k][index] for k in self.data_dict.keys()}

    def get_iteritem(self):
        pass

    def get_constants_data(self):  # Dict where each value is a torch tensor shape 32*64
        return self.constants

    def get_metadata(self):  # Dict where each value is a ndarray
        return {"lat": self.lat, "lon": self.lon}


ERA5Args._data_class = ERA5

import os
import glob
import xarray as xr
import torch
import random
import copy
import numpy

from tqdm import tqdm
from typing import Callable, Dict, Iterable, Sequence, Tuple
from climate_learn.data.climate_dataset import ClimateDataset
from climate_learn.data.climate_dataset.args import ERA5Args
from ..constants import (
    NAME_TO_VAR,
    DEFAULT_PRESSURE_LEVELS,
    CONSTANTS,
    SINGLE_LEVEL_VARS,
    PRESSURE_LEVEL_VARS,
    NAME_LEVEL_TO_VAR_LEVEL,
)


class ERA5(ClimateDataset):
    _args_class: Callable[..., ERA5Args] = ERA5Args

    def __init__(self, data_args: ERA5Args) -> None:
        super().__init__(data_args)
        self.root_dir: str = data_args.root_dir
        self.years: Iterable[int] = data_args.years

    def get_file_name_from_variable(self, var: str) -> str:
        if var in SINGLE_LEVEL_VARS or var in PRESSURE_LEVEL_VARS:
            return var
        else:
            return "_".join(var.split("_")[:-1])

    def set_lat_lon(self, year: int) -> None:
        # lat lon is stored in each of the nc files, just need to load one and extract
        dir_var = os.path.join(
            self.root_dir, self.get_file_name_from_variable(self.variables[0])
        )
        ps = glob.glob(os.path.join(dir_var, f"*{year}*.nc"))
        xr_data = xr.open_mfdataset(ps, combine="by_coords")
        self.lat: numpy.ndarray = xr_data["lat"].values
        self.lon: numpy.ndarray = xr_data["lon"].values

    def setup_metadata(self, year: int) -> None:
        ## Prevent setup if already called before
        if hasattr(self, "lat") and hasattr(self, "lon"):
            return
        self.set_lat_lon(year)

    def setup_constants(self, data_dir: str) -> None:
        ## Prevent loading constants if already loaded
        if hasattr(self, "constant_names"):
            return
        self.constant_names: Sequence[str] = [
            name
            for name in self.variables
            if NAME_TO_VAR[self.get_file_name_from_variable(name)] in CONSTANTS
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

    def initialize_data_dict(
        self,
    ) -> Tuple[Dict[str, torch.tensor], Dict[str, Sequence[str]]]:
        if hasattr(self, "data_dict"):
            return {k: [] for k in self.data_dict.keys()}, {}
        data_dict: Dict[str, torch.tensor] = {}
        variables_to_update: Dict[str, Sequence[str]] = {}
        for name in self.variables:
            if name in SINGLE_LEVEL_VARS:
                data_dict[name] = []
            ## if variable is an instance of specific multi level vars
            elif name in NAME_LEVEL_TO_VAR_LEVEL:
                data_dict[name] = []
            elif name in PRESSURE_LEVEL_VARS:
                variables_to_add = []
                for level in DEFAULT_PRESSURE_LEVELS:
                    variables_to_add.append(f"{name}_{level}")
                    data_dict[f"{name}_{level}"] = []
                variables_to_update[name] = variables_to_add
            else:
                raise NotImplementedError(
                    f"{name} is not either in single-level or pressure-level dict"
                )
        return data_dict, variables_to_update

    def load_from_nc_by_years(self, data_dir: str, years) -> Dict[str, torch.tensor]:
        data_dict, variables_to_update = self.initialize_data_dict()
        for year in tqdm(years):
            for var in self.variables:
                dir_var = os.path.join(data_dir, self.get_file_name_from_variable(var))
                ps = glob.glob(os.path.join(dir_var, f"*{year}*.nc"))
                xr_data = xr.open_mfdataset(ps, combine="by_coords")
                xr_data = xr_data[NAME_TO_VAR[self.get_file_name_from_variable(var)]]
                # np_data = xr_data.to_numpy()
                if len(xr_data.shape) == 3:  # 8760, 32, 64
                    data_dict[var].append(xr_data)
                else:  # multi level data
                    if var in NAME_LEVEL_TO_VAR_LEVEL:  ## loading only a specific level
                        level = int(var.split("_")[-1])
                        xr_data_level = (xr_data.sel(level=[level])).squeeze(axis=1)
                        data_dict[var].append(xr_data_level)
                    else:  ## loading all levels
                        for level in DEFAULT_PRESSURE_LEVELS:
                            xr_data_level = (xr_data.sel(level=[level])).squeeze(axis=1)
                            data_dict[f"{var}_{level}"].append(xr_data_level)

        data_dict = {k: xr.concat(data_dict[k], dim="time") for k in data_dict.keys()}
        # precipitation and solar radiation miss a few data points in the beginning
        len_min = min([data_dict[k].shape[0] for k in data_dict.keys()])
        data_dict = {k: data_dict[k][-len_min:] for k in data_dict.keys()}
        # using next(iter) isntead of list(data_dict.keys())[0] to get random element
        self.time = data_dict[next(iter(data_dict.keys()))].time.values
        data_dict = {k: torch.from_numpy(data_dict[k].values) for k in data_dict.keys()}
        return data_dict, variables_to_update

    def setup_map(self) -> Tuple[int, Dict[str, Sequence[str]]]:
        self.setup_constants(self.root_dir)
        self.setup_metadata(self.years[0])
        self.data_dict, variables_to_update = self.load_from_nc_by_years(
            self.root_dir, self.years
        )
        return (
            len(self.data_dict[next(iter(self.data_dict.keys()))]),
            variables_to_update,
        )

    def build_years_to_iterate(self, seed: int, drop_last: bool) -> Sequence[int]:
        temp_years = list(copy.deepcopy(self.years))
        random.Random(seed).shuffle(temp_years)
        if drop_last:
            n_years = len(temp_years) // self.world_size
        else:
            n_years = -(len(temp_years) // (-self.world_size))
        years_to_iterate = []
        start_index = self.rank
        for _ in range(n_years):
            years_to_iterate.append(temp_years[start_index])
            start_index = (start_index + self.world_size) % len(temp_years)
        return years_to_iterate

    def setup_shard(self, args: Dict[str, int]) -> Tuple[int, Dict[str, Sequence[str]]]:
        assert "world_size" in args.keys()
        assert "rank" in args.keys()
        assert "seed" in args.keys()
        assert "n_chunks" in args.keys()

        self.world_size: int = args["world_size"]
        self.rank: int = args["rank"]
        self.n_chunks: int = args["n_chunks"]
        if "drop_last" in args.keys():
            drop_last = True
        else:
            drop_last = False

        years_to_iterate: Sequence[int] = self.build_years_to_iterate(
            args["seed"], drop_last
        )
        self.setup_constants(self.root_dir)
        self.setup_metadata(years_to_iterate[0])

        assert len(years_to_iterate) >= self.n_chunks
        self.years_to_iterate: Sequence[int] = years_to_iterate
        self.chunks_iterated_till_now: int = 0
        self.data_dict, variables_to_update = self.initialize_data_dict()
        return -1, variables_to_update

    def load_chunk(self, chunk_id: int) -> int:
        years_to_iterate_this_chunk = self.years_to_iterate[chunk_id :: self.n_chunks]
        self.data_dict, _ = self.load_from_nc_by_years(
            self.root_dir, years_to_iterate_this_chunk
        )
        return len(self.data_dict[next(iter(self.data_dict.keys()))])

    def setup(
        self, style: str = "map", setup_args: Dict = {}
    ) -> Tuple[int, Dict[str, Sequence[str]]]:
        assert style in ["map", "shard"]
        if style == "map":
            return self.setup_map()
        else:
            return self.setup_shard(setup_args)

    def get_item(
        self, index: int
    ) -> Dict[str, torch.tensor]:  # Dict where each value is a torch tensor shape 32*64
        return {k: self.data_dict[k][index] for k in self.data_dict.keys()}

    def get_constants_data(
        self,
    ) -> Dict[str, torch.tensor]:  # Dict where each value is a torch tensor shape 32*64
        return self.constants

    def get_metadata(
        self,
    ) -> Dict[str, numpy.ndarray]:  # Dict where each value is a ndarray
        return {"lat": self.lat, "lon": self.lon}


ERA5Args._data_class = ERA5

from climate_learn.data.climate_dataset.args import ERA5Args, StackedClimateDatasetArgs
from climate_learn.data.task.args import DownscalingArgs, ForecastingArgs
from climate_learn.data.dataset import MapDatasetArgs, ShardDatasetArgs
from climate_learn.data import DataModuleArgs, DataModule
import os
import pytest

DATA_PATH = "/data0/datasets/weatherbench/data/weatherbench/"
GITHUB_ACTIONS = os.environ.get("GITHUB_ACTIONS") == "true"


@pytest.mark.skipif(GITHUB_ACTIONS, reason="only works locally")
class TestModuleInstantiation:
    def test_map_initialization(self):
        climate_dataset_args = ERA5Args(
            root_dir=os.path.join(DATA_PATH, "era5/5.625deg/"),
            variables=[
                "2m_temperature",
                "geopotential",
                "temperature_250",
                "orography",
                "land_sea_mask",
            ],
            years=range(2009, 2015),
            split="train"
        )
        high_res_climate_dataset_args = ERA5Args(
            root_dir=os.path.join(DATA_PATH, "era5/2.8125deg/"),
            variables=[
                "geopotential",
                "temperature_250",
            ],
            years=range(2009, 2015),
            split="train"
        )
        stacked_climate_dataset_args = StackedClimateDatasetArgs([climate_dataset_args, high_res_climate_dataset_args])
        downscaling_args = DownscalingArgs(
            in_vars=[
                "2m_temperature",
                "geopotential",
                "temperature_250",
            ],
            out_vars=[
                "geopotential",
                "temperature_250",
            ],
            constant_names=[
                "orography",
                "land_sea_mask",
            ],
            subsample=6,
        )
        dataset_args = ShardDatasetArgs(stacked_climate_dataset_args, downscaling_args, 2)
        
        DataModule(DataModuleArgs(dataset_args, 2009, 2015, 2017))

    def test_shard_initialization(self):
        climate_dataset_args = ERA5Args(
            root_dir=os.path.join(DATA_PATH, "era5/5.625deg/"),
            variables=[
                "2m_temperature",
                "geopotential",
                "temperature_250",
                "orography",
                "land_sea_mask",
            ],
            years=range(2009, 2015),
            split="train"
        )
        forecasting_args = ForecastingArgs(
            in_vars=[
                "2m_temperature",
                "geopotential",
                "temperature_250",
            ],
            out_vars=[
                "geopotential",
                "temperature_250",
            ],
            constant_names=[
                "orography",
                "land_sea_mask",
            ],
            pred_range=3 * 24,
            subsample=6,
        )
        dataset_args = MapDatasetArgs(climate_dataset_args, forecasting_args)
        
        DataModule(DataModuleArgs(dataset_args, 2014, 2016, 2017))

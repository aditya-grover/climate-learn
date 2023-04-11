from climate_learn.data.climate_dataset.args import ERA5Args, StackedClimateDatasetArgs
from climate_learn.data.task.args import DownscalingArgs, ForecastingArgs
from climate_learn.data.dataset import MapDatasetArgs, ShardDatasetArgs
from climate_learn.data import DataModule
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
            ],
            years=range(2009, 2015),
            constants=[
                "orography",
                "land_sea_mask",
            ],
            split="train",
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
            constants=[
                "orography",
                "land_sea_mask",
            ],
            pred_range=3 * 24,
            subsample=6,
        )
        train_dataset_args = MapDatasetArgs(climate_dataset_args, forecasting_args)

        modified_args_for_val_dataset = {
            "climate_dataset_args": {"years": range(2015, 2017), "split": "val"}
        }
        val_dataset_args = train_dataset_args.create_copy(modified_args_for_val_dataset)

        modified_args_for_test_dataset = {
            "climate_dataset_args": {"years": range(2017, 2019), "split": "test"}
        }
        test_dataset_args = train_dataset_args.create_copy(
            modified_args_for_test_dataset
        )

        DataModule(train_dataset_args, val_dataset_args, test_dataset_args)

    def test_shard_initialization(self):
        climate_dataset_args = ERA5Args(
            root_dir=os.path.join(DATA_PATH, "era5/5.625deg/"),
            variables=[
                "2m_temperature",
                "geopotential",
                "temperature_250",
            ],
            years=range(2009, 2015),
            constants=[
                "orography",
                "land_sea_mask",
            ],
            split="train",
        )
        high_res_climate_dataset_args = ERA5Args(
            root_dir=os.path.join(DATA_PATH, "era5/2.8125deg/"),
            variables=[
                "geopotential",
                "temperature_250",
            ],
            years=range(2009, 2015),
            split="train",
        )
        stacked_climate_dataset_args = StackedClimateDatasetArgs(
            [climate_dataset_args, high_res_climate_dataset_args]
        )
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
            constants=[
                "orography",
                "land_sea_mask",
            ],
            subsample=6,
        )
        train_dataset_args = ShardDatasetArgs(
            stacked_climate_dataset_args, downscaling_args, 2
        )

        modified_args_for_val_dataset = {
            "climate_dataset_args": {
                "child_data_args": [
                    {"years": range(2015, 2017), "split": "val"},
                    {"years": range(2015, 2017), "split": "val"},
                ],
                "split": "val",
            }
        }
        val_dataset_args = train_dataset_args.create_copy(modified_args_for_val_dataset)

        modified_args_for_test_dataset = {
            "climate_dataset_args": {
                "child_data_args": [
                    {"years": range(2017, 2019), "split": "test"},
                    {"years": range(2017, 2019), "split": "test"},
                ],
                "split": "test",
            }
        }
        test_dataset_args = train_dataset_args.create_copy(
            modified_args_for_test_dataset
        )

        DataModule(train_dataset_args, val_dataset_args, test_dataset_args)

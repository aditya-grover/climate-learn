from climate_learn.data import DataModuleArgs, DataModule
from climate_learn.data.climate_dataset.args import ClimateDatasetArgs, ERA5Args
from climate_learn.data.tasks.args import DownscalingArgs
import os
import pytest

DATA_PATH = "/data0/datasets/weatherbench/data/weatherbench/"
GITHUB_ACTIONS = os.environ.get("GITHUB_ACTIONS") == "true"


@pytest.mark.skipif(GITHUB_ACTIONS, reason="only works locally")
class TestModuleInstantiation:
    def test_datamodule_initialization(self):
        temp_data_args = ClimateDatasetArgs(variables=["2m_temperature"], split="train")
        temp_highres_data_args = ClimateDatasetArgs(
            variables=["2m_temperature"], split="train"
        )
        temp_task_args = DownscalingArgs(
            temp_data_args,
            temp_highres_data_args,
            in_vars=["2m_temperature"],
            out_vars=["2m_temperature"],
        )
        DataModuleArgs(temp_task_args, 2014, 2016, 2017)

    def test_initialization(self):
        temp_data_args = ERA5Args(
            root_dir=os.path.join(DATA_PATH, "era5/5.625deg/"),
            variables=["2m_temperature"],
            years=range(2010, 2015),
            split="train",
        )
        temp_highres_data_args = ERA5Args(
            root_dir=os.path.join(DATA_PATH, "era5/2.8125deg/"),
            variables=["2m_temperature"],
            years=range(2010, 2015),
            split="train",
        )
        temp_task_args = DownscalingArgs(
            temp_data_args,
            temp_highres_data_args,
            in_vars=["2m_temperature"],
            out_vars=["2m_temperature"],
        )
        DataModule(DataModuleArgs(temp_task_args, 2014, 2016, 2017))

from climate_learn.data_module import DataModuleArgs, DataModule
from climate_learn.data_module.data.args import DataArgs, ERA5Args
from climate_learn.data_module.tasks.args import DownscalingArgs
import os
import pytest

DATA_PATH = "/data0/datasets/weatherbench/data/weatherbench/"
GITHUB_ACTIONS = os.environ.get("GITHUB_ACTIONS") == "true"

@pytest.mark.skipif(GITHUB_ACTIONS, reason="only works locally")
class TestModuleInstantiation:
    def test_datamodule_initialization(self):
        temp_data_args = DataArgs(variables=["2m_temperature"], split="Train")
        temp_highres_data_args = DataArgs(variables=["2m_temperature"], split="Train")
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
            split="Train",
        )
        temp_highres_data_args = ERA5Args(
            root_dir=os.path.join(DATA_PATH, "era5/2.8125deg/"),
            variables=["2m_temperature"],
            years=range(2010, 2015),
            split="Train",
        )
        temp_task_args = DownscalingArgs(
            temp_data_args,
            temp_highres_data_args,
            in_vars=["2m_temperature"],
            out_vars=["2m_temperature"],
        )
        DataModule(DataModuleArgs(temp_task_args, 2014, 2016, 2017))

from climate_learn.data.climate_dataset.args import ClimateDatasetArgs
from climate_learn.data.task.args import TaskArgs
from climate_learn.data.dataset import MapDatasetArgs, MapDataset
import pytest


@pytest.mark.skip("Shelving map/shard datasets")
class TestMapDatasetInstantiation:
    def test_initialization(self):
        climate_dataset_args = ClimateDatasetArgs(
            variables=["random_variable_1", "random_variable_2"],
            constants=["random_constant"],
            name="my_climate_dataset",
        )
        task_args = TaskArgs(
            in_vars=["my_climate_dataset:random_variable_1"],
            out_vars=["my_climate_dataset:random_variable_2"],
            constants=["my_climate_dataset:random_constant"],
            subsample=3,
        )
        MapDataset(MapDatasetArgs(climate_dataset_args, task_args))

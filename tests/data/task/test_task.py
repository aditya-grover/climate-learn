from climate_learn.data.task import TaskArgs, Task
import pytest

@pytest.mark.skip("Shelving map/shard datasets")
class TestTaskInstantiation:
    def test_initialization(self):
        Task(
            TaskArgs(
                in_vars=["my_climate_dataset:random_variable_1"],
                out_vars=["my_climate_dataset:random_variable_2"],
                constants=["my_climate_dataset:random_constant"],
                subsample=3,
            )
        )

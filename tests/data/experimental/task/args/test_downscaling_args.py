from climate_learn.data.task.args import DownscalingArgs
import pytest


@pytest.mark.skip("Shelving map/shard datasets")
class TestDownscalingArgsInstantiation:
    def test_initialization(self):
        DownscalingArgs(
            in_vars=["my_climate_dataset:random_variable_1"],
            out_vars=["my_climate_dataset:random_variable_2"],
            constants=["my_climate_dataset:random_constant"],
            subsample=3,
        )

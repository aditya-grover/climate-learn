from climate_learn.data.task import DownscalingArgs, Downscaling
import pytest


@pytest.mark.skip("Shelving map/shard datasets")
class TestDownscalingInstantiation:
    def test_initialization(self):
        Downscaling(
            DownscalingArgs(
                in_vars=["my_climate_dataset:random_variable_1"],
                out_vars=["my_climate_dataset:random_variable_2"],
                constants=["my_climate_dataset:random_constant"],
                subsample=3,
            )
        )

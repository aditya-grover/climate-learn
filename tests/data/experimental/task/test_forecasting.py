from climate_learn.data.task import ForecastingArgs, Forecasting
import pytest


@pytest.mark.skip("Shelving map/shard datasets")
class TestForecastingInstantiation:
    def test_initialization(self):
        Forecasting(
            ForecastingArgs(
                in_vars=["my_climate_dataset:random_variable_1"],
                out_vars=["my_climate_dataset:random_variable_2"],
                constants=["my_climate_dataset:random_constant"],
                history=10,
                window=4,
                pred_range=24,
                subsample=3,
            )
        )

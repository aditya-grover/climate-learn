from climate_learn.data.tasks import Forecasting
from climate_learn.data.climate_dataset.args import ClimateDatasetArgs
from climate_learn.data.tasks.args import ForecastingArgs


class TestForecastingInstantiation:
    def test_initialization(self):
        temp_data_args = ClimateDatasetArgs(
            variables=["random_variable_1", "random_variable_2"], split="train"
        )
        Forecasting(
            ForecastingArgs(
                temp_data_args,
                in_vars=["random_variable_1"],
                out_vars=["random_variable_2"],
            )
        )

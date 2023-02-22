from climate_learn.data_module.tasks import Forecasting
from climate_learn.data_module.data.args import DataArgs
from climate_learn.data_module.tasks.args import ForecastingArgs


class TestForecastingInstantiation:
    def test_initialization(self):
        temp_data_args = DataArgs(
            variables=["random_variable_1", "random_variable_2"], split="train"
        )
        Forecasting(
            ForecastingArgs(
                temp_data_args,
                in_vars=["random_variable_1"],
                out_vars=["random_variable_2"],
            )
        )

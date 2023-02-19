from climate_learn.data_module.tasks.args import ForecastingArgs
from climate_learn.data_module.data.args import DataArgs


class TestForecastingArgsInstantiation:
    def test_initialization(self):
        temp_data_args = DataArgs(
            variables=["random_variable_1", "random_variable_2"], split="Train"
        )
        ForecastingArgs(
            temp_data_args,
            in_vars=["random_variable_1"],
            out_vars=["random_variable_2"],
        )

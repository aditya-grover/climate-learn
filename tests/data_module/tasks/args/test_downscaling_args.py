from climate_learn.data_module.tasks.args import DownscalingArgs
from climate_learn.data_module.data.args import DataArgs


class TestDownscalingArgsInstantiation:
    def test_initialization(self):
        temp_data_args = DataArgs(variables=["random_variable_1"], split="train")
        temp_highres_data_args = DataArgs(
            variables=["random_variable_2"], split="train"
        )
        DownscalingArgs(
            temp_data_args,
            temp_highres_data_args,
            in_vars=["random_variable_1"],
            out_vars=["random_variable_2"],
        )

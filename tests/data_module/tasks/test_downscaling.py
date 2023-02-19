from climate_learn.data_module.tasks import Downscaling
from climate_learn.data_module.data.args import DataArgs
from climate_learn.data_module.tasks.args import DownscalingArgs


class TestDownscalingInstantiation:
    def test_initialization(self):
        temp_data_args = DataArgs(variables=["random_variable_1"], split="Train")
        temp_highres_data_args = DataArgs(
            variables=["random_variable_2"], split="Train"
        )
        Downscaling(
            DownscalingArgs(
                temp_data_args,
                temp_highres_data_args,
                in_vars=["random_variable_1"],
                out_vars=["random_variable_2"],
            )
        )

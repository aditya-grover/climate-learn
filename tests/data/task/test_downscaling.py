from climate_learn.data.task import DownscalingArgs, Downscaling


class TestDownscalingInstantiation:
    def test_initialization(self):
        Downscaling(
            DownscalingArgs(
                in_vars=["random_variable_1"],
                out_vars=["random_variable_2"],
                constant_names=["random_constant"],
                subsample=3,
            )
        )

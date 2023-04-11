from climate_learn.data.task.args import DownscalingArgs


class TestDownscalingArgsInstantiation:
    def test_initialization(self):
        DownscalingArgs(
            in_vars=["random_variable_1"],
            out_vars=["random_variable_2"],
            constants=["random_constant"],
            subsample=3,
        )

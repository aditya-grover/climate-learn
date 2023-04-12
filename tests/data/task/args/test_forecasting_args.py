from climate_learn.data.task.args import ForecastingArgs


class TestForecastingArgsInstantiation:
    def test_initialization(self):
        ForecastingArgs(
            in_vars=["random_variable_1"],
            out_vars=["random_variable_2"],
            constants=["random_constant"],
            history=10,
            window=4,
            pred_range=24,
            subsample=3,
        )

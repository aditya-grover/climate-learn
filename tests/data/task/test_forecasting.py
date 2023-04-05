from climate_learn.data.task import ForecastingArgs, Forecasting


class TestForecastingInstantiation:
    def test_initialization(self):
        Forecasting(
            ForecastingArgs(
                in_vars=["random_variable_1"],
                out_vars=["random_variable_2"],
                constant_names=["random_constant"],
                history=10,
                window=4,
                pred_range=24,
                subsample=3
            )
        )

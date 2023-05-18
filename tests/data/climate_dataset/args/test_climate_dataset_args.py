from climate_learn.data.climate_dataset.args import ClimateDatasetArgs


class TestClimateDatasetArgsInstantiation:
    def test_initialization(self):
        ClimateDatasetArgs(
            variables=["random_variable_1", "random_variable_2"],
            constants=["random_constant"],
            name="my_climate_dataset",
        )

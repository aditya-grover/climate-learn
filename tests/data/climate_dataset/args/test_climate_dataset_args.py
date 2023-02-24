from climate_learn.data.climate_dataset.args import ClimateDatasetArgs


class TestDataArgsInstantiation:
    def test_initialization(self):
        ClimateDatasetArgs(
            variables=["random_variable_1", "random_variable_2"], split="train"
        )

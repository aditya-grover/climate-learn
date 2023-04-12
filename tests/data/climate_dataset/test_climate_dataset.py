from climate_learn.data.climate_dataset import ClimateDatasetArgs, ClimateDataset


class TestClimateDatasetInstantiation:
    def test_initialization(self):
        ClimateDataset(
            ClimateDatasetArgs(
                variables=["random_variable_1", "random_variable_2"],
                constants=["random_constant"],
                split="train",
            )
        )

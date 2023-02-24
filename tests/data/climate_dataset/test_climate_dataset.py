from climate_learn.data.climate_dataset.args import ClimateDatasetArgs
from climate_learn.data.climate_dataset import ClimateDataset


class TestDataInstantiation:
    def test_initialization(self):
        ClimateDataset(
            ClimateDatasetArgs(
                variables=["random_variable_1", "random_variable_2"], split="train"
            )
        )

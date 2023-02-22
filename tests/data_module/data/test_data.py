from climate_learn.data_module.data.args import DataArgs
from climate_learn.data_module.data import Data


class TestDataInstantiation:
    def test_initialization(self):
        Data(
            DataArgs(
                variables=["random_variable_1", "random_variable_2"], split="train"
            )
        )

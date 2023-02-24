from climate_learn.data.climate_dataset.args import ERA5Args
from climate_learn.data.climate_dataset import ERA5


class TestERA5Instantiation:
    def test_initialization(self):
        ERA5(
            ERA5Args(
                root_dir="my_data_path",
                variables=["random_variable_1", "random_variable_2"],
                years=range(2010, 2015),
                split="train",
            )
        )

from climate_learn.data.climate_dataset.args import ERA5Args


class TestERA5ArgsInstantiation:
    def test_initialization(self):
        ERA5Args(
            root_dir="my_data_path",
            variables=["random_variable_1", "random_variable_2"],
            years=range(2010, 2015),
            split="train",
        )

from climate_learn.data.climate_dataset.args import ERA5Args


class TestERA5ArgsInstantiation:
    def test_initialization(self):
        ERA5Args(
            root_dir="my_data_path",
            variables=["geopotential", "2m_temperature"],
            years=range(2010, 2015),
            constants=["random_constant"],
            split="train",
        )

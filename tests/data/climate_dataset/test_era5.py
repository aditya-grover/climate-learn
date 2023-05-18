from climate_learn.data.climate_dataset import ERA5Args, ERA5


class TestERA5Instantiation:
    def test_initialization(self):
        ERA5(
            ERA5Args(
                root_dir="my_data_path",
                variables=["geopotential", "2m_temperature"],
                years=range(2010, 2015),
                constants=["land_sea_mask", "orography"],
                name="era5",
            )
        )

from climate_learn.data.modules import ERA5


class TestERA5DatasetInstantiation:
    def test_era5(self):
        ERA5(
            root_dir="/data0/datasets/weatherbench/data/weatherbench/era5/5.625deg/",
            root_highres_dir=None,
            variables=[
                "2m_temperature",
                "10m_u_component_of_wind",
                "10m_v_component_of_wind",
            ],
            years=range(2000, 2014),
        )

    def test_era5_with_highres(self):
        ERA5(
            root_dir="/data0/datasets/weatherbench/data/weatherbench/era5/5.625deg/",
            root_highres_dir="/data0/datasets/weatherbench/data/weatherbench/era5/2.8125deg/",
            variables=[
                "2m_temperature",
                "10m_u_component_of_wind",
                "10m_v_component_of_wind",
            ],
            years=range(2000, 2014),
        )

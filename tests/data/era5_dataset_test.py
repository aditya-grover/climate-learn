import os
from climate_learn.data_module.data import ERA5
import pytest

DATA_PATH = "/data0/datasets/weatherbench/data/weatherbench/"
GITHUB_ACTIONS = os.environ.get("GITHUB_ACTIONS") == "true"


@pytest.mark.skipif(GITHUB_ACTIONS, reason="only works locally")
class TestERA5DatasetInstantiation:
    def test_era5(self):
        ERA5(
            root_dir=os.path.join(DATA_PATH, "era5/5.625deg/"),
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
            root_dir=os.path.join(DATA_PATH, "era5/5.625deg/"),
            root_highres_dir=os.path.join(DATA_PATH, "era5/2.8125deg/"),
            variables=[
                "2m_temperature",
                "10m_u_component_of_wind",
                "10m_v_component_of_wind",
            ],
            years=range(2000, 2014),
        )

import os
from climate_learn.data.modules import ERA5Forecasting
from climate_learn.utils.datetime import Hours
import pytest

DATA_PATH = "/data0/datasets/weatherbench/data/weatherbench/"
GITHUB_ACTIONS = os.environ.get("GITHUB_ACTIONS") == "true"


@pytest.mark.skipif(GITHUB_ACTIONS, reason="only works locally")
class TestERA5ForecastingDatasetInstantiation:
    def test_era5_forecasting(self):
        ERA5Forecasting(
            root_dir=os.path.join(DATA_PATH, "era5/5.625deg/"),
            root_highres_dir=None,
            in_vars=["2m_temperature", "10m_u_component_of_wind"],
            out_vars=["2m_temperature", "10m_v_component_of_wind"],
            history=1,
            window=6,
            pred_range=(Hours(6)).hours(),
            years=range(2010, 2014),
        )

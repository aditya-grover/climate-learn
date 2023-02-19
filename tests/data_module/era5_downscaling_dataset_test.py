# import os
# from climate_learn.data_module.data import ERA5Downscaling
# from climate_learn.utils.datetime import Hours
# import pytest

# DATA_PATH = "/data0/datasets/weatherbench/data/weatherbench/"
# GITHUB_ACTIONS = os.environ.get("GITHUB_ACTIONS") == "true"


# @pytest.mark.skipif(GITHUB_ACTIONS, reason="only works locally")
# class TestERA5DownscalingDatasetInstantiation:
#     def test_era5_downscaling(self):
#         ERA5Downscaling(
#             root_dir=os.path.join(DATA_PATH, "era5/5.625deg/"),
#             root_highres_dir=os.path.join(DATA_PATH, "era5/2.8125deg/"),
#             in_vars=["2m_temperature", "10m_u_component_of_wind"],
#             out_vars=["2m_temperature", "10m_v_component_of_wind"],
#             history=1,
#             window=6,
#             pred_range=(Hours(6)).hours(),
#             years=range(2010, 2014),
#         )

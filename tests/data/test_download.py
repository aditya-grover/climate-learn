import climate_learn as cl
import pytest


# The following tests should work as examples, but they are skipped since they
# take a long time to run.


@pytest.mark.skip()
def test_download_prism_tmax(tmp_path):
    dst = tmp_path / "prism"
    num_days_in_2019 = 365
    cl.data.download_prism(dst, variable="tmax", years=(2019, 2020))
    num_subdirs = len(list(dst.glob("*")))
    assert num_subdirs == num_days_in_2019


@pytest.mark.skip()
def test_download_weatherbench_era5_constants(tmp_path):
    dataset = "era5"
    variable = "constants"
    res = 5.625
    dst = tmp_path / "weatherbench" / dataset
    cl.data.download_weatherbench(dst, dataset, variable, res)
    expected_output_file = dst / f"{variable}_{res}deg.nc"
    assert expected_output_file.exists()


@pytest.mark.skip()
def test_download_weatherbench_era5_t2m(tmp_path):
    dataset = "era5"
    variable = "2m_temperature"
    res = 5.625
    dst = tmp_path / "weatherbench" / dataset / variable
    cl.data.download_weatherbench(dst, dataset, variable, res)
    expected_num_years = 40
    assert len(list(dst.iterdir())) == expected_num_years

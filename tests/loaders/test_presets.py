# Local application
import climate_learn as cl
from .utils import MockDataModule

# Third party
import torch
import pytest

FORECASTING_PRESETS = [
    "climatology",
    "persistence",
    "linear-regression",
    "rasp-theurey-2020",
]

DOWNSCALING_PRESETS = [
    "bilinear-interpolation",
    "nearest-interpolation",
]


@pytest.mark.parametrize("preset", FORECASTING_PRESETS)
def test_known_forecasting_presets(preset):
    mock_dm = MockDataModule(32, 3, 2, 2, 32, 64)
    mock_dm.setup()
    model, optimizer, lr_scheduler = cl.load_architecture(
        "forecasting", mock_dm, preset=preset
    )
    if preset == FORECASTING_PRESETS[0]:
        assert isinstance(model, cl.models.hub.Climatology)
        assert optimizer is None
        assert lr_scheduler is None
    elif preset == FORECASTING_PRESETS[1]:
        assert isinstance(model, cl.models.hub.Persistence)
        assert optimizer is None
        assert lr_scheduler is None
    elif preset == FORECASTING_PRESETS[2]:
        assert isinstance(model, cl.models.hub.LinearRegression)
        assert isinstance(optimizer, torch.optim.SGD)
        assert lr_scheduler is None
    elif preset == FORECASTING_PRESETS[3]:
        assert isinstance(model, cl.models.hub.ResNet)
        assert isinstance(optimizer, torch.optim.Adam)
        assert lr_scheduler is None


illegal_persistence_dms = [
    # len(out_vars) > len(in_vars)
    MockDataModule(32, 3, 2, 4, 32, 64),
    # not out_vars.issubset(in_vars)
    MockDataModule(32, 3, 2, 2, 32, 64, ["a", "b"], ["c", "d"]),
]


@pytest.mark.parametrize("mock_dm", illegal_persistence_dms)
def test_illegal_persistence(mock_dm):
    mock_dm.setup()
    with pytest.raises(RuntimeError) as exc_info:
        cl.load_architecture("forecasting", mock_dm, preset="persistence")
    assert str(exc_info.value) == (
        "Persistence requires the output variables to be a subset of the input"
        " variables."
    )


@pytest.mark.parametrize("preset", DOWNSCALING_PRESETS)
def test_known_downscaling_presets(preset):
    mock_dm = MockDataModule(32, 0, 3, 3, 32, 64)
    mock_dm.setup()
    model, optimizer, lr_scheduler = cl.load_architecture(
        "downscaling", mock_dm, preset=preset
    )
    if preset in DOWNSCALING_PRESETS[:3]:
        assert isinstance(model, cl.models.hub.Interpolation)
        assert optimizer is None
        assert lr_scheduler is None

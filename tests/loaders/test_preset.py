# Standard library
from dataclasses import dataclass
from itertools import repeat
from typing import List, Optional, Tuple

# Local application
import climate_learn as cl

# Third party
import torch
import pytest

FORECASTING_PRESETS = [
    "climatology",
    "persistence",
    "linear-regression",
    "rasp-theurey-2020"
]

DOWNSCALING_PRESETS = [
    "linear-interpolation",
    "bilinear-interpolation",
    "nearest-interpolation"
]

@dataclass
class MockTensor:
    shape: Tuple[int, ...]

class MockDataModule:
    def __init__(
        self,
        num_batches: int,
        history: int,
        num_in_vars: int,
        num_out_vars: int,
        height: int,
        width: int,
        in_vars: Optional[List[str]] = None,
        out_vars: Optional[List[str]] = None
    ):
        if in_vars is None:
            in_vars = [f"var{i}" for i in range(num_in_vars)]
        if out_vars is None:
            out_vars = [f"var{i}" for i in range(num_out_vars)]
        if history == 0:
            x = MockTensor((num_batches, num_in_vars, height, width))
        else:
            x = MockTensor((num_batches, history, num_in_vars, height, width))
        y = MockTensor((num_batches, num_out_vars, height, width))
        batch = (x, y, in_vars, out_vars)
        self.batches = repeat(batch)

    def train_dataloader(self, *args, **kwargs):
        return self.batches

    def get_climatology(self, *args, **kwargs):
        return {"a": torch.zeros(2, 2), "b": torch.zeros(2, 2)}


@pytest.mark.parametrize("preset", FORECASTING_PRESETS)
def test_known_forecasting_presets(preset):
    mock_dm = MockDataModule(32, 3, 2, 2, 32, 64)
    model, optimizer, lr_scheduler = cl.load_preset(
        "forecasting",
        mock_dm,
        preset=preset
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
    MockDataModule(32, 3, 2, 2, 32, 64, ["a", "b"], ["c", "d"])
]
@pytest.mark.parametrize("mock_dm", illegal_persistence_dms)
def test_illegal_persistence(mock_dm):
    with pytest.raises(RuntimeError) as exc_info:
        cl.load_preset("forecasting", mock_dm, preset="persistence")
    assert str(exc_info.value) == (
        "Persistence requires the output variables to be a subset of the input"
        " variables."
    )


@pytest.mark.parametrize("preset", DOWNSCALING_PRESETS)
def test_known_downscaling_presets(preset):
    mock_dm = MockDataModule(32, 0, 3, 3, 32, 64)
    model, optimizer, lr_scheduler = cl.load_preset(
        "downscaling",
        mock_dm,
        preset=preset
    )
    if preset in DOWNSCALING_PRESETS[:3]:
        assert isinstance(model, cl.models.hub.Interpolation)
        assert optimizer is None
        assert lr_scheduler is None
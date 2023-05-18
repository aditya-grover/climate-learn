# Local application
import climate_learn as cl
from .utils import MockDataModule

# Third party
import pytest


def test_dm_not_setup():
    mock_dm = MockDataModule(32, 0, 3, 3, 32, 64)
    with pytest.raises(RuntimeError) as exc_info:
        cl.load_model_module("foobar", mock_dm)
    assert str(exc_info.value) == "Data module has not been set up yet."

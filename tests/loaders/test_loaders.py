import climate_learn as cl
import pytest


class TestModelLoader:
    """Test high-level stuff that can't be tested in the subroutines."""
    def test_empty_preset_and_model(self):
        # Try loading no preset and no model
        with pytest.raises(RuntimeError) as exec_info:
            cl.load_model_module(None, None)
        assert exec_info.value.args[0] == "Please specify 'preset' or 'model'"
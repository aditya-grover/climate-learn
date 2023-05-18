# Local application
import climate_learn as cl

# Third party
import pytest


class MockNoParamModel:
    optimizer = None


def test_trainer_no_param_model():
    """Check that Trainer raises an error if given a model with no params."""
    model = MockNoParamModel()
    trainer = cl.Trainer()
    with pytest.raises(RuntimeError) as exc_info:
        trainer.fit(model)
    assert str(exc_info.value) == (
        "Model module has no optimizer - maybe it has no parameters?"
    )

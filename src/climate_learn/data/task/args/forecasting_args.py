# Standard library
from __future__ import annotations
from typing import Any, Callable, Dict, Sequence, TYPE_CHECKING, Union

# Local application
from .task_args import TaskArgs

if TYPE_CHECKING:
    from ..forecasting import Forecasting


class ForecastingArgs(TaskArgs):
    _task_class: Union[Callable[..., Forecasting], str] = "Forecasting"

    def __init__(
        self,
        in_vars: Sequence[str],
        out_vars: Sequence[str],
        constants: Sequence[str] = [],
        history: int = 1,
        window: int = 6,
        pred_range: int = 6,
        subsample: int = 1,
    ) -> None:
        super().__init__(in_vars, out_vars, constants, subsample)
        self.history: int = history
        self.window: int = window
        self.pred_range: int = pred_range
        ForecastingArgs.check_validity(self)

    def create_copy(self, args: Dict[str, Any] = {}) -> ForecastingArgs:
        new_instance: ForecastingArgs = super().create_copy(args)
        ForecastingArgs.check_validity(new_instance)
        return new_instance

    def check_validity(self) -> None:
        super().check_validity()
        if self.history < 0:
            raise RuntimeError(
                f"History should be a non-negative integer. "
                f"Currently set to {self.history}."
            )
        if self.window < 0:
            raise RuntimeError(
                f"Window should be a non-negative integer. "
                f"Currently set to {self.window}."
            )
        if self.pred_range < 0:
            raise RuntimeError(
                f"Prediction range should be a non-negative integer. "
                f"Currently set to {self.pred_range}."
            )

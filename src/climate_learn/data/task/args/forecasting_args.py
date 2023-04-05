from __future__ import annotations
from typing import Callable, Sequence, TYPE_CHECKING, Union
from climate_learn.data.task.args import TaskArgs

if TYPE_CHECKING:
    from climate_learn.data.task import Forecasting


class ForecastingArgs(TaskArgs):
    _task_class: Union[Callable[..., Forecasting], str] = "Forecasting"

    def __init__(
        self,
        in_vars: Sequence[str],
        out_vars: Sequence[str],
        constant_names: Sequence[str] = [],
        history: int = 1,
        window: int = 6,
        pred_range: int = 6,
        subsample: int = 1,
    ) -> None:
        super().__init__(in_vars, out_vars, constant_names, subsample)
        self.history: int = history
        self.window: int = window
        self.pred_range: int = pred_range

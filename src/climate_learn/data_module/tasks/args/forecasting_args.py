from typing import Any, Callable, Sequence, Union
from climate_learn.data_module.data.args import DataArgs
from climate_learn.data_module.tasks.args import TaskArgs


class ForecastingArgs(TaskArgs):
    _task_class: Union[Callable[..., Any], str] = "Forecasting"

    def __init__(
        self,
        dataset_args: DataArgs,
        in_vars: Sequence[str],
        out_vars: Sequence[str],
        constant_names: Sequence[str] = [],
        history: int = 1,
        window: int = 6,
        pred_range: int = 6,
        subsample: int = 1,
        split: str = "train",
    ) -> None:
        super().__init__(
            dataset_args, in_vars, out_vars, constant_names, subsample, split
        )
        self.history: int = history
        self.window: int = window
        self.pred_range: int = pred_range

    def setup(self, data_module_args: Any) -> None:  # TODO add stronger typecheck
        super().setup(data_module_args)

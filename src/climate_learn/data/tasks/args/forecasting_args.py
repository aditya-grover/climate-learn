from __future__ import annotations
from typing import Callable, Sequence, TYPE_CHECKING, Union
from climate_learn.data.climate_dataset.args import ClimateDatasetArgs
from climate_learn.data.tasks.args import TaskArgs

if TYPE_CHECKING:
    from climate_learn.data.tasks import Forecasting
    from climate_learn.data.module import DataModuleArgs


class ForecastingArgs(TaskArgs):
    _task_class: Union[Callable[..., Forecasting], str] = "Forecasting"

    def __init__(
        self,
        dataset_args: ClimateDatasetArgs,
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

    def setup(self, data_module_args: DataModuleArgs) -> None:
        super().setup(data_module_args)

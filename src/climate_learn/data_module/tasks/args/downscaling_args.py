from __future__ import annotations
from typing import Any, Callable, Sequence, TYPE_CHECKING, Union
from climate_learn.data_module.data.args import DataArgs
from climate_learn.data_module.tasks.args import TaskArgs

if TYPE_CHECKING:
    from climate_learn.data_module.module import DataModuleArgs


class DownscalingArgs(TaskArgs):
    _task_class: Union[Callable[..., Any], str] = "Downscaling"

    def __init__(
        self,
        dataset_args: DataArgs,
        highres_dataset_args: DataArgs,
        in_vars: Sequence[str],
        out_vars: Sequence[str],
        constant_names: Sequence[str] = [],
        subsample: int = 1,
        split: str = "train",
    ) -> None:
        super().__init__(
            dataset_args, in_vars, out_vars, constant_names, subsample, split
        )
        self.highres_dataset_args: DataArgs = highres_dataset_args

    def setup(self, data_module_args: DataModuleArgs) -> None:
        super().setup(data_module_args)
        self.highres_dataset_args.split = self.split
        self.highres_dataset_args.setup(data_module_args)

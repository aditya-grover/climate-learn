from __future__ import annotations
from abc import ABC
from typing import Callable, Sequence, TYPE_CHECKING, Union
from climate_learn.data.climate_dataset.args import ClimateDatasetArgs

if TYPE_CHECKING:
    from climate_learn.data.tasks import Task
    from climate_learn.data.module import DataModuleArgs


class TaskArgs(ABC):
    _task_class: Union[Callable[..., Task], str] = "Task"

    def __init__(
        self,
        dataset_args: ClimateDatasetArgs,
        in_vars: Sequence[str],
        out_vars: Sequence[str],
        constant_names: Sequence[str] = [],
        subsample: int = 1,
        split: str = "train",
    ) -> None:
        self.dataset_args: ClimateDatasetArgs = dataset_args
        self.in_vars: Sequence[str] = in_vars
        self.out_vars: Sequence[str] = out_vars
        self.constant_names: Sequence[str] = constant_names
        self.subsample: int = subsample
        self.split: str = split

    def setup(self, data_module_args: DataModuleArgs) -> None:
        assert self.split in ["train", "val", "test"]
        self.dataset_args.split = self.split
        self.dataset_args.setup(data_module_args)

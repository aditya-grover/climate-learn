# Standard library
from __future__ import annotations
from abc import ABC
from typing import Callable, TYPE_CHECKING, Union

# Local application
from climate_learn.data.climate_dataset.args import ClimateDatasetArgs
from climate_learn.data.task.args import TaskArgs

if TYPE_CHECKING:
    from climate_learn.data.dataset import MapDataset
    from climate_learn.data.module import DataModuleArgs


class MapDatasetArgs(ABC):
    _data_class: Union[Callable[..., MapDataset], str] = "MapDataset"

    def __init__(
        self, climate_dataset_args: ClimateDatasetArgs, task_args: TaskArgs
    ) -> None:
        self.climate_dataset_args: ClimateDatasetArgs = climate_dataset_args
        self.task_args: TaskArgs = task_args

    def setup(self, data_module_args: DataModuleArgs, split: str) -> None:
        self.climate_dataset_args.setup(data_module_args, split)

from __future__ import annotations
from abc import ABC
from typing import Callable, TYPE_CHECKING, Union

from climate_learn.data.climate_dataset.args import ClimateDatasetArgs
from climate_learn.data.task.args import TaskArgs

if TYPE_CHECKING:
    from climate_learn.data.dataset import ShardDataset
    from climate_learn.data.module import DataModuleArgs


class ShardDatasetArgs(ABC):
    _data_class: Union[Callable[..., ShardDataset], str] = "ShardDataset"

    def __init__(
        self,
        climate_dataset_args: ClimateDatasetArgs,
        task_args: TaskArgs,
        n_chunks: int,
    ) -> None:
        self.climate_dataset_args: ClimateDatasetArgs = climate_dataset_args
        self.task_args: TaskArgs = task_args
        self.n_chunks: int = n_chunks

    def setup(self, data_module_args: DataModuleArgs, split: str) -> None:
        self.climate_dataset_args.setup(data_module_args, split)

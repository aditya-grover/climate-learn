# Standard library
from __future__ import annotations
from abc import ABC
import copy
from typing import Any, Callable, Dict, TYPE_CHECKING, Union

# Local application
from ...climate_dataset.args import ClimateDatasetArgs
from ...task.args import TaskArgs

if TYPE_CHECKING:
    from ..shard_dataset import ShardDataset


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

    def create_copy(self, args: Dict[str, Any] = {}) -> ShardDatasetArgs:
        new_instance: ShardDatasetArgs = copy.deepcopy(self)
        for arg in args:
            if arg == "climate_dataset_args":
                new_instance.climate_dataset_args = (
                    new_instance.climate_dataset_args.create_copy(args[arg])
                )
            elif arg == "task_args":
                new_instance.task_args = new_instance.task_args.create_copy(args[arg])
            elif hasattr(new_instance, arg):
                setattr(new_instance, arg, args[arg])
        ShardDatasetArgs.check_validity(new_instance)
        return new_instance

    def check_validity(self) -> None:
        if self.n_chunks <= 0:
            raise RuntimeError(
                f"Number of chunks should be a positive integer. "
                f"Currently set to {self.n_chunks}."
            )

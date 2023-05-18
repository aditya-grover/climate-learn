# Standard library
from __future__ import annotations
import copy
from typing import Any, Callable, Dict, Sequence, TYPE_CHECKING, Union

# Local application
from .climate_dataset_args import ClimateDatasetArgs

if TYPE_CHECKING:
    from ..stacked_climate_dataset import StackedClimateDataset


class StackedClimateDatasetArgs(ClimateDatasetArgs):
    _data_class: Union[
        Callable[..., StackedClimateDataset], str
    ] = "StackedClimateDataset"

    def __init__(
        self,
        data_args: Sequence[ClimateDatasetArgs],
        name: str = "stacked_climate_dataset",
    ) -> None:
        self.child_data_args: Sequence[ClimateDatasetArgs] = data_args
        self.name: str = name
        StackedClimateDatasetArgs.check_validity(self)

    def create_copy(self, args: Dict[str, Any] = {}) -> StackedClimateDatasetArgs:
        new_instance: StackedClimateDatasetArgs = copy.deepcopy(self)
        for arg in args:
            if arg == "child_data_args":
                child_data_args = []
                for index, child_data_arg in enumerate(new_instance.child_data_args):
                    child_data_args.append(child_data_arg.create_copy(args[arg][index]))
                new_instance.child_data_args = child_data_args
            elif hasattr(new_instance, arg):
                setattr(new_instance, arg, args[arg])
        StackedClimateDatasetArgs.check_validity(new_instance)
        return new_instance

    def check_validity(self) -> None:
        if len(self.child_data_args) == 0:
            raise RuntimeError(
                f"StackedClimateDataset requires a sequence of ClimateDatasetArgs. "
                f"You have provided none."
            )
        names: Sequence[str] = [
            child_data_arg.name for child_data_arg in self.child_data_args
        ]
        if len(set(names)) != len(names):
            raise RuntimeError(
                f"StackedClimateDatasetArgs requires all the data_args to have "
                f"unique names."
            )

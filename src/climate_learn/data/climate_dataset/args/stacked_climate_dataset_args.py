# Standard library
from __future__ import annotations
import copy
from typing import Any, Callable, Dict, Sequence, TYPE_CHECKING, Union

# Local application
from climate_learn.data.climate_dataset.args import ClimateDatasetArgs

if TYPE_CHECKING:
    from climate_learn.data.climate_dataset import StackedClimateDataset


class StackedClimateDatasetArgs(ClimateDatasetArgs):
    _data_class: Union[
        Callable[..., StackedClimateDataset], str
    ] = "StackedClimateDataset"

    def __init__(self, data_args: Sequence[ClimateDatasetArgs]) -> None:
        self.child_data_args: Sequence[ClimateDatasetArgs] = data_args
        self.split: str = data_args[0].split
        StackedClimateDatasetArgs.check_validity(self)

    def create_copy(self, args: Dict[str, Any]) -> StackedClimateDatasetArgs:
        new_instance: StackedClimateDatasetArgs = copy.deepcopy(self)
        for arg in args:
            if arg == "child_data_args":
                child_data_args = []
                for index, child_data_arg in enumerate(new_instance.child_data_args):
                    child_data_args.append(child_data_arg.create_copy(args[arg][index]))
                new_instance.child_data_args = child_data_args
                continue
            if hasattr(new_instance, arg):
                setattr(new_instance, arg, args[arg])
        StackedClimateDatasetArgs.check_validity(new_instance)
        return new_instance

    def check_validity(self) -> None:
        if len(self.child_data_args) == 0:
            raise RuntimeError(
                f"StackedClimateDataset requires a sequence of ClimateDatasetArgs. "
                f"You have provided none."
            )
        split: str = self.split
        for data_arg in self.child_data_args:
            if data_arg.split != split:
                raise RuntimeError(
                    f"StackedClimateDatasetArgs requires all the data_args to have "
                    f"same split."
                )

# Standard library
from __future__ import annotations
from abc import ABC
import copy
from typing import Any, Callable, Dict, Sequence, TYPE_CHECKING, Union

# Local application
if TYPE_CHECKING:
    from climate_learn.data.climate_dataset import ClimateDataset


class ClimateDatasetArgs(ABC):
    _data_class: Union[Callable[..., ClimateDataset], str] = "ClimateDataset"

    def __init__(
        self,
        variables: Sequence[str],
        constants: Sequence[str] = [],
        split: str = "train",
    ) -> None:
        self.variables: Sequence[str] = variables
        self.constants: Sequence[str] = constants
        self.split: str = split
        ClimateDatasetArgs.check_validity(self)

    def create_copy(self, args: Dict[str, Any]) -> ClimateDatasetArgs:
        new_instance: ClimateDatasetArgs = copy.deepcopy(self)
        for arg in args:
            if hasattr(new_instance, arg):
                setattr(new_instance, arg, args[arg])
        ClimateDatasetArgs.check_validity(new_instance)
        return new_instance

    def check_validity(self) -> None:
        if self.split not in ["train", "val", "test"]:
            raise RuntimeError(
                f"Split {self.split} is not recognized! "
                f"please choose from ['train', 'val', 'test']."
            )

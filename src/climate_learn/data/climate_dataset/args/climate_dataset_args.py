from __future__ import annotations
from abc import ABC
from typing import Callable, Sequence, TYPE_CHECKING, Union

if TYPE_CHECKING:
    from climate_learn.data.climate_dataset import ClimateDataset
    from climate_learn.data.module import DataModuleArgs


class ClimateDatasetArgs(ABC):
    _data_class: Union[Callable[..., ClimateDataset], str] = "ClimateDataset"

    def __init__(self, variables: Sequence[str], split: str = "train") -> None:
        self.variables: Sequence[str] = variables
        self.split: str = split

    def setup(self, data_module_args: DataModuleArgs) -> None:
        assert self.split in ["train", "val", "test"]

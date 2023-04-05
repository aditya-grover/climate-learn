from __future__ import annotations
from typing import Callable, Sequence, TYPE_CHECKING, Union
from climate_learn.data.climate_dataset.args import ClimateDatasetArgs

if TYPE_CHECKING:
    from climate_learn.data.climate_dataset import StackedClimateDataset
    from climate_learn.data.module import DataModuleArgs


class StackedClimateDatasetArgs(ClimateDatasetArgs):
    _data_class: Union[
        Callable[..., StackedClimateDataset], str
    ] = "StackedClimateDataset"

    def __init__(self, data_args: Sequence[ClimateDatasetArgs]) -> None:
        self.child_data_args: Sequence[ClimateDatasetArgs] = data_args
        assert len(data_args) > 0
        self.split: str = data_args[0].split

    def setup(self, data_module_args: DataModuleArgs, split: str) -> None:
        for data_args in self.child_data_args:
            data_args.setup(data_module_args, split)
        self.split = split

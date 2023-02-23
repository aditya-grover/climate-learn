from abc import ABC
from typing import Callable, Sequence
from climate_learn.data.climate_dataset.args import ClimateDatasetArgs


class ClimateDataset(ABC):
    args_class: Callable[..., ClimateDatasetArgs] = ClimateDatasetArgs

    def __init__(self, data_args: ClimateDatasetArgs) -> None:
        self.variables: Sequence[str] = data_args.variables
        self.split: str = data_args.split

    def setup(self) -> None:
        pass


ClimateDatasetArgs._data_class = ClimateDataset

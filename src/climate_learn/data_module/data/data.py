from abc import ABC
from typing import Callable, Sequence
from climate_learn.data_module.data.args import DataArgs


class Data(ABC):
    args_class: Callable[..., DataArgs] = DataArgs

    def __init__(self, data_args: DataArgs) -> None:
        self.variables: Sequence[str] = data_args.variables
        self.split: str = data_args.split

    def setup(self) -> None:
        pass


DataArgs._data_class = Data

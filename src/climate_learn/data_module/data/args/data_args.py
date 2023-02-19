from __future__ import annotations
from typing import Callable, Sequence, TYPE_CHECKING, Union

if TYPE_CHECKING:
    from climate_learn.data_module.data import Data
    from climate_learn.data_module.module import DataModuleArgs


class DataArgs:
    _data_class: Union[Callable[..., Data], str] = "Data"

    def __init__(self, variables: Sequence[str], split: str = "train") -> None:
        self.variables: Sequence[str] = variables
        self.split: str = split

    def setup(self, data_module_args: DataModuleArgs) -> None:
        pass

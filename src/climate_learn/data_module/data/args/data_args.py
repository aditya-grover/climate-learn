from __future__ import annotations
from typing import Any, Callable, Sequence, TYPE_CHECKING, Union

if TYPE_CHECKING:
    from climate_learn.data_module.module import DataModuleArgs


class DataArgs:
    _data_class: Union[Callable[..., Any], str] = "Data"

    def __init__(self, variables: Sequence[str], split: str = "train") -> None:
        self.variables: Sequence[str] = variables
        self.split: str = split

    def setup(self, data_module_args: DataModuleArgs) -> None:
        pass

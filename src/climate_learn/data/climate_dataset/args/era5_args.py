# Standard library
from __future__ import annotations
from typing import Callable, Iterable, Sequence, TYPE_CHECKING, Union

# Local application
from climate_learn.data.climate_dataset.args import ClimateDatasetArgs

if TYPE_CHECKING:
    from climate_learn.data.climate_dataset import ERA5


class ERA5Args(ClimateDatasetArgs):
    _data_class: Union[Callable[..., ERA5], str] = "ERA5"

    def __init__(
        self,
        root_dir: str,
        variables: Sequence[str],
        years: Iterable[int],
        constants: Sequence[str] = [],
        split: str = "train",
    ) -> None:
        super().__init__(variables, constants, split)
        self.root_dir: str = root_dir
        self.years: Iterable[int] = years

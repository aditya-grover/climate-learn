from __future__ import annotations
from typing import Callable, Iterable, Sequence, TYPE_CHECKING, Union
from climate_learn.data.climate_dataset.args import ClimateDatasetArgs

if TYPE_CHECKING:
    from climate_learn.data.climate_dataset import ERA5
    from climate_learn.data.module import DataModuleArgs


class ERA5Args(ClimateDatasetArgs):
    _data_class: Union[Callable[..., ERA5], str] = "ERA5"

    def __init__(
        self,
        root_dir: str,
        variables: Sequence[str],
        years: Iterable[int],
        split: str = "train",
    ) -> None:
        super().__init__(variables, split)
        self.root_dir: str = root_dir
        self.years: Iterable[int] = years

    def setup(self, data_module_args: DataModuleArgs) -> None:
        super().setup(data_module_args)
        if self.split == "train":
            self.years = range(
                data_module_args.train_start_year, data_module_args.val_start_year
            )
        elif self.split == "val":
            self.years = range(
                data_module_args.val_start_year, data_module_args.test_start_year
            )
        elif self.split == "test":
            self.years = range(
                data_module_args.test_start_year, data_module_args.end_year + 1
            )
        else:
            raise ValueError(" Invalid split")

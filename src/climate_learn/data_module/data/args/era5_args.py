from typing import Any, Callable, Iterable, Sequence, Union
from climate_learn.data_module.data.args import DataArgs


class ERA5Args(DataArgs):
    _data_class: Union[Callable[..., Any], str] = "ERA5"

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

    def setup(self, data_module_args: Any) -> None:  # TODO add stronger typecheck
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

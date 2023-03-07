from abc import ABC
from typing import Callable, Sequence
from climate_learn.data.climate_dataset.args import ClimateDatasetArgs


class ClimateDataset(ABC):
    _args_class: Callable[..., ClimateDatasetArgs] = ClimateDatasetArgs

    def __init__(self, data_args: ClimateDatasetArgs) -> None:
        self.variables: Sequence[str] = data_args.variables
        self.split: str = data_args.split

    def setup(self, style="map") -> None:
        self.setup_metadata()
        if style == "map":
            return self.setup_map(), {}
        elif style == "iter":
            return self.setup_iter(), {}
        else:
            raise ValueError

    def setup_metadata(self) -> None:
        pass

    def setup_map(self) -> None:
        return None

    def setup_iter(self) -> None:
        return None

    def get_item(self, index):
        pass

    def get_iteritem(self):
        pass

    def get_constants_data(self):
        pass

    def get_metadata(self):
        pass


ClimateDatasetArgs._data_class = ClimateDataset

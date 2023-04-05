from abc import ABC
from typing import Callable, Sequence
from climate_learn.data.climate_dataset.args import ClimateDatasetArgs


class ClimateDataset(ABC):
    _args_class: Callable[..., ClimateDatasetArgs] = ClimateDatasetArgs

    def __init__(self, data_args: ClimateDatasetArgs) -> None:
        self.variables: Sequence[str] = data_args.variables
        self.split: str = data_args.split

    def setup(self, style: str = "map", setup_args: dict = {}) -> None:
        assert style in ["map", "shard"]
        if style == "map":
            return self.setup_map(), {}
        elif style == "shard":
            return self.setup_shard(setup_args), {}
        else:
            raise ValueError

    def setup_metadata(self) -> None:
        pass

    def setup_constants(self) -> None:
        pass

    def setup_map(self) -> None:
        self.setup_constants()
        self.setup_metadata()
        return None

    def setup_shard(self, setup_args: dict = {}) -> None:
        self.setup_constants()
        self.setup_metadata()
        return None

    def load_chunk(self, chunk_id: int) -> None:
        return None

    def get_item(self, index: int):
        pass

    def get_constants_data(self):
        pass

    def get_metadata(self):
        pass


ClimateDatasetArgs._data_class = ClimateDataset

# Standard library
from abc import ABC
from typing import Any, Callable, Dict, Sequence, Tuple, Union

# Third party
import numpy as np
import torch

# Local application
from .args import ClimateDatasetArgs


class ClimateDataset(ABC):
    _args_class: Callable[..., ClimateDatasetArgs] = ClimateDatasetArgs

    def __init__(self, data_args: ClimateDatasetArgs) -> None:
        self.variables: Sequence[str] = data_args.variables
        self.constants: Sequence[str] = data_args.constants
        self.name: str = data_args.name

    def setup_constants(self) -> None:
        raise NotImplementedError

    def setup_metadata(self) -> None:
        raise NotImplementedError

    def setup_map(self) -> Tuple[int, Any]:
        self.setup_constants()
        self.setup_metadata()
        return -1, {}

    def setup_shard(self, setup_args: dict = {}) -> Tuple[int, Any]:
        self.setup_constants()
        self.setup_metadata()
        return -1, {}

    def setup(
        self, style: str = "map", setup_args: Dict[str, Any] = {}
    ) -> Tuple[int, Any]:
        supported_styles: Sequence[str] = ["map", "shard"]
        if style == "map":
            length, var_to_update = self.setup_map()
        elif style == "shard":
            length, var_to_update = self.setup_shard(setup_args)
        else:
            raise RuntimeError(
                f"Please choose a valid style of loading data. "
                f"Current available options include: {supported_styles}. "
                f"You have choosen: {style}"
            )
        variables_to_update: Dict[str, Sequence[str]] = {}
        for var in var_to_update.keys():
            variables_to_update[self.name + ":" + var] = [
                self.name + ":" + v for v in var_to_update[var]
            ]
        return length, variables_to_update

    def load_chunk(self, chunk_id: int) -> int:
        raise NotImplementedError

    def get_item(self, index: int) -> Dict[str, torch.tensor]:
        raise NotImplementedError

    def get_constants_data(self) -> Dict[str, torch.tensor]:
        raise NotImplementedError

    def get_time(self) -> Dict[str, Union[np.ndarray, None]]:
        raise NotImplementedError

    def get_metadata(self) -> Dict[str, Any]:
        raise NotImplementedError


ClimateDatasetArgs._data_class = ClimateDataset

# Standard library
from dataclasses import dataclass
from itertools import repeat
from typing import List, Optional, Tuple

# Third party
import torch


@dataclass
class MockTensor:
    shape: Tuple[int, ...]


class MockDataModule:
    def __init__(
        self,
        num_batches: int,
        history: int,
        num_in_vars: int,
        num_out_vars: int,
        height: int,
        width: int,
        in_vars: Optional[List[str]] = None,
        out_vars: Optional[List[str]] = None,
    ):
        if in_vars is None:
            self.in_vars = [f"var{i}" for i in range(num_in_vars)]
        else:
            self.in_vars = in_vars
        if out_vars is None:
            self.out_vars = [f"var{i}" for i in range(num_out_vars)]
        else:
            self.out_vars = out_vars
        if history == 0:
            x = MockTensor((num_batches, num_in_vars, height, width))
        else:
            x = MockTensor((num_batches, history, num_in_vars, height, width))
        y = MockTensor((num_batches, num_out_vars, height, width))
        self.lat = None
        self.lon = None
        batch = (x, y, in_vars, out_vars)
        self.batches = repeat(batch)
        self.setup_complete = False

    def setup(self):
        self.lat = torch.zeros(2, 2)
        self.lon = torch.zeros(2, 2)
        self.setup_complete = True

    def train_dataloader(self):
        if self.setup_complete:
            return self.batches
        return None

    def get_climatology(self, *args, **kwargs):
        if self.setup_complete:
            return {"a": torch.zeros(2, 2), "b": torch.zeros(2, 2)}
        return None

    def get_lat_lon(self):
        return self.lat, self.lon

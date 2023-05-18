# Standard library
from typing import Callable, Dict, Sequence, Tuple

# Third party
import torch

# Local application
from .args import ForecastingArgs
from .task import Task

Data = Dict[str, torch.tensor]


class Forecasting(Task):
    _args_class: Callable[..., ForecastingArgs] = ForecastingArgs

    def __init__(self, task_args: ForecastingArgs) -> None:
        super().__init__(task_args)
        self.history: int = task_args.history
        self.window: int = task_args.window
        self.pred_range: int = task_args.pred_range

    def setup(
        self, data_len: int, variables_to_update: Dict[str, Sequence[str]] = {}
    ) -> int:
        super().setup(data_len, variables_to_update)
        return (
            data_len - ((self.history - 1) * self.window + self.pred_range)
        ) // self.subsample

    def get_raw_index(self, index: int) -> Sequence[int]:
        indices: Sequence[int] = []
        raw_index = index * self.subsample
        for i in range(self.history):
            indices.append(raw_index + self.window * i)
        # Add the index for output
        out_idx = raw_index + (self.history - 1) * self.window + self.pred_range
        indices.append(out_idx)
        return indices

    def get_time_index(self, index: int) -> int:
        return index * self.subsample + (self.history - 1) * self.window

    def create_constants_data(
        self, constants_data: Data, apply_transform: bool = 1
    ) -> Data:
        # Repeating constants along history dimension
        const_data: Data = {
            k: constants_data[k].repeat(self.history, 1, 1) for k in self.constants
        }  # [history, lat, lon]

        # transforms.Normalize works only on image like data (C * H * W)
        # hence adding channel via unsqueeze and
        # then removing it after transformation
        if apply_transform:
            const_data = {
                k: (self.const_transform[k](const_data[k].unsqueeze(1))).squeeze(1)
                for k in self.constants
            }
        return const_data

    def create_inp_out(
        self,
        raw_data: Data,
        constants_data: Data,
        apply_transform: bool = 1,
    ) -> Tuple[Data, Data]:
        inp_data: Data = {
            k: raw_data[k][:-1] for k in self.in_vars
        }  # [history, lat, lon]
        out_data: Data = {k: raw_data[k][-1] for k in self.out_vars}  # [lat, lon]

        # transforms.Normalize works only on image like data (C * H * W)
        # hence adding channel via unsqueeze and
        # then removing it after transformation
        if apply_transform:
            # Need to unsqueeze for inp_data as history is not the same as channel
            inp_data = {
                k: (self.inp_transform[k](inp_data[k].unsqueeze(1))).squeeze(1)
                for k in self.in_vars
            }
            out_data = {
                k: (self.out_transform[k](out_data[k].unsqueeze(0))).squeeze(0)
                for k in self.out_vars
            }

        return inp_data, out_data


ForecastingArgs._task_class = Forecasting

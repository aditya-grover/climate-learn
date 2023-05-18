# Standard library
from typing import Callable, Dict, Sequence, Tuple

# Third party
import torch

# Local application
from .args import DownscalingArgs
from .task import Task

Data = Dict[str, torch.tensor]


class Downscaling(Task):
    _args_class: Callable[..., DownscalingArgs] = DownscalingArgs

    def __init__(self, task_args: DownscalingArgs) -> None:
        super().__init__(task_args)

    def setup(
        self, data_len: int, variables_to_update: Dict[str, Sequence[str]] = {}
    ) -> int:
        # # why is this a single number instead of a tuple
        # self.downscale_ratio: Any = (
        #     self.out_data.shape[-1] // self.inp_data.shape[-1]
        # )  # TODO add stronger typecheck
        return super().setup(data_len, variables_to_update)

    def get_raw_index(self, index: int) -> int:
        return index * self.subsample

    def get_time_index(self, index: int) -> int:
        return index * self.subsample

    def create_constants_data(
        self, constants_data: Data, apply_transform: bool = 1
    ) -> Data:
        const_data: Data = {k: constants_data[k] for k in self.constants}

        # transforms.Normalize works only on image like data (C * H * W)
        # hence adding channel via unsqueeze and
        # then removing it after transformation
        if apply_transform:
            const_data = {
                k: (self.const_transform[k](const_data[k].unsqueeze(0))).squeeze(0)
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
            k: raw_data[k] for k in self.in_vars
        }  # [lowres_lat, lowres_lon]
        out_data: Data = {
            k: raw_data[k] for k in self.out_vars
        }  # [highres_lat, highres_lon]

        # transforms.Normalize works only on image like data (C * H * W)
        # hence adding channel via unsqueeze and
        # then removing it after transformation
        if apply_transform:
            inp_data = {
                k: (self.inp_transform[k](inp_data[k].unsqueeze(0))).squeeze(0)
                for k in self.in_vars
            }
            out_data = {
                k: (self.out_transform[k](out_data[k].unsqueeze(0))).squeeze(0)
                for k in self.out_vars
            }

        return inp_data, out_data


DownscalingArgs._task_class = Downscaling

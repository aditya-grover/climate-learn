from typing import Any, Callable, Dict, Sequence, Tuple
import torch
import numpy as np

# import xarray as xr
# from torchvision.transforms import transforms

from climate_learn.data.climate_dataset import *
from climate_learn.data.task.task import Task
from climate_learn.data.task.args import DownscalingArgs


class Downscaling(Task):
    _args_class: Callable[..., DownscalingArgs] = DownscalingArgs

    def __init__(self, task_args: DownscalingArgs) -> None:
        super().__init__(task_args)

    def setup(
        self,
        data_len: int,
        variables_to_update: Sequence[Dict[str, Sequence[str]]] = [{}, {}],
    ) -> int:
        # # why is this a single number instead of a tuple
        # self.downscale_ratio: Any = (
        #     self.out_data.shape[-1] // self.inp_data.shape[-1]
        # )  # TODO add stronger typecheck
        # Assuming that variables_to_update is a list of dict
        # As it is coming from StackedClimateDataset
        for variable in variables_to_update[0]:
            if variable in self.in_vars:
                self.in_vars.remove(variable)
                for variable_to_add in variables_to_update[0][variable]:
                    self.in_vars.append(variable_to_add)

        for variable in variables_to_update[1]:
            if variable in self.out_vars:
                self.out_vars.remove(variable)
                for variable_to_add in variables_to_update[1][variable]:
                    self.out_vars.append(variable_to_add)

        return data_len // self.subsample

    def get_raw_index(self, index: int) -> int:
        return index * self.subsample

    def create_inp_out(
        self,
        raw_data: Dict[str, torch.tensor],
        constants_data: Dict[str, torch.tensor],
        apply_transform: bool = 1,
    ) -> Tuple[Dict[str, torch.tensor], Dict[str, torch.tensor]]:
        inp_data: Dict[str, torch.tensor] = {
            k: raw_data[0][k] for k in self.in_vars
        }  # [32, 64]
        out_data: Dict[str, torch.tensor] = {
            k: raw_data[1][k] for k in self.out_vars
        }  # [64, 128]

        # transforms.Normalize works only on image like data (C * H * W), hence adding channel via unsqueeze and then removing it after transformation
        if apply_transform:
            inp_data = {
                k: (self.inp_transform[k](inp_data[k].unsqueeze(0))).squeeze(0)
                for k in self.in_vars
            }
            out_data = {
                k: (self.out_transform[k](out_data[k].unsqueeze(0))).squeeze(0)
                for k in self.out_vars
            }

        for constant in self.constant_names:
            constant_data: Dict[str, torch.tensor] = constants_data[0][
                constant
            ]  # [32, 64]
            if apply_transform:
                constant_data = (
                    self.constant_transform[constant](constant_data.unsqueeze(0))
                ).squeeze(0)
            inp_data[constant] = constant_data

        return inp_data, out_data


DownscalingArgs._task_class = Downscaling

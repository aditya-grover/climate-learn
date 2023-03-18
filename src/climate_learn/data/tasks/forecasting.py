from typing import Callable, Sequence, Tuple, Union
import torch
import numpy as np

from climate_learn.data.tasks.task import Task
from climate_learn.data.tasks.args import ForecastingArgs


class Forecasting(Task):
    _args_class: Callable[..., ForecastingArgs] = ForecastingArgs

    def __init__(self, task_args: ForecastingArgs) -> None:
        super().__init__(task_args)

        # assert set(self.in_vars) <= set(self.dataset.variables)
        # assert set(self.out_vars) <= set(self.dataset.variables)

        self.history: int = task_args.history
        self.window: int = task_args.window
        self.pred_range: int = task_args.pred_range

    def setup(self, data_len, variables_to_update={}) -> None:
        for variable in variables_to_update:
            if variable in self.in_vars:
                self.in_vars.remove(variable)
                for variable_to_add in variables_to_update[variable]:
                    self.in_vars.append(variable_to_add)
            if variable in self.out_vars:
                self.out_vars.remove(variable)
                for variable_to_add in variables_to_update[variable]:
                    self.out_vars.append(variable_to_add)

        return (
            data_len - ((self.history - 1) * self.window + self.pred_range)
        ) // self.subsample

    def get_raw_index(self, index):
        indices = []
        raw_index = index * self.subsample
        for i in range(self.history):
            indices.append(raw_index + self.window * i)
        # Add the index for output
        out_idx = raw_index + (self.history - 1) * self.window + self.pred_range
        indices.append(out_idx)
        return indices

    def create_inp_out(
        self, raw_data, constants_data, apply_transform: bool = 1
    ) -> Tuple[np.ndarray, np.ndarray]:
        inp_data = {k: raw_data[k][:-1] for k in self.in_vars}  # [history, 32, 64]
        out_data = {k: raw_data[k][-1] for k in self.out_vars}  # [32, 64]

        # transforms.Normalize works only on image like data (C * H * W), hence adding channel via unsqueeze and then removing it after transformation
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

        for constant in self.constant_names:
            constant_data = constants_data[constant].repeat(self.history, 1, 1)
            if apply_transform:
                constant_data = (
                    self.constant_transform[constant](constant_data.unsqueeze(0))
                ).squeeze(0)
            inp_data[constant] = constant_data

        return inp_data, out_data


ForecastingArgs._task_class = Forecasting

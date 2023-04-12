# Standard library
from typing import Callable, Dict, Sequence, Tuple

# Third party
import torch

# Local application
from climate_learn.data.task.task import Task
from climate_learn.data.task.args import ForecastingArgs

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
        # Assuming that variables_to_update is a dict
        in_vars: Sequence[str] = []
        out_vars: Sequence[str] = []
        for variable in self.in_vars:
            if variable in variables_to_update.keys():
                for variable_to_add in variables_to_update[variable]:
                    in_vars.append(variable_to_add)
            else:
                in_vars.append(variable)
        for variable in self.out_vars:
            if variable in variables_to_update.keys():
                for variable_to_add in variables_to_update[variable]:
                    out_vars.append(variable_to_add)
            else:
                out_vars.append(variable)
        ## using dict instead of set to preserve insertion order
        self.in_vars = list(dict.fromkeys(in_vars))
        self.out_vars = list(dict.fromkeys(out_vars))

        variables_available: Sequence[str] = []
        for variables in variables_to_update.values():
            variables_available.extend(variables)
        variables_available = set(variables_available)

        if not set(self.in_vars).issubset(variables_available):
            RuntimeError(
                f"The input variables required by the task: {self.in_vars} "
                f"are not available in the dataset: {variables_available}"
            )

        if not set(self.out_vars).issubset(variables_available):
            RuntimeError(
                f"The output variables required by the task: {self.in_vars} "
                f"are not available in the dataset: {variables_available}"
            )

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

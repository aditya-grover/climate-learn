# Standard library
from typing import Callable, Dict, Sequence, Tuple

# Third party
import torch

# Local application
from climate_learn.data.climate_dataset import *
from climate_learn.data.task.task import Task
from climate_learn.data.task.args import DownscalingArgs

Data = Dict[str, torch.tensor]


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
        in_vars: Sequence[str] = []
        out_vars: Sequence[str] = []
        for variable in self.in_vars:
            if variable in variables_to_update[0].keys():
                for variable_to_add in variables_to_update[0][variable]:
                    in_vars.append(variable_to_add)
            else:
                in_vars.append(variable)
        for variable in self.out_vars:
            if variable in variables_to_update[1].keys():
                for variable_to_add in variables_to_update[1][variable]:
                    out_vars.append(variable_to_add)
            else:
                out_vars.append(variable)
        ## using dict instead of set to preserve insertion order
        self.in_vars = list(dict.fromkeys(in_vars))
        self.out_vars = list(dict.fromkeys(out_vars))

        variables_available_input: Sequence[str] = []
        for variables in variables_to_update[0].values():
            variables_available_input.extend(variables)
        variables_available_input = set(variables_available_input)

        variables_available_output: Sequence[str] = []
        for variables in variables_to_update[1].values():
            variables_available_output.extend(variables)
        variables_available_output = set(variables_available_output)

        if not set(self.in_vars).issubset(variables_available_input):
            RuntimeError(
                f"The input variables required by the task: {self.in_vars} "
                f"are not available in the dataset: {variables_available_input}"
            )

        if not set(self.out_vars).issubset(variables_available_output):
            RuntimeError(
                f"The output variables required by the task: {self.in_vars} "
                f"are not available in the dataset: {variables_available_output}"
            )

        return data_len // self.subsample

    def get_raw_index(self, index: int) -> int:
        return index * self.subsample

    def get_time_index(self, index: int) -> int:
        return index * self.subsample

    def create_constants_data(
        self, constants_data: Data, apply_transform: bool = 1
    ) -> Data:
        ## Need constants data from the first dataset only
        const_data: Data = {
            k: constants_data[0][k] for k in self.constants
        }  # [lowres_lat, lowres_lon]

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
        ## First dataset contains the input
        inp_data: Data = {
            k: raw_data[0][k] for k in self.in_vars
        }  # [lowres_lat, lowres_lon]
        ## Second dataset contains the output
        out_data: Data = {
            k: raw_data[1][k] for k in self.out_vars
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

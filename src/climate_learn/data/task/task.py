# Standard library
from abc import ABC
from typing import Callable, Dict, Sequence, Tuple, Union

# Third party
import torch
from torchvision.transforms import transforms

# Local application
from .args import TaskArgs

Data = Dict[str, torch.tensor]
Transform = Dict[str, transforms.Normalize]


class Task(ABC):
    _args_class: Callable[..., TaskArgs] = TaskArgs

    def __init__(self, task_args: TaskArgs) -> None:
        super().__init__()
        self.in_vars: Sequence[str] = task_args.in_vars
        self.out_vars: Sequence[str] = task_args.out_vars
        self.constants: Sequence[str] = task_args.constants
        self.subsample: int = task_args.subsample

        self.inp_transform: Union[None, Transform] = None
        self.out_transform: Union[None, Transform] = None
        self.const_transform: Union[None, Transform] = None

    def setup(
        self, data_len: int, variables_to_update: Dict[str, Sequence[str]] = {}
    ) -> int:
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
        return data_len // self.subsample

    def set_normalize(
        self,
        inp_normalize: Transform,
        out_normalize: Transform,
        const_normalize: Transform,
    ) -> None:  # for val and test
        self.inp_transform = {}
        for var in self.in_vars:
            self.inp_transform[var] = inp_normalize[var]

        self.out_transform = {}
        for var in self.out_vars:
            self.out_transform[var] = out_normalize[var]

        self.const_transform = {}
        for var in self.constants:
            self.const_transform[var] = const_normalize[var]

    def get_transforms(self) -> Tuple[Transform, Transform, Transform]:
        if self.inp_transform == None:
            raise RuntimeError(f"Input transforms has not been yet set.")
        if self.out_transform == None:
            raise RuntimeError(f"Output transforms has not been yet set.")
        if self.const_transform == None:
            raise RuntimeError(f"Constants transforms has not been yet set.")
        return self.inp_transform, self.out_transform, self.const_transform

    def get_raw_index(self, index: int) -> Union[Sequence[int], int]:
        raise NotImplementedError

    def get_time_index(self, index: int) -> int:
        raise NotImplementedError

    def create_constants_data(
        self, constants_data: Data, apply_transform: bool = 1
    ) -> Data:
        raise NotImplementedError

    def create_inp_out(
        self,
        raw_data: Data,
        constants_data: Data,
        apply_transform: bool = 1,
    ) -> Tuple[Data, Data, Data]:
        raise NotImplementedError


TaskArgs._task_class = Task

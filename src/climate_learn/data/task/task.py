# Standard library
from abc import ABC
from typing import Callable, Dict, Sequence, Tuple, Union

# Third party
import torch
from torchvision.transforms import transforms

# Local application
from climate_learn.data.task.args import TaskArgs

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

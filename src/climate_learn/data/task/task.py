# Standard library
from abc import ABC
from typing import Callable, Dict, Sequence, Tuple, Union

# Third party
import torch
from torchvision.transforms import transforms

# Local application
from climate_learn.data.task.args import TaskArgs


class Task(ABC):
    _args_class: Callable[..., TaskArgs] = TaskArgs

    def __init__(self, task_args: TaskArgs) -> None:
        super().__init__()
        self.in_vars: Sequence[str] = task_args.in_vars
        self.out_vars: Sequence[str] = task_args.out_vars
        self.constant_names: Sequence[str] = task_args.constant_names
        self.subsample: int = task_args.subsample

    def setup(
        self, data_len: int, variables_to_update: Dict[str, Sequence[str]] = {}
    ) -> int:
        return data_len // self.subsample

    def set_normalize(
        self,
        inp_normalize: Dict[str, transforms.Normalize],
        out_normalize: Dict[str, transforms.Normalize],
    ) -> None:  # for val and test
        self.inp_transform: Dict[str, transforms.Normalize] = {}
        for var in self.in_vars:
            self.inp_transform[var] = inp_normalize[var]

        self.out_transform: Dict[str, transforms.Normalize] = {}
        for var in self.out_vars:
            self.out_transform[var] = out_normalize[var]

        self.constant_transform: Dict[str, transforms.Normalize] = {}
        for var in self.constant_names:
            self.constant_transform[var] = inp_normalize[var]

    def get_raw_index(self, index: int) -> Union[Sequence[int], int]:
        pass

    def create_inp_out(
        self,
        raw_data: Dict[str, torch.tensor],
        constants_data: Dict[str, torch.tensor],
        apply_transform: bool = 1,
    ) -> Tuple[Dict[str, torch.tensor], Dict[str, torch.tensor]]:
        pass


TaskArgs._task_class = Task

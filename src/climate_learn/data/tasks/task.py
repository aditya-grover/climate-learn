from abc import ABC
from typing import Callable, Sequence, Union
import numpy as np

from climate_learn.data.tasks.args import TaskArgs


class Task(ABC):
    _args_class: Callable[..., TaskArgs] = TaskArgs

    def __init__(self, task_args: TaskArgs) -> None:
        super().__init__()
        self.in_vars: Sequence[str] = task_args.in_vars
        self.out_vars: Sequence[str] = task_args.out_vars
        self.constant_names: Sequence[str] = task_args.constant_names
        self.subsample: int = task_args.subsample

    def setup(self, data_len, variables_to_update) -> None:
        return data_len

    def set_normalize(
        self,
        inp_normalize,
        out_normalize,
    ) -> None:  # for val and test
        self.inp_transform = {}
        for var in self.in_vars:
            self.inp_transform[var] = inp_normalize[var]

        self.out_transform = {}
        for var in self.out_vars:
            self.out_transform[var] = out_normalize[var]

        self.constant_transform = {}
        for var in self.constant_names:
            self.constant_transform[var] = inp_normalize[var]

    def get_raw_index(self, index):
        pass

    def create_inp_out(self, raw_data, constants_data, apply_transform: bool = 1):
        pass


TaskArgs._task_class = Task

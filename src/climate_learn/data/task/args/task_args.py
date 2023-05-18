# Standard library
from __future__ import annotations
from abc import ABC
import copy
from typing import Any, Callable, Dict, Sequence, TYPE_CHECKING, Union

# Local application
if TYPE_CHECKING:
    from ..task import Task


class TaskArgs(ABC):
    _task_class: Union[Callable[..., Task], str] = "Task"

    def __init__(
        self,
        in_vars: Sequence[str],
        out_vars: Sequence[str],
        constants: Sequence[str] = [],
        subsample: int = 1,
    ) -> None:
        self.in_vars: Sequence[str] = in_vars
        self.out_vars: Sequence[str] = out_vars
        self.constants: Sequence[str] = constants
        self.subsample: int = subsample
        TaskArgs.check_validity(self)

    def create_copy(self, args: Dict[str, Any] = {}) -> TaskArgs:
        new_instance: TaskArgs = copy.deepcopy(self)
        for arg in args:
            if hasattr(new_instance, arg):
                setattr(new_instance, arg, args[arg])
        TaskArgs.check_validity(new_instance)
        return new_instance

    def check_validity(self) -> None:
        if self.subsample <= 0:
            raise RuntimeError(
                f"Subsample rate should be a positive integer. "
                f"Currently set to {self.subsample}."
            )

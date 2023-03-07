from __future__ import annotations
from abc import ABC
from typing import Callable, Sequence, TYPE_CHECKING, Union

if TYPE_CHECKING:
    from climate_learn.data.tasks import Task


class TaskArgs(ABC):
    _task_class: Union[Callable[..., Task], str] = "Task"

    def __init__(
        self,
        in_vars: Sequence[str],
        out_vars: Sequence[str],
        constant_names: Sequence[str] = [],
        subsample: int = 1,
    ) -> None:
        self.in_vars: Sequence[str] = in_vars
        self.out_vars: Sequence[str] = out_vars
        self.constant_names: Sequence[str] = constant_names
        self.subsample: int = subsample

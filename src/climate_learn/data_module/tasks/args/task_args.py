from typing import Any, Callable, Sequence, Union
from climate_learn.data_module.data.args import DataArgs


class TaskArgs:
    _task_class: Union[Callable[..., Any], str] = "Task"

    def __init__(
        self,
        dataset_args: DataArgs,
        in_vars: Sequence[str],
        out_vars: Sequence[str],
        constant_names: Sequence[str] = [],
        subsample: int = 1,
        split: str = "train",
    ) -> None:
        self.dataset_args: DataArgs = dataset_args
        self.in_vars: Sequence[str] = in_vars
        self.out_vars: Sequence[str] = out_vars
        self.constant_names: Sequence[str] = constant_names
        self.subsample: int = subsample
        self.split: str = split

    def setup(self, data_module_args: Any) -> None:  # TODO add stronger typecheck
        self.dataset_args.split = self.split
        self.dataset_args.setup(data_module_args)

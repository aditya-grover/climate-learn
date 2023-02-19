from typing import Any, Callable, Sequence, Union


class DataArgs:
    _data_class: Union[Callable[..., Any], str] = "Data"

    def __init__(self, variables: Sequence[str], split: str = "train") -> None:
        self.variables: Sequence[str] = variables
        self.split: str = split

    def setup(self, data_module_args: Any) -> None:  # TODO add stronger typecheck
        pass

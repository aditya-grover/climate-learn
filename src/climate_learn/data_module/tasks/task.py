from typing import Any, Callable, Sequence
import numpy as np

from torch.utils.data import Dataset
from torchvision.transforms import transforms

from climate_learn.data_module.data import Data
from climate_learn.data_module.tasks.args import TaskArgs


class Task(Dataset):
    args_class: Callable[..., TaskArgs] = TaskArgs

    def __init__(self, task_args: TaskArgs) -> None:
        super().__init__()
        if isinstance(task_args.dataset_args._data_class, str):
            dataset_class: Callable[..., Data] = eval(
                task_args.dataset_args._data_class
            )
        else:
            dataset_class: Callable[..., Data] = task_args.dataset_args._data_class
        self.dataset: Data = dataset_class(task_args.dataset_args)

        self.in_vars: Sequence[str] = task_args.in_vars
        self.out_vars: Sequence[str] = task_args.out_vars
        self.constant_names: Sequence[str] = task_args.constant_names
        self.subsample: int = task_args.subsample
        self.split: str = task_args.split

    def setup(self) -> None:
        print(f"Creating {self.split} dataset")
        self.dataset.setup()
        self.lat: Any = self.dataset.lat  # TODO add stronger typecheck
        self.lon: Any = self.dataset.lon  # TODO add stronger typecheck

    def get_normalize(self, data: Any) -> Any:  # TODO add stronger typecheck
        mean = np.mean(data, axis=(0, 2, 3))
        std = np.std(data, axis=(0, 2, 3))
        return transforms.Normalize(mean, std)

    def set_normalize(
        self, inp_normalize: Any, out_normalize: Any, constant_normalize: Any
    ) -> None:  # for val and test #TODO add stronger typecheck
        self.inp_transform: Any = inp_normalize
        self.out_transform: Any = out_normalize
        self.constant_transform: Any = constant_normalize


TaskArgs._task_class = Task

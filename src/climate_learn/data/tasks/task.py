from typing import Callable, Sequence, Union
import numpy as np

from torch.utils.data import Dataset
from torchvision.transforms import transforms

from climate_learn.data.climate_dataset import ClimateDataset
from climate_learn.data.tasks.args import TaskArgs


class Task(Dataset):
    args_class: Callable[..., TaskArgs] = TaskArgs

    def __init__(self, task_args: TaskArgs) -> None:
        super().__init__()
        if isinstance(task_args.dataset_args._data_class, str):
            dataset_class: Callable[..., ClimateDataset] = eval(
                task_args.dataset_args._data_class
            )
        else:
            dataset_class: Callable[
                ..., ClimateDataset
            ] = task_args.dataset_args._data_class
        self.dataset: ClimateDataset = dataset_class(task_args.dataset_args)

        self.in_vars: Sequence[str] = task_args.in_vars
        self.out_vars: Sequence[str] = task_args.out_vars
        self.constant_names: Sequence[str] = task_args.constant_names
        self.subsample: int = task_args.subsample
        self.split: str = task_args.split

    def setup(self) -> None:
        print(f"Creating {self.split} dataset")
        self.dataset.setup()
        self.lat: np.ndarray = self.dataset.lat
        self.lon: np.ndarray = self.dataset.lon

    def get_normalize(self, data: np.ndarray) -> transforms.Normalize:
        mean = np.mean(data, axis=(0, 2, 3))
        std = np.std(data, axis=(0, 2, 3))
        return transforms.Normalize(mean, std)

    def set_normalize(
        self,
        inp_normalize: Union[transforms.Normalize, None],
        out_normalize: Union[transforms.Normalize, None],
        constant_normalize: Union[transforms.Normalize, None],
    ) -> None:  # for val and test
        self.inp_transform: Union[transforms.Normalize, None] = inp_normalize
        self.out_transform: Union[transforms.Normalize, None] = out_normalize
        self.constant_transform: Union[transforms.Normalize, None] = constant_normalize


TaskArgs._task_class = Task

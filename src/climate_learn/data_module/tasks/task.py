import numpy as np

from torch.utils.data import Dataset
from torchvision.transforms import transforms

from climate_learn.data_module.data import *
from climate_learn.data_module.tasks.args import TaskArgs


class Task(Dataset):
    args_class = TaskArgs

    def __init__(self, task_args):
        super().__init__()
        dataset_class = eval(task_args.dataset_args.data_class)
        self.dataset = dataset_class(task_args.dataset_args)

        self.in_vars = task_args.in_vars
        self.out_vars = task_args.out_vars
        self.constant_names = task_args.constant_names
        self.subsample = task_args.subsample
        self.split = task_args.split

    def setup(self):
        print(f"Creating {self.split} dataset")
        self.dataset.setup()
        self.lat = self.dataset.lat
        self.lon = self.dataset.lon

    def get_normalize(self, data):
        mean = np.mean(data, axis=(0, 2, 3))
        std = np.std(data, axis=(0, 2, 3))
        return transforms.Normalize(mean, std)

    def set_normalize(
        self, inp_normalize, out_normalize, constant_normalize
    ):  # for val and test
        self.inp_transform = inp_normalize
        self.out_transform = out_normalize
        self.constant_transform = constant_normalize

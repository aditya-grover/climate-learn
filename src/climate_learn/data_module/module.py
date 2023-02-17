# Local application
from .data import *
from ..utils.datetime import Year, Hours

import copy

# Third party
import torch
from torch.utils.data import DataLoader
from lightning import LightningDataModule

# TODO: include exceptions in docstrings
# TODO: document legal input/output variables for each dataset


def collate_fn(batch):
    r"""Collate function for DataLoaders.

    :param batch: A batch of data samples.
    :type batch: List[Tuple[torch.Tensor, torch.Tensor, List[str], List[str]]]
    :return: A tuple of `input`, `output`, `variables`, and `out_variables`.
    :rtype: Tuple[torch.Tensor, torch.Tensor, List[str], List[str]]
    """
    inp = torch.stack([batch[i][0] for i in range(len(batch))])
    out = torch.stack([batch[i][1] for i in range(len(batch))])
    variables = batch[0][2]
    out_variables = batch[0][3]
    return inp, out, variables, out_variables


class DataModuleArgs:
    def __init__(
        self,
        task_args,
        train_start_year,
        val_start_year,
        test_start_year,
        end_year=Year(2018),
    ):
        
        self.train_start_year = train_start_year
        self.val_start_year = val_start_year
        self.test_start_year = test_start_year
        self.end_year = end_year

        self.train_task_args = copy.deepcopy(task_args)
        self.train_task_args.split = "train"
        self.train_task_args.setup(self)

        self.val_task_args = copy.deepcopy(task_args)
        self.val_task_args.split = "val"
        self.val_task_args.setup(self)

        self.test_task_args = copy.deepcopy(task_args)
        self.test_task_args.split = "test"
        self.test_task_args.setup(self)


class DataModule(LightningDataModule):
    """ClimateLearn's data module interface. Encapsulates dataset/task-specific
    data modules."""

    def __init__(
        self,
        data_module_args,
        batch_size=64,
        num_workers=0,
        pin_memory=False,
    ):
        r"""
        .. highlight:: python

        :param dataset: The name of the dataset to use. Currently supported
            options are: "ERA5".
        :type dataset: str
        :param task: The name of the task to use. Currently supported options
            are: "forecasting", "downscaling".
        :type task: str
        :param root_dir: The name of the local directory containing the
            specified dataset.
        :type root_dir: str
        :param in_vars: A list of input variables to use.
        :type in_vars: List[str]
        :param out_vars: A list of output variables to use.
        :type out_vars: List[str]
        :param train_start_year: The first year of the training set, inclusive.
        :type train_start_year: Year
        :param val_start_year: The first year of the validation set, inclusive.
            :python:`val_start_year` must be at least
            :python:`train_start_year+1` since the training set ends the year
            before :python:`val_start_year`. E.g., if
            :python:`train_start_year` is 1970, then
            :python:`val_start_year` must be 1971 or later.
        :type val_start_year: Year
        :param test_start_year: The first year of the testing set, inclusive.
            :python:`test_start_year` must be at least
            :python:`val_start_year+1` since the validation set ends the year
            before :python:`test_start_year`. E.g., if
            :python:`val_start_year` is 2015, then
            :python:`test_start_year` must be 2016 or later.
        :type test_start_year: Year
        :param end_year: The last year of the testing set, inclusive.
            Default is :python:`Year(2018)`.
        :type end_year: Year, optional
        :param root_highres_dir: The name of the high-res data directory, which
            is needed for downsclaing task. Default is `None`.
        :type root_highres_dir: str, optional
        """
        super().__init__()

        assert (
            data_module_args.end_year >= data_module_args.test_start_year
            and data_module_args.test_start_year > data_module_args.val_start_year
            and data_module_args.val_start_year > data_module_args.train_start_year
        )
        self.save_hyperparameters(logger=False)
        task_class = data_module_args.task_args.task_class

        self.train_dataset = task_class(data_module_args.train_task_args)
        self.train_dataset.setup()

        self.val_dataset = task_class(data_module_args.val_task_args)
        self.val_dataset.setup()
        self.val_dataset.set_normalize(
            self.train_dataset.inp_transform,
            self.train_dataset.out_transform,
            self.train_dataset.constant_transform,
        )

        self.test_dataset = task_class(data_module_args.test_task_args)
        self.test_dataset.setup()
        self.test_dataset.set_normalize(
            self.train_dataset.inp_transform,
            self.train_dataset.out_transform,
            self.train_dataset.constant_transform,
        )

    def get_lat_lon(self):
        return self.train_dataset.lat, self.train_dataset.lon

    def get_out_transforms(self):
        return self.train_dataset.out_transform

    def get_climatology(self, split="val"):
        if split == "train":
            return self.train_dataset.get_climatology()
        elif split == "val":
            return self.val_dataset.get_climatology()
        elif split == "test":
            return self.test_dataset.get_climatology()
        else:
            raise NotImplementedError

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            shuffle=True,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            collate_fn=collate_fn,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            shuffle=False,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            collate_fn=collate_fn,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            shuffle=False,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            collate_fn=collate_fn,
        )

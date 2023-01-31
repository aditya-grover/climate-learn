# Local application
from .modules import *
from ..utils.datetime import Year, Hours

# Third party
import torch
from torch.utils.data import DataLoader
from pytorch_lightning import LightningDataModule

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


class DataModule(LightningDataModule):
    """ClimateLearn's data module interface. Encapsulates dataset/task-specific
    data modules."""

    def __init__(
        self,
        dataset,
        task,
        root_dir,
        in_vars,
        out_vars,
        train_start_year,
        val_start_year,
        test_start_year,
        end_year=Year(2018),
        aux_dirs=[],
        history: int = 1,
        window: int = 6,
        pred_range=Hours(6),
        subsample=Hours(1),
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
        :param aux_dirs: The names of auxilliary dataset directories, which
            need to be provided for some tasks. Default is the empty list.
        :type aux_dirs: List[str], optional
        """
        super().__init__()

        assert (
            end_year >= test_start_year
            and test_start_year > val_start_year
            and val_start_year > train_start_year
        )
        self.save_hyperparameters(logger=False)

        if dataset != "ERA5":
            raise NotImplementedError("Only support ERA5")
        if task == "downscaling" and root_highres_dir is None:
            raise NotImplementedError(
                "High-resolution data has to be provided for downscaling"
            )

        task_string = "Forecasting" if task == "forecasting" else "Downscaling"
        caller = eval(f"{dataset.upper()}{task_string}")

        train_years = range(train_start_year, val_start_year)
        self.train_dataset = caller(
            root_dir,
            root_highres_dir,
            in_vars,
            out_vars,
            history,
            window,
            pred_range.hours(),
            train_years,
            subsample.hours(),
            "train",
        )

        val_years = range(val_start_year, test_start_year)
        self.val_dataset = caller(
            root_dir,
            root_highres_dir,
            in_vars,
            out_vars,
            history,
            window,
            pred_range.hours(),
            val_years,
            subsample.hours(),
            "val",
        )
        self.val_dataset.set_normalize(
            self.train_dataset.inp_transform,
            self.train_dataset.out_transform,
            self.train_dataset.constant_transform,
        )

        test_years = range(test_start_year, end_year + 1)
        self.test_dataset = caller(
            root_dir,
            root_highres_dir,
            in_vars,
            out_vars,
            history,
            window,
            pred_range.hours(),
            test_years,
            subsample.hours(),
            "test",
        )
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

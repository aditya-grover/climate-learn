# Standard library
from abc import ABC
import copy
from typing import Callable, Dict, Sequence, Tuple, Union

# Third party
import numpy as np
from pytorch_lightning import LightningDataModule
import torch
from torch.utils.data import DataLoader, IterableDataset
from torchvision.transforms import transforms

# Local application
from climate_learn.data.dataset import (
    MapDatasetArgs,
    MapDataset,
    ShardDatasetArgs,
    ShardDataset,
)
from climate_learn.utils.datetime import Year, Hours

# TODO: include exceptions in docstrings
# TODO: document legal input/output variables for each dataset


def collate_fn(
    batch,
) -> Tuple[torch.tensor, torch.tensor, Sequence[str], Sequence[str]]:
    r"""Collate function for DataLoaders.

    :param batch: A batch of data samples.
    :type batch: List[Tuple[torch.Tensor, torch.Tensor, List[str], List[str]]]
    :return: A tuple of `input`, `output`, `variables`, and `out_variables`.
    :rtype: Tuple[torch.Tensor, torch.Tensor, List[str], List[str]]
    """

    def handle_dict_features(t: Dict[str, torch.tensor]) -> torch.tensor:
        ## Hotfix for the models to work with dict style data
        t = torch.stack(tuple(t.values()))
        ## Handles the case for forecasting input as it has history in it
        ## TODO: Come up with an efficient solution instead of if condition
        if len(t.size()) == 4:
            return torch.transpose(t, 0, 1)
        return t

    inp = torch.stack([handle_dict_features(batch[i][0]) for i in range(len(batch))])
    out = torch.stack([handle_dict_features(batch[i][1]) for i in range(len(batch))])
    variables = list(batch[0][0].keys())
    out_variables = list(batch[0][1].keys())
    return inp, out, variables, out_variables


class DataModuleArgs(ABC):
    def __init__(
        self,
        dataset_args: Union[MapDatasetArgs, ShardDatasetArgs],
        train_start_year: int,
        val_start_year: int,
        test_start_year: int,
        end_year: int = 2018,
    ) -> None:
        self.train_start_year: int = train_start_year
        self.val_start_year: int = val_start_year
        self.test_start_year: int = test_start_year
        self.end_year: int = end_year

        self.train_dataset_args: Union[
            MapDatasetArgs, ShardDatasetArgs
        ] = copy.deepcopy(dataset_args)
        self.train_dataset_args.setup(self, "train")

        self.val_dataset_args: Union[MapDatasetArgs, ShardDatasetArgs] = copy.deepcopy(
            dataset_args
        )
        self.val_dataset_args.setup(self, "val")

        self.test_dataset_args: Union[MapDatasetArgs, ShardDatasetArgs] = copy.deepcopy(
            dataset_args
        )
        self.test_dataset_args.setup(self, "test")


class DataModule(LightningDataModule):
    """ClimateLearn's data module interface. Encapsulates dataset/task-specific
    data modules."""

    def __init__(
        self,
        data_module_args: DataModuleArgs,
        batch_size: int = 64,
        num_workers: int = 0,
        pin_memory: bool = False,
    ) -> None:
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
        if isinstance(data_module_args.train_dataset_args._data_class, str):
            data_class: Callable[..., Union[MapDataset, ShardDataset]] = eval(
                data_module_args.train_dataset_args._data_class
            )
        else:
            data_class: Callable[
                ..., Union[MapDataset, ShardDataset]
            ] = data_module_args.train_dataset_args._data_class

        self.train_dataset: Union[MapDataset, ShardDataset] = data_class(
            data_module_args.train_dataset_args
        )

        data_class = MapDataset
        self.val_dataset: Union[MapDataset, ShardDataset] = data_class(
            data_module_args.val_dataset_args
        )

        self.test_dataset: Union[MapDataset, ShardDataset] = data_class(
            data_module_args.test_dataset_args
        )

        self.train_dataset.setup()

        self.val_dataset.setup()
        self.val_dataset.set_normalize(
            self.train_dataset.inp_transforms,
            self.train_dataset.out_transforms,
        )

        self.test_dataset.setup()
        self.test_dataset.set_normalize(
            self.train_dataset.inp_transforms,
            self.train_dataset.out_transforms,
        )

    def get_lat_lon(self) -> Tuple[np.ndarray, np.ndarray]:
        return self.train_dataset.lat, self.train_dataset.lon

    def get_out_transforms(self) -> Union[transforms.Normalize, None]:
        return self.train_dataset.out_transforms

    def get_climatology(self, split: str = "val") -> torch.Tensor:
        if split == "train":
            return self.train_dataset.get_climatology()
        elif split == "val":
            return self.val_dataset.get_climatology()
        elif split == "test":
            return self.test_dataset.get_climatology()
        else:
            raise NotImplementedError

    def build_dataloader(
        self, dataset: Union[MapDataset, ShardDataset], shuffle: bool
    ) -> DataLoader:
        if isinstance(dataset, IterableDataset):
            return DataLoader(
                dataset,
                batch_size=self.hparams.batch_size,
                num_workers=self.hparams.num_workers,
                pin_memory=self.hparams.pin_memory,
                collate_fn=collate_fn,
            )
        else:
            return DataLoader(
                dataset,
                shuffle=shuffle,
                batch_size=self.hparams.batch_size,
                num_workers=self.hparams.num_workers,
                pin_memory=self.hparams.pin_memory,
                collate_fn=collate_fn,
            )

    def train_dataloader(self) -> DataLoader:
        return self.build_dataloader(self.train_dataset, True)

    def val_dataloader(self) -> DataLoader:
        return self.build_dataloader(self.val_dataset, False)

    def test_dataloader(self) -> DataLoader:
        return self.build_dataloader(self.test_dataset, False)

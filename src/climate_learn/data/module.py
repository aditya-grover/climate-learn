# Standard library
from typing import Any, Callable, Dict, Sequence, Tuple, Union

# Third party
import numpy as np
from pytorch_lightning import LightningDataModule
import torch
from torch.utils.data import DataLoader, IterableDataset
from torchvision.transforms import transforms

# Local application
from ..utils.datetime import Year, Hours
from .dataset import (
    MapDatasetArgs,
    MapDataset,
    ShardDatasetArgs,
    ShardDataset,
)

# TODO: include exceptions in docstrings
# TODO: document legal input/output variables for each dataset

DatasetArgs = Union[MapDatasetArgs, ShardDatasetArgs]
Dataset = Union[MapDataset, ShardDataset]


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

    ## As a hotfix inp is just stacking input and constants data
    ## via {**inp_data, **const_data} i.e. merging both of them unto one dict
    inp = torch.stack(
        [
            handle_dict_features({**batch[i][0], **batch[i][2]})
            for i in range(len(batch))
        ]
    )
    out = torch.stack([handle_dict_features(batch[i][1]) for i in range(len(batch))])
    variables = list(batch[0][0].keys()) + list(batch[0][2].keys())
    out_variables = list(batch[0][1].keys())
    return inp, out, variables, out_variables


def get_data_class(dataset_args: DatasetArgs) -> Callable[..., Dataset]:
    if isinstance(dataset_args._data_class, str):
        return eval(dataset_args._data_class)
    else:
        return dataset_args._data_class


class DataModule(LightningDataModule):
    """ClimateLearn's data module interface. Encapsulates dataset/task-specific
    data modules."""

    def __init__(
        self,
        train_dataset_args: DatasetArgs,
        val_dataset_args: DatasetArgs,
        test_dataset_args: DatasetArgs,
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
        self.save_hyperparameters(logger=False)

        train_data_class = get_data_class(train_dataset_args)
        self.train_dataset: Dataset = train_data_class(train_dataset_args)

        val_data_class = get_data_class(val_dataset_args)
        self.val_dataset: Dataset = val_data_class(val_dataset_args)

        test_data_class = get_data_class(test_dataset_args)
        self.test_dataset: Dataset = test_data_class(test_dataset_args)

        self.train_dataset.setup()
        (
            inp_transforms,
            out_transforms,
            const_transforms,
        ) = self.train_dataset.get_transforms()

        self.val_dataset.setup()
        self.val_dataset.set_normalize(inp_transforms, out_transforms, const_transforms)

        self.test_dataset.setup()
        self.test_dataset.set_normalize(
            inp_transforms, out_transforms, const_transforms
        )

    def get_lat_lon(self) -> Tuple[Union[np.ndarray, None], Union[np.ndarray, None]]:
        ## Get rid of this in future; Use get_metadata() only
        train_metadata: Dict[str, Any] = self.get_metadata("train")
        #### TODO: Come up with better way to extract lat and lon
        key_list: Sequence[str] = list(train_metadata.keys())
        data_sources: Sequence[str] = list(
            dict.fromkeys([":".join(k.split(":")[:-1]) for k in key_list])
        )
        metadata_grouped_by_sources: Dict[str, Dict[str, Any]] = {}
        for data_source in data_sources:
            metadata_grouped_by_sources[data_source] = {}
        for key in train_metadata.keys():
            data_source: str = ":".join(key.split(":")[:-1])
            field_name: str = key.split(":")[-1]
            metadata_grouped_by_sources[data_source][field_name] = train_metadata[key]
        random_data_source: str = next(iter(metadata_grouped_by_sources.keys()))
        ### Bug prone, can have data sources which don't have lat and lon
        lat = metadata_grouped_by_sources[random_data_source]["lat"]
        lon = metadata_grouped_by_sources[random_data_source]["lon"]
        return lat, lon

    def get_out_transforms(self) -> Union[transforms.Normalize, None]:
        _, out_transforms, _ = self.train_dataset.get_transforms()
        return out_transforms

    def get_metadata(self, split: str = "train") -> Dict[str, Any]:
        if split == "train":
            return self.train_dataset.get_metadata()
        elif split == "val":
            return self.val_dataset.get_metadata()
        elif split == "test":
            return self.test_dataset.get_metadata()
        else:
            raise RuntimeError(f"Unrecognized split: {split}.")

    def get_climatology(
        self, split: str = "val"
    ) -> Union[Dict[str, torch.tensor], None]:
        if split == "train":
            return self.train_dataset.get_climatology()
        elif split == "val":
            return self.val_dataset.get_climatology()
        elif split == "test":
            return self.test_dataset.get_climatology()
        else:
            raise RuntimeError(f"Unrecognized split: {split}.")

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

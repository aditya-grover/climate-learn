import os
from typing import Optional

import numpy as np
import torch
import torchdata.datapipes as dp
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader, IterableDataset
from torchvision.transforms import transforms

from datamodules import VAR_LEVEL_TO_NAME_LEVEL

from .era5_iterdataset import (
    ERA5Forecast,
    ERA5ForecastMultiStep,
    ERA5Npy,
    IndividualForecastDataIter,
    ShuffleIterableDataset,
)


def collate_fn(batch):
    inp = torch.stack([batch[i][0] for i in range(len(batch))])
    out = torch.stack([batch[i][1] for i in range(len(batch))])
    variables = batch[0][2]
    out_variables = batch[0][3]
    return (
        inp,
        out,
        [VAR_LEVEL_TO_NAME_LEVEL[v] for v in variables],
        [VAR_LEVEL_TO_NAME_LEVEL[v] for v in out_variables],
    )


class ERA5IterDatasetModule(LightningDataModule):
    def __init__(
        self,
        root_dir,  # contains metadata and train + val + test
        variables,
        buffer_size,
        out_variables=None,
        predict_range: int = 6,  # only used for forecast
        predict_steps: int = 4,  # only used for forecast
        history: int = 3,  # used for forecast
        interval: int = 6,  # used for forecast and video
        subsample: int = 1,  # used for forecast
        batch_size: int = 64,
        num_workers: int = 0,
        pin_memory: bool = False,
    ):
        super().__init__()

        # this line allows to access init params with 'self.hparams' attribute
        self.save_hyperparameters(logger=False)

        self.lister_train = list(dp.iter.FileLister(os.path.join(root_dir, "train")))
        self.lister_val = list(dp.iter.FileLister(os.path.join(root_dir, "val")))
        self.lister_test = list(dp.iter.FileLister(os.path.join(root_dir, "test")))

        self.train_dataset_args = {
            "predict_range": predict_range,
            "history": history,
            "interval": interval,
            "subsample": subsample,
        }
        self.val_dataset_args = {
            "pred_range": predict_range,
            "pred_steps": predict_steps,
            "history": history,
            "interval": interval,
            "subsample": subsample,
        }

        self.transforms = self.get_normalize()
        self.output_transforms = self.get_normalize(out_variables)

        self.data_train: Optional[IterableDataset] = None
        self.data_val: Optional[IterableDataset] = None
        self.data_test: Optional[IterableDataset] = None

    def get_normalize(self, variables=None):
        if variables is None:
            variables = self.hparams.variables
        normalize_mean = dict(np.load(os.path.join(self.hparams.root_dir, "normalize_mean.npz")))
        mean = []
        for var in variables:
            if var != "tp":
                mean.append(normalize_mean[VAR_LEVEL_TO_NAME_LEVEL[var]])
            else:
                mean.append(np.array([0.0]))
        normalize_mean = np.concatenate(mean)
        normalize_std = dict(np.load(os.path.join(self.hparams.root_dir, "normalize_std.npz")))
        normalize_std = np.concatenate(
            [normalize_std[VAR_LEVEL_TO_NAME_LEVEL[var]] for var in variables]
        )
        return transforms.Normalize(normalize_mean, normalize_std)

    def get_lat_lon(self):
        lat = np.load(os.path.join(self.hparams.root_dir, "lat.npy"))
        lon = np.load(os.path.join(self.hparams.root_dir, "lon.npy"))
        return lat, lon

    def setup(self, stage: Optional[str] = None):
        # load datasets only if they're not loaded already
        if not self.data_train and not self.data_val and not self.data_test:
            self.data_train = ShuffleIterableDataset(
                IndividualForecastDataIter(
                    ERA5Forecast(
                        ERA5Npy(
                            self.lister_train,
                            variables=self.hparams.variables,
                            out_variables=self.hparams.out_variables,
                            shuffle=True,
                        ),
                        **self.train_dataset_args,
                    ),
                    self.transforms,
                    self.output_transforms,
                ),
                self.hparams.buffer_size,
            )

            if self.lister_val is not None:
                self.data_val = IndividualForecastDataIter(
                    ERA5ForecastMultiStep(
                        ERA5Npy(
                            self.lister_val,
                            variables=self.hparams.variables,
                            out_variables=self.hparams.out_variables,
                        ),
                        **self.val_dataset_args,
                    ),
                    self.transforms,
                    self.output_transforms,
                )

            if self.lister_test is not None:
                self.data_test = self.data_iter(
                    self.val_dataset_class(
                        self.reader(
                            self.lister_test,
                            variables=self.hparams.variables,
                            out_variables=self.hparams.out_variables,
                        ),
                        **self.val_dataset_args,
                    ),
                    self.transforms,
                    self.output_transforms,
                )

    def train_dataloader(self):
        return DataLoader(
            self.data_train,
            batch_size=self.hparams.batch_size,
            drop_last=True,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            collate_fn=collate_fn,
        )

    def val_dataloader(self):
        return DataLoader(
            self.data_val,
            batch_size=self.hparams.batch_size,
            shuffle=False,
            drop_last=True,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            collate_fn=collate_fn,
        )

    def test_dataloader(self):
        return DataLoader(
            self.data_test,
            batch_size=self.hparams.batch_size,
            shuffle=False,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            collate_fn=collate_fn,
        )

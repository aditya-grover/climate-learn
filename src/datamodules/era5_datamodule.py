import os
import numpy as np
from torch.utils.data import DataLoader
from pytorch_lightning import LightningDataModule

from era5_dataset import ERA5Forecast

class ERA5DataModule(LightningDataModule):
    def __init__(
        self,
        root_dir,
        inp_vars,
        out_vars,
        train_start_year,
        val_start_year,
        test_start_year,
        end_year: int = 2018,
        pred_range: int = 6,
        # predict_steps: int = 4,
        # subsample: int = 1,  # used for forecast
        batch_size: int = 64,
        num_workers: int = 0,
        pin_memory: bool = False,
    ):
        super().__init__()

        # this line allows to access init params with 'self.hparams' attribute
        self.save_hyperparameters(logger=False)

        train_years = range(train_start_year, val_start_year)
        self.train_dataset = ERA5Forecast(root_dir, inp_vars, out_vars, pred_range, train_years, 'train')

        val_years = range(val_start_year, test_start_year)
        self.val_dataset = ERA5Forecast(root_dir, inp_vars, out_vars, pred_range, val_years, 'val')
        self.val_dataset.set_normalize(self.train_dataset.inp_transform, self.train_dataset.out_transform)

        test_years = range(test_start_year, end_year + 1)
        self.test_dataset = ERA5Forecast(root_dir, inp_vars, out_vars, pred_range, test_years, 'test')
        self.test_dataset.set_normalize(self.train_dataset.inp_transform, self.train_dataset.out_transform)

    def get_lat_lon(self):
        lat = np.load(os.path.join(self.hparams.root_dir, "lat.npy"))
        lon = np.load(os.path.join(self.hparams.root_dir, "lon.npy"))
        return lat, lon

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            shuffle=True,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            shuffle=False,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            shuffle=False,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
        )

# era5_module = ERA5DataModule(
#     root_dir='/datadrive/datasets/5.625deg',
#     inp_vars=['2m_temperature', '10m_u_component_of_wind', '10m_v_component_of_wind'],
#     out_vars=['2m_temperature'],
#     train_start_year=1979,
#     val_start_year=2015,
#     test_start_year=2017,
#     end_year=2018,
#     pred_range=6,
#     batch_size=64,
#     num_workers=1,
#     pin_memory=False
# )
# train_loader = era5_module.train_dataloader()
# for x, y in train_loader:
#     print (x.shape)
#     print (y.shape)
#     break
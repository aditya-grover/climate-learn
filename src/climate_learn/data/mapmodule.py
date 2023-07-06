import os
import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from .npzdataset import NpzDataset


def collate_fn(batch):
    inp = torch.stack([batch[i][0] for i in range(len(batch))])
    out = torch.stack([batch[i][1] for i in range(len(batch))])
    out = F.pad(out, (2, 2, 3, 3))
    return inp, out, ["daily_tmax"], ["daily_tmax"]


class ERA5toPRISMDataModule(pl.LightningDataModule):
    def __init__(self, in_root_dir, out_root_dir, batch_size=32, num_workers=4):
        super().__init__()
        self.save_hyperparameters(logger=False)
        self.hparams.out_vars = ["daily_tmax"]
        self.hparams.history = 1
        self.hparams.task = "downscaling"

    def setup(self, stage="foobar"):
        self.train_dataset = NpzDataset(
            os.path.join(self.hparams.in_root_dir, "train.npz"),
            os.path.join(self.hparams.out_root_dir, "train.npz"),
        )
        self.in_transform = self.train_dataset.in_transform
        self.out_transform = self.train_dataset.out_transform
        self.val_dataset = NpzDataset(
            os.path.join(self.hparams.in_root_dir, "val.npz"),
            os.path.join(self.hparams.out_root_dir, "val.npz"),
            self.in_transform,
            self.out_transform,
        )
        self.test_dataset = NpzDataset(
            os.path.join(self.hparams.in_root_dir, "test.npz"),
            os.path.join(self.hparams.out_root_dir, "test.npz"),
            self.in_transform,
            self.out_transform,
        )
        self.out_mask = torch.from_numpy(
            np.load(os.path.join(self.hparams.out_root_dir, "mask.npy"))
        )
        with open(os.path.join(self.hparams.in_root_dir, "coords.npz"), "rb") as f:
            npz = np.load(f)
            self.in_lat = torch.from_numpy(npz["lat"])
            self.in_lon = torch.from_numpy(npz["lon"])
        with open(os.path.join(self.hparams.out_root_dir, "coords.npz"), "rb") as f:
            npz = np.load(f)
            self.out_lat = torch.from_numpy(npz["lat"])
            self.out_lon = torch.from_numpy(npz["lon"])

    def get_lat_lon(self):
        return self.out_lat, self.out_lon

    def get_data_dims(self):
        x, y = self.train_dataset[0]
        y = F.pad(y, (2, 2, 3, 3))
        return x.unsqueeze(0).shape, y.unsqueeze(0).shape

    def get_data_variables(self):
        return ["daily_tmax"], ["daily_tmax"]

    def get_climatology(self, split):
        if split == "train":
            return self.train_dataset.out_per_pixel_mean
        elif split == "val":
            return self.val_dataset.out_per_pixel_mean
        elif split == "test":
            return self.test_dataset.out_per_pixel_mean
        else:
            raise NotImplementedError()

    def get_out_transforms(self):
        return self.out_transform

    def get_out_mask(self):
        padded_mask = F.pad(self.out_mask, (2, 2, 3, 3))
        return padded_mask

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            collate_fn=collate_fn,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            collate_fn=collate_fn,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            collate_fn=collate_fn,
        )

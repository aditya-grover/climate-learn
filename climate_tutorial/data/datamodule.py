import torch
from torch.utils.data import DataLoader
from pytorch_lightning import LightningDataModule

from .datamodules import *
from ..utils import Year, Days, Hours

def collate_fn(batch):
    inp = torch.stack([batch[i][0] for i in range(len(batch))])
    out = torch.stack([batch[i][1] for i in range(len(batch))])
    variables = batch[0][2]
    out_variables = batch[0][3]
    return inp, out, variables, out_variables

class DataModule(LightningDataModule):
    def __init__(
        self,
        dataset,
        task,
        root_dir,
        inp_vars,
        out_vars,
        train_start_year,
        val_start_year,
        test_start_year,
        end_year = Year(2018),
        pred_range = Hours(6),
        subsample = Hours(1),
        batch_size = 64,
        num_workers = 0,
        pin_memory = False,
    ):
        super().__init__()

        assert end_year >= test_start_year and test_start_year > val_start_year and val_start_year > train_start_year
        self.save_hyperparameters(logger=False)

        if(dataset != "ERA5" or task != "forecasting"):
            raise NotImplementedError
            
        task_string = "Forecasting" if task == "forecasting" else "Downscaling"
        caller = eval(f"{dataset.upper()}{task_string}")
        
        train_years = range(train_start_year, val_start_year)
        self.train_dataset = caller(root_dir, inp_vars, out_vars, pred_range.hours, train_years, subsample.hours, "train")

        val_years = range(val_start_year, test_start_year)
        self.val_dataset = caller(root_dir, inp_vars, out_vars, pred_range.hours, val_years, subsample.hours, "val")
        self.val_dataset.set_normalize(self.train_dataset.inp_transform, self.train_dataset.out_transform)

        test_years = range(test_start_year, end_year + 1)
        self.test_dataset = caller(root_dir, inp_vars, out_vars, pred_range.hours, test_years, subsample.hours, "test")
        self.test_dataset.set_normalize(self.train_dataset.inp_transform, self.train_dataset.out_transform)

    def get_lat_lon(self):
        return self.train_dataset.lat, self.train_dataset.lon

    def get_out_transforms(self):
        return self.train_dataset.out_transform

    def get_climatology(self, split="val"):
        if split == "val":
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
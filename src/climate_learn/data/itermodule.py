# Standard library
import copy
import glob
import os
from typing import Dict, Optional

# Third party
import numpy as np
import torch
from torch.utils.data import DataLoader, IterableDataset
from torchvision.transforms import transforms
import pytorch_lightning as pl

# Local application
from .iterdataset import (
    NpyReader,
    DirectForecast,
    ContinuousForecast,
    Downscale,
    IndividualDataIter,
    ShuffleIterableDataset,
)


class IterDataModule(pl.LightningDataModule):
    """ClimateLearn's iter data module interface. Encapsulates dataset/task-specific
    data modules."""

    def __init__(
        self,
        task,
        inp_root_dir,
        out_root_dir,
        in_vars,
        out_vars,
        src=None,
        history=1,
        window=6,
        pred_range=6,
        random_lead_time=True,
        max_pred_range=120,
        hrs_each_step=1,
        subsample=1,
        buffer_size=10000,
        batch_size=64,
        num_workers=0,
        pin_memory=False,
    ):
        super().__init__()
        self.save_hyperparameters(logger=False)
        if task in ("direct-forecasting", "iterative-forecasting"):
            self.dataset_caller = DirectForecast
            self.dataset_arg = {
                "src": src,
                "pred_range": pred_range,
                "history": history,
                "window": window,
            }
            self.collate_fn = collate_fn
        elif task == "continuous-forecasting":
            self.dataset_caller = ContinuousForecast
            self.dataset_arg = {
                "random_lead_time": random_lead_time,
                "min_pred_range": pred_range,
                "max_pred_range": max_pred_range,
                "hrs_each_step": hrs_each_step,
                "history": history,
                "window": window,
            }
            self.collate_fn = collate_fn_continuous
        elif task == "downscaling":
            self.dataset_caller = Downscale
            self.dataset_arg = {}
            self.collate_fn = collate_fn

        self.inp_lister_train = sorted(
            glob.glob(os.path.join(inp_root_dir, "train", "*.npz"))
        )
        self.out_lister_train = sorted(
            glob.glob(os.path.join(out_root_dir, "train", "*.npz"))
        )
        self.inp_lister_val = sorted(
            glob.glob(os.path.join(inp_root_dir, "val", "*.npz"))
        )
        self.out_lister_val = sorted(
            glob.glob(os.path.join(out_root_dir, "val", "*.npz"))
        )
        self.inp_lister_test = sorted(
            glob.glob(os.path.join(inp_root_dir, "test", "*.npz"))
        )
        self.out_lister_test = sorted(
            glob.glob(os.path.join(out_root_dir, "test", "*.npz"))
        )

        self.transforms = self.get_normalize(inp_root_dir, in_vars)
        self.output_transforms = self.get_normalize(out_root_dir, out_vars)

        self.data_train: Optional[IterableDataset] = None
        self.data_val: Optional[IterableDataset] = None
        self.data_test: Optional[IterableDataset] = None

    def get_lat_lon(self):
        lat = np.load(os.path.join(self.hparams.out_root_dir, "lat.npy"))
        lon = np.load(os.path.join(self.hparams.out_root_dir, "lon.npy"))
        return lat, lon

    def get_data_variables(self):
        out_vars = copy.deepcopy(self.hparams.out_vars)
        if "2m_temperature_extreme_mask" in out_vars:
            out_vars.remove("2m_temperature_extreme_mask")
        return self.hparams.in_vars, out_vars

    def get_data_dims(self):
        lat = len(np.load(os.path.join(self.hparams.out_root_dir, "lat.npy")))
        lon = len(np.load(os.path.join(self.hparams.out_root_dir, "lon.npy")))
        forecasting_tasks = [
            "direct-forecasting",
            "iterative-forecasting",
            "continuous-forecasting",
        ]
        if self.hparams.task in forecasting_tasks:
            in_size = torch.Size(
                [
                    self.hparams.batch_size,
                    self.hparams.history,
                    len(self.hparams.in_vars),
                    lat,
                    lon,
                ]
            )
        elif self.hparams.task == "downscaling":
            in_size = torch.Size(
                [self.hparams.batch_size, len(self.hparams.in_vars), lat, lon]
            )
        ##TODO: change out size
        out_vars = copy.deepcopy(self.hparams.out_vars)
        if "2m_temperature_extreme_mask" in out_vars:
            out_vars.remove("2m_temperature_extreme_mask")
        out_size = torch.Size([self.hparams.batch_size, len(out_vars), lat, lon])
        return in_size, out_size

    def get_normalize(self, root_dir, variables):
        normalize_mean = dict(np.load(os.path.join(root_dir, "normalize_mean.npz")))
        normalize_std = dict(np.load(os.path.join(root_dir, "normalize_std.npz")))
        return {
            var: transforms.Normalize(normalize_mean[var][0], normalize_std[var][0])
            for var in variables
        }

    def get_out_transforms(self):
        out_transforms = {}
        for key in self.output_transforms.keys():
            if key == "2m_temperature_extreme_mask":
                continue
            out_transforms[key] = self.output_transforms[key]
        return out_transforms

    def get_climatology(self, split="val"):
        path = os.path.join(self.hparams.out_root_dir, split, "climatology.npz")
        clim_dict = np.load(path)
        new_clim_dict = {}
        for var in self.hparams.out_vars:
            if var == "2m_temperature_extreme_mask":
                continue
            new_clim_dict[var] = torch.from_numpy(
                np.squeeze(clim_dict[var].astype(np.float32), axis=0)
            )
        return new_clim_dict

    def setup(self, stage: Optional[str] = None):
        # load datasets only if they're not loaded already
        if stage != "test":
            if not self.data_train and not self.data_val and not self.data_test:
                self.data_train = ShuffleIterableDataset(
                    IndividualDataIter(
                        self.dataset_caller(
                            NpyReader(
                                inp_file_list=self.inp_lister_train,
                                out_file_list=self.out_lister_train,
                                variables=self.hparams.in_vars,
                                out_variables=self.hparams.out_vars,
                                shuffle=True,
                            ),
                            **self.dataset_arg,
                        ),
                        transforms=self.transforms,
                        output_transforms=self.output_transforms,
                        subsample=self.hparams.subsample,
                    ),
                    buffer_size=self.hparams.buffer_size,
                )

                self.data_val = IndividualDataIter(
                    self.dataset_caller(
                        NpyReader(
                            inp_file_list=self.inp_lister_val,
                            out_file_list=self.out_lister_val,
                            variables=self.hparams.in_vars,
                            out_variables=self.hparams.out_vars,
                            shuffle=False,
                        ),
                        **self.dataset_arg,
                    ),
                    transforms=self.transforms,
                    output_transforms=self.output_transforms,
                    subsample=self.hparams.subsample,
                )

                self.data_test = IndividualDataIter(
                    self.dataset_caller(
                        NpyReader(
                            inp_file_list=self.inp_lister_test,
                            out_file_list=self.out_lister_test,
                            variables=self.hparams.in_vars,
                            out_variables=self.hparams.out_vars,
                            shuffle=False,
                        ),
                        **self.dataset_arg,
                    ),
                    transforms=self.transforms,
                    output_transforms=self.output_transforms,
                    subsample=self.hparams.subsample,
                )
        else:
            self.data_test = IndividualDataIter(
                self.dataset_caller(
                    NpyReader(
                        inp_file_list=self.inp_lister_test,
                        out_file_list=self.out_lister_test,
                        variables=self.hparams.in_vars,
                        out_variables=self.hparams.out_vars,
                        shuffle=False,
                    ),
                    **self.dataset_arg,
                ),
                transforms=self.transforms,
                output_transforms=self.output_transforms,
                subsample=self.hparams.subsample,
            )

    def train_dataloader(self):
        return DataLoader(
            self.data_train,
            batch_size=self.hparams.batch_size,
            drop_last=False,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            collate_fn=self.collate_fn,
        )

    def val_dataloader(self):
        return DataLoader(
            self.data_val,
            batch_size=self.hparams.batch_size,
            shuffle=False,
            drop_last=False,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            collate_fn=self.collate_fn,
        )

    def test_dataloader(self):
        return DataLoader(
            self.data_test,
            batch_size=self.hparams.batch_size,
            shuffle=False,
            drop_last=False,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            collate_fn=self.collate_fn,
        )


def collate_fn(batch):
    def handle_dict_features(t: Dict[str, torch.tensor]) -> torch.tensor:
        t = torch.stack(tuple(t.values()))
        if len(t.size()) == 4:
            return torch.transpose(t, 0, 1)
        return t

    inp = torch.stack([handle_dict_features(batch[i][0]) for i in range(len(batch))])
    has_extreme_mask = False
    for key in batch[0][1]:
        if key == "2m_temperature_extreme_mask":
            has_extreme_mask = True
    if not has_extreme_mask:
        out = torch.stack(
            [handle_dict_features(batch[i][1]) for i in range(len(batch))]
        )
        variables = list(batch[0][0].keys())
        out_variables = list(batch[0][1].keys())
        return inp, out, variables, out_variables
    out = []
    mask = []
    for i in range(len(batch)):
        out_dict = {}
        mask_dict = {}
        for key in batch[i][1]:
            if key == "2m_temperature_extreme_mask":
                mask_dict[key] = batch[i][1][key]
            else:
                out_dict[key] = batch[i][1][key]
        out.append(handle_dict_features(out_dict))
        if mask_dict != {}:
            mask.append(handle_dict_features(mask_dict))
    out = torch.stack(out)
    if mask != []:
        mask = torch.stack(mask)
    variables = list(batch[0][0].keys())
    out_variables = list(out_dict.keys())
    return inp, out, mask, variables, out_variables


def collate_fn_continuous(batch):
    def handle_dict_features(t: Dict[str, torch.tensor]) -> torch.tensor:
        t = torch.stack(tuple(t.values()))
        if len(t.size()) == 4:
            return torch.transpose(t, 0, 1)
        return t

    inp = torch.stack([handle_dict_features(batch[i][0]) for i in range(len(batch))])
    out = torch.stack([handle_dict_features(batch[i][1]) for i in range(len(batch))])
    lead_times = torch.stack([batch[i][2] for i in range(len(batch))])
    b, t, _, h, w = inp.shape
    lead_times = lead_times.reshape(b, 1, 1, 1, 1).repeat(1, t, 1, h, w)
    inp = torch.cat((inp, lead_times), dim=2)
    variables = list(batch[0][0].keys())
    out_variables = list(batch[0][1].keys())
    return inp, out, variables, out_variables

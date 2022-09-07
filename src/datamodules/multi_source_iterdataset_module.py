import os
from typing import Dict, List, Optional

import numpy as np
import torch
import torchdata.datapipes as dp
from pytorch_lightning import LightningDataModule
from pytorch_lightning.trainer.supporters import CombinedLoader
from torch.utils.data import DataLoader, IterableDataset
from torchvision.transforms import transforms

from datamodules import VAR_LEVEL_TO_NAME_LEVEL

from .era5_iterdataset import (
    ERA5Npy,
    ERA5,
    ERA5Video,
    ERA5Forecast,
    IndividualDataIter,
    IndividualForecastDataIter,
    ShuffleIterableDataset,
)


def collate_fn(batch):
    inp = torch.stack([batch[i][0] for i in range(len(batch))])
    variables = batch[0][1]
    return inp, [VAR_LEVEL_TO_NAME_LEVEL[v] for v in variables]


def collate_forecast_fn(batch):
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


class MultiSourceTrainDatasetModule(LightningDataModule):
    def __init__(
        self,
        dataset_type: str,  # image, video, or forecast (finetune)
        dict_root_dirs: Dict,  # dict of root dirs
        dict_buffer_sizes: Dict, # dict of buffer sizes 
        dict_in_variables: Dict, # dict of lists of input variables
        dict_out_variables: Dict,
        dict_timesteps: Dict = {'mpi-esm': 2},  # only used for video
        dict_predict_ranges: Dict = {'mpi-esm': 72},  # only used for forecast
        dict_histories: Dict = {'mpi-esm': 3},  # used for forecast
        dict_intervals: Dict = {'mpi-esm': 6},  # used for forecast and video
        dict_subsamples: Dict = {'mpi-esm': 1},  # used for forecast
        batch_size: int = 64,
        num_workers: int = 0,
        pin_memory: bool = False,
    ):
        super().__init__()

        # this line allows to access init params with 'self.hparams' attribute
        self.save_hyperparameters(logger=False)

        out_variables = {}
        for k, list_out in dict_out_variables.items():
            if list_out is not None:
                out_variables[k] = list_out
            else:
                out_variables[k] = dict_in_variables[k]
        self.hparams.dict_out_variables = out_variables

        self.reader = ERA5Npy
        self.dict_lister_trains = {
            k: list(dp.iter.FileLister(os.path.join(root_dir, "train"))) for k, root_dir in dict_root_dirs.items()
        }

        if dataset_type == "image":
            self.train_dataset_class = ERA5
            self.train_dataset_args = {k: {} for k in dict_root_dirs.keys()}
            self.data_iter = IndividualDataIter
            self.collate_fn = collate_fn
        elif dataset_type == "video":
            self.train_dataset_class = ERA5Video
            self.train_dataset_args = {
                k: {"timesteps": dict_timesteps[k], "interval": dict_intervals[k]}
                for k in dict_root_dirs.keys()
            }
            self.data_iter = IndividualDataIter
            self.collate_fn = collate_fn
        elif dataset_type == "forecast":
            self.train_dataset_class = ERA5Forecast
            self.train_dataset_args = {
                k: {
                    "predict_range": dict_predict_ranges[k],
                    "history": dict_histories[k],
                    "interval": dict_intervals[k],
                    "subsample": dict_subsamples[k],
                } for k in dict_root_dirs.keys()
            }
            self.data_iter = IndividualForecastDataIter
            self.collate_fn = collate_forecast_fn
        else:
            raise NotImplementedError("Only support image, video, or forecast dataset")

        self.transforms = self.get_normalize()
        self.output_transforms = self.get_normalize(self.hparams.dict_out_variables)

        self.dict_data_train: Optional[Dict] = None

    def get_normalize(self, dict_variables: Optional[Dict] = None):
        if dict_variables is None:
            dict_variables = self.hparams.dict_in_variables
        dict_transforms = {}
        for k in dict_variables.keys():
            root_dir = self.hparams.dict_root_dirs[k]
            variables = dict_variables[k]
            normalize_mean = dict(np.load(os.path.join(root_dir, "normalize_mean.npz")))
            mean = []
            for var in variables:
                if var != "tp":
                    mean.append(normalize_mean[VAR_LEVEL_TO_NAME_LEVEL[var]])
                else:
                    mean.append(np.array([0.0]))
            normalize_mean = np.concatenate(mean)
            normalize_std = dict(np.load(os.path.join(root_dir, "normalize_std.npz")))
            normalize_std = np.concatenate(
                [normalize_std[VAR_LEVEL_TO_NAME_LEVEL[var]] for var in variables]
            )
            dict_transforms[k] = transforms.Normalize(normalize_mean, normalize_std)
        return dict_transforms

    def get_lat_lon(self):
        # assume different data sources have the same lat and lon coverage
        lat = np.load(os.path.join(list(self.hparams.dict_root_dirs.values())[0], "lat.npy"))
        lon = np.load(os.path.join(list(self.hparams.dict_root_dirs.values())[0], "lon.npy"))
        return lat, lon

    def setup(self, stage: Optional[str] = None):
        # load datasets only if they're not loaded already
        if not self.dict_data_train:
            dict_data_train = {}
            for k in self.dict_lister_trains.keys():
                lister_train = self.dict_lister_trains[k]
                variables = self.hparams.dict_in_variables[k]
                out_variables = self.hparams.dict_out_variables[k]
                dataset_args = self.train_dataset_args[k]
                transforms = self.transforms[k]
                output_transforms = self.output_transforms[k]
                buffer_size = self.hparams.dict_buffer_sizes[k]
                dict_data_train[k] = ShuffleIterableDataset(
                    self.data_iter(
                        self.train_dataset_class(
                            self.reader(
                                lister_train,
                                variables=variables,
                                out_variables=out_variables,
                                shuffle=True,
                            ),
                            **dataset_args,
                        ),
                        transforms,
                        output_transforms,
                    ),
                    buffer_size,
                )
            self.dict_data_train = dict_data_train

    def train_dataloader(self):
        loaders = {
            k: DataLoader(
                data_train,
                batch_size=self.hparams.batch_size,
                # shuffle=True,
                drop_last=True,
                num_workers=self.hparams.num_workers,
                pin_memory=self.hparams.pin_memory,
                collate_fn=self.collate_fn,
            ) for k, data_train in self.dict_data_train.items()
        }
        return CombinedLoader(loaders, mode="max_size_cycle")

# dataset_type = 'forecast'
# dict_root_dirs = {
#     'mpi-esm': '/datadrive/datasets/CMIP6/MPI-ESM/5.625deg_equally_np_all_levels',
#     'taiesm': '/datadrive/datasets/CMIP6/TaiESM1/5.625deg_equally_np_all_levels'
# }
# dict_buffer_sizes = {'mpi-esm': 1000, 'taiesm': 1000}
# dict_in_variables = {
#     'mpi-esm': ['t2m', 'z_500', 't_850'],
#     'taiesm': ['z_500', 't_850']
# }
# dict_out_variables = {
#     'mpi-esm': ['z_500', 't_850'],
#     'taiesm': ['z_500', 't_850']
# }
# dict_predict_ranges = {'mpi-esm': 12, 'taiesm': 12}
# dict_histories = {'mpi-esm': 1, 'taiesm': 1}
# dict_intervals = {'mpi-esm': 0, 'taiesm': 0}
# dict_subsamples = {'mpi-esm': 1, 'taiesm': 1}

# datamodule = MultiSourceTrainDatasetModule(
#     dataset_type,
#     dict_root_dirs,
#     dict_buffer_sizes,
#     dict_in_variables,
#     dict_out_variables,
#     dict_predict_ranges=dict_predict_ranges,
#     dict_histories=dict_histories,
#     dict_intervals=dict_intervals,
#     dict_subsamples=dict_subsamples,
#     batch_size=16,
#     num_workers=1,
#     pin_memory=False
# )
# datamodule.setup()
# dataloader = datamodule.train_dataloader()
# for batch in dataloader:
#     for k in batch.keys():
#         print (k)
#         x1, y1, in1, out1 = batch[k]
#         print (x1.shape)
#         print (y1.shape)
#         print (in1)
#         print (out1)
#     break

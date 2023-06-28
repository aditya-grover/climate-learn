# Standard library
import glob
import copy
from typing import Dict, Optional, Sequence, Tuple

# Third party
import torch
from torch.utils.data import DataLoader, IterableDataset
from torchvision.transforms import transforms
from pytorch_lightning import LightningDataModule

# Local application
from .climate_dataset.era5_iterdataset import *
from ..utils.datetime import Hours

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

    ## As a hotfix inp is just stacking input and constants data
    ## via {**inp_data, **const_data} i.e. merging both of them unto one dict
    inp = torch.stack([handle_dict_features(batch[i][0]) for i in range(len(batch))])
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
    # out = torch.stack([handle_dict_features(batch[i][1]) for i in range(len(batch))])
    variables = list(batch[0][0].keys())
    out_variables = list(out_dict.keys())
    return inp, out, mask, variables, out_variables


class IterDataModule(LightningDataModule):
    """ClimateLearn's iter data module interface. Encapsulates dataset/task-specific
    data modules."""

    def __init__(
        self,
        task,
        inp_root_dir,
        out_root_dir,
        in_vars,
        out_vars,
        history: int = 1,
        window: int = 6,
        pred_range=Hours(6),
        subsample=Hours(1),
        buffer_size=10000,
        batch_size=64,
        num_workers=0,
        pin_memory=False,
    ):
        r"""
        .. highlight:: python

        :param task: The name of the task. Currently supported options
            are: "forecasting", "downscaling".
        :type task: str
        :param inp_root_dir: The path to the local directory containing the
            specified input dataset.
        :type inp_root_dir: str
        :param out_root_dir: The path to the local directory containing the
            specified out dataset.
        :type out_root_dir: str
        :param in_vars: A list of input variables to use.
        :type in_vars: List[str]
        :param out_vars: A list of output variables to use.
        :type out_vars: List[str]
        """
        super().__init__()

        self.save_hyperparameters(logger=False)

        if task == "forecasting":
            # assert inp_root_dir == out_root_dir
            self.dataset_caller = Forecast
            self.dataset_arg = {
                "pred_range": pred_range.hours(),
                "history": history,
                "window": window,
            }
        else:  # downscaling
            self.dataset_caller = Downscale
            self.dataset_arg = {}

        self.inp_lister_train = glob.glob(os.path.join(inp_root_dir, "train", "*.npz"))
        self.inp_lister_train.sort()
        self.out_lister_train = glob.glob(os.path.join(out_root_dir, "train", "*.npz"))
        self.out_lister_train.sort()
        self.inp_lister_val = glob.glob(os.path.join(inp_root_dir, "val", "*.npz"))
        self.inp_lister_val.sort()
        self.out_lister_val = glob.glob(os.path.join(out_root_dir, "val", "*.npz"))
        self.out_lister_val.sort()
        self.inp_lister_test = glob.glob(os.path.join(inp_root_dir, "test", "*.npz"))
        self.inp_lister_test.sort()
        self.out_lister_test = glob.glob(os.path.join(out_root_dir, "test", "*.npz"))
        self.out_lister_test.sort()

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
        ##TODO: change out_vars
        out_vars = copy.deepcopy(self.hparams.out_vars)
        if "2m_temperature_extreme_mask" in out_vars:
            out_vars.remove("2m_temperature_extreme_mask")
        return self.hparams.in_vars, out_vars

    def get_data_dims(self):
        lat = len(np.load(os.path.join(self.hparams.out_root_dir, "lat.npy")))
        lon = len(np.load(os.path.join(self.hparams.out_root_dir, "lon.npy")))
        if self.hparams.task == "forecasting":
            in_size = torch.Size([self.hparams.batch_size, self.hparams.history, len(self.hparams.in_vars), lat, lon])
        else:
            in_size = torch.Size([self.hparams.batch_size, len(self.hparams.in_vars), lat, lon])
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
            new_clim_dict[var] = torch.from_numpy(np.squeeze(clim_dict[var].astype(np.float32), axis=0))
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
                        subsample=self.hparams.subsample.hours(),
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
                    subsample=self.hparams.subsample.hours(),
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
                    subsample=self.hparams.subsample.hours(),
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
                subsample=self.hparams.subsample.hours(),
            )

    def train_dataloader(self):
        return DataLoader(
            self.data_train,
            batch_size=self.hparams.batch_size,
            drop_last=False,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            collate_fn=collate_fn,
        )

    def val_dataloader(self):
        return DataLoader(
            self.data_val,
            batch_size=self.hparams.batch_size,
            shuffle=False,
            drop_last=False,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            collate_fn=collate_fn,
        )

    def test_dataloader(self):
        return DataLoader(
            self.data_test,
            batch_size=self.hparams.batch_size,
            shuffle=False,
            drop_last=False,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            collate_fn=collate_fn,
        )

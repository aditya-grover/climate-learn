# Standard library
import glob
from typing import Dict, Optional, Sequence, Tuple

# Third party
import torch
from torch.utils.data import DataLoader, IterableDataset
from torchvision.transforms import transforms
from pytorch_lightning import LightningDataModule

# Local application
from .climate_dataset.era5_continuous_iterdataset import *
from ..utils.datetime import Hours

# TODO: include exceptions in docstrings
# TODO: document legal input/output variables for each dataset


def collate_fn_continuous(
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
    inp = torch.stack([handle_dict_features(batch[i][0]) for i in range(len(batch))]) # B, T, C, H, W
    out = torch.stack([handle_dict_features(batch[i][1]) for i in range(len(batch))]) # B, C', H, W
    lead_times = torch.stack([batch[i][2] for i in range(len(batch))]) # B, 
    b, t, _, h, w = inp.shape
    # lead_times = lead_times.reshape(b, 1, 1, 1, 1).repeat(1, t, 1, h, w)
    # inp = torch.cat((inp, lead_times), dim=2)
    variables = list(batch[0][0].keys())
    out_variables = list(batch[0][1].keys())
    return inp, out, lead_times, variables, out_variables


class ContinuousIterDataModule(LightningDataModule):
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
        random_lead_time: bool = True,
        min_pred_range=Hours(6),
        max_pred_range=Hours(120),
        hrs_each_step=Hours(1),
        subsample=Hours(1),
        buffer_size=10000,
        batch_size=64,
        num_workers=0,
        pin_memory=False,
        fixed_lead_time_eval=None,
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

        assert inp_root_dir == out_root_dir
        self.dataset_caller = Forecast
        self.dataset_arg = {
            "random_lead_time": random_lead_time,
            "min_pred_range": min_pred_range.hours(),
            "max_pred_range": max_pred_range.hours(),
            "hrs_each_step": hrs_each_step.hours(),
            "history": history,
            "window": window,
        }

        if fixed_lead_time_eval is not None:
            self.eval_arg = {
                "random_lead_time": False,
                "min_pred_range": fixed_lead_time_eval // hrs_each_step.hours(),
                "max_pred_range": fixed_lead_time_eval // hrs_each_step_hours(),
                "hrs_each_step": hrs_each_step.hours(),
                "history": history,
                "window": window,
            }
        else:
            self.eval_arg = self.dataset_arg

        self.inp_lister_train = glob.glob(os.path.join(inp_root_dir, "train", "*.npz"))
        self.out_lister_train = glob.glob(os.path.join(out_root_dir, "train", "*.npz"))
        self.inp_lister_val = glob.glob(os.path.join(inp_root_dir, "val", "*.npz"))
        self.out_lister_val = glob.glob(os.path.join(out_root_dir, "val", "*.npz"))
        self.inp_lister_test = glob.glob(os.path.join(inp_root_dir, "test", "*.npz"))
        self.out_lister_test = glob.glob(os.path.join(out_root_dir, "test", "*.npz"))

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
        return self.hparams.in_vars, self.hparams.out_vars

    def get_data_dims(self):
        lat = len(np.load(os.path.join(self.hparams.out_root_dir, "lat.npy")))
        lon = len(np.load(os.path.join(self.hparams.out_root_dir, "lon.npy")))
        if self.hparams.task == "forecasting":
            in_size = torch.Size([self.hparams.batch_size, self.hparams.history, len(self.hparams.in_vars), lat, lon])
        else:
            in_size = torch.Size([self.hparams.batch_size, len(self.hparams.in_vars), lat, lon])
        out_size = torch.Size([self.hparams.batch_size, len(self.hparams.out_vars), lat, lon])
        return in_size, out_size

    def get_normalize(self, root_dir, variables):
        normalize_mean = dict(np.load(os.path.join(root_dir, "normalize_mean.npz")))
        normalize_std = dict(np.load(os.path.join(root_dir, "normalize_std.npz")))
        return {
            var: transforms.Normalize(normalize_mean[var][0], normalize_std[var][0])
            for var in variables
        }

    def get_out_transforms(self):
        return self.output_transforms

    def get_climatology(self, split="val"):
        path = os.path.join(self.hparams.out_root_dir, split, "climatology.npz")
        clim_dict = np.load(path)
        clim_dict = {
            var: torch.from_numpy(np.squeeze(clim_dict[var].astype(np.float32), axis=0))
            for var in self.hparams.out_vars
        }
        return clim_dict

    def setup(self, stage: Optional[str] = None):
        # load datasets only if they're not loaded already
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
                    **self.eval_arg,
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
                    **self.eval_arg,
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
            collate_fn=collate_fn_continuous,
        )

    def val_dataloader(self):
        return DataLoader(
            self.data_val,
            batch_size=self.hparams.batch_size,
            shuffle=False,
            drop_last=False,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            collate_fn=collate_fn_continuous,
        )

    def test_dataloader(self):
        return DataLoader(
            self.data_test,
            batch_size=self.hparams.batch_size,
            shuffle=False,
            drop_last=False,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            collate_fn=collate_fn_continuous,
        )

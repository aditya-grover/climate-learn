# Standard library
import glob
from typing import Optional

# Local application
from .climate_dataset.era5_iterdataset import *
from ..utils.datetime import Hours
from .module import collate_fn

# Third party
import torch
from torch.utils.data import DataLoader, IterableDataset
from torchvision.transforms import transforms
from pytorch_lightning import LightningDataModule

# TODO: include exceptions in docstrings
# TODO: document legal input/output variables for each dataset


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
            assert inp_root_dir == out_root_dir
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

    def get_normalize(self, root_dir, variables):
        normalize_mean = dict(np.load(os.path.join(root_dir, "normalize_mean.npz")))
        mean = []
        for var in variables:
            if var != "total_precipitation":
                mean.append(normalize_mean[var])
            else:
                mean.append(np.array([0.0]))
        normalize_mean = np.concatenate(mean)
        normalize_std = dict(np.load(os.path.join(root_dir, "normalize_std.npz")))
        normalize_std = np.concatenate([normalize_std[var] for var in variables])
        return transforms.Normalize(normalize_mean, normalize_std)

    def get_out_transforms(self):
        return self.output_transforms

    def get_climatology(self, split="val"):
        path = os.path.join(self.hparams.out_root_dir, split, "climatology.npz")
        clim_dict = np.load(path)
        clim = np.concatenate([clim_dict[var] for var in self.hparams.out_vars])
        clim = torch.from_numpy(clim)
        return clim

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

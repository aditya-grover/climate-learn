# Standard library
import glob
from typing import Dict, Optional, Sequence, Tuple

# Third party
import torch
from torch.utils.data import DataLoader, IterableDataset
from torchvision.transforms import transforms
from pytorch_lightning import LightningDataModule

# Local application
from .climate_dataset.era5_multisource_continuous_iterdataset import *
from .continuous_itermodule import collate_fn_continuous
from ..utils.datetime import Hours
from .climate_dataset.era5.constants import CONSTANTS

# TODO: include exceptions in docstrings
# TODO: document legal input/output variables for each dataset


class MultiSourcContinuouseDataModule(LightningDataModule):
    """ClimateLearn's iter data module interface. Encapsulates dataset/task-specific
    data modules."""

    def __init__(
        self,
        task,
        dict_root_dir: Dict,
        dict_start_idx: Dict,
        dict_end_idx: Dict,
        dict_in_variables: Dict,
        dict_out_variables: Dict,
        dict_random_lead_time: Dict = {"mpi-esm": True},
        dict_min_pred_range: Dict = {"mpi-esm": 1},
        dict_max_pred_range: Dict = {"mpi-esm": 28},
        dict_hrs_each_step: Dict = {"mpi-esm": 6},
        dict_subsample: Dict = {"mpi-esm": 1},
        dict_buffer_size: Dict = {"mpi-esm": 1000},
        history: int = 1,
        window: int = 0,
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

        if num_workers > 1:
            raise NotImplementedError(
                "num_workers > 1 is not supported yet. Performance will likely degrage too with larger num_workers."
            )
        # this line allows to access init params with 'self.hparams' attribute
        self.save_hyperparameters(logger=False)
        
        out_variables = {}
        for k, list_out in dict_out_variables.items():
            if list_out is not None:
                out_variables[k] = list_out
            else:
                out_variables[k] = dict_in_variables[k]
        self.hparams.dict_out_variables = out_variables
        
        self.dict_lister_trains = {
            k: glob.glob(os.path.join(root_dir, "train", "*.npz")) for k, root_dir in dict_root_dir.items()
        }

        self.transforms = self.get_normalize(self.hparams.dict_in_variables)
        self.output_transforms = self.get_normalize(self.hparams.dict_out_variables)

        self.dict_data_train: Optional[Dict] = None

    def get_lat_lon(self):
        lat = np.load(os.path.join(list(self.hparams.dict_root_dir.values())[0], "lat.npy"))
        lon = np.load(os.path.join(list(self.hparams.dict_root_dir.values())[0], "lon.npy"))
        return lat, lon
    
    def get_data_variables(self):
        return None, None

    def get_data_dims(self):
        lat, lon = self.get_lat_lon()
        lat, lon = len(lat), len(lon)
        if self.hparams.task == "forecasting":
            in_size = (self.hparams.batch_size, self.hparams.history, None, lat, lon)
        else:
            in_size = (self.hparams.batch_size, None, lat, lon)
        out_size = (self.hparams.batch_size, None, lat, lon)
        return in_size, out_size

    def get_normalize(self, dict_variables: Optional[Dict] = None):
        if dict_variables is None:
            dict_variables = self.hparams.dict_in_variables
        dict_transforms = {}
        for k in dict_variables.keys():
            root_dir = self.hparams.dict_root_dir[k]
            variables = dict_variables[k]
            normalize_mean = dict(np.load(os.path.join(root_dir, "normalize_mean.npz")))
            normalize_std = dict(np.load(os.path.join(root_dir, "normalize_std.npz")))
            dict_transforms[k] = {
                var: transforms.Normalize(normalize_mean[var][0], normalize_std[var][0])
                for var in variables
            }
        return dict_transforms

    def get_out_transforms(self):
        return None

    # def get_climatology(self, split="val"):
    #     path = os.path.join(self.hparams.out_root_dir, split, "climatology.npz")
    #     clim_dict = np.load(path)
    #     clim_dict = {
    #         var: torch.from_numpy(np.squeeze(clim_dict[var].astype(np.float32), axis=0))
    #         for var in self.hparams.out_vars if var in clim_dict.keys()
    #     }
    #     for var in self.hparams.out_vars:
    #         if var not in clim_dict.keys() and var in CONSTANTS:
    #             # just load the constant value from any npz data file
    #             constant_value = np.load(self.inp_lister_train[0])[var][0]
    #             clim_dict[var] = torch.from_numpy(np.squeeze(constant_value.astype(np.float32), axis=0))
    #     return clim_dict

    def setup(self, stage: Optional[str] = None):
        # load datasets only if they're not loaded already
        if not self.dict_data_train:
            dict_data_train = {}
            for k in self.dict_lister_trains.keys():
                lister_train = self.dict_lister_trains[k]
                start_idx = self.hparams.dict_start_idx[k]
                end_idx = self.hparams.dict_end_idx[k]
                variables = self.hparams.dict_in_variables[k]
                out_variables = self.hparams.dict_out_variables[k]
                min_pred_range = self.hparams.dict_min_pred_range[k]
                max_pred_range = self.hparams.dict_max_pred_range[k]
                random_lead_time = self.hparams.dict_random_lead_time[k]
                hrs_each_step = self.hparams.dict_hrs_each_step[k]
                transforms = self.transforms[k]
                output_transforms = self.output_transforms[k]
                buffer_size = self.hparams.dict_buffer_size[k]
                subsample = self.hparams.dict_subsample[k]
                
                # print ('k')
                # print ('start idx', start_idx)
                # print ('end idx', end_idx)
                # print ('variables', variables)
                # print ('min pred range', min_pred_range)
                # print ('max pred range', max_pred_range)
                # print ('random lead time', random_lead_time)
                # print ('hrs each step', hrs_each_step)
                # import sys
                # sys.exit()
                
                dict_data_train[k] = ShuffleIterableDataset(
                    IndividualDataIter(
                        Forecast(
                            NpyReader(
                                lister_train,
                                start_idx=start_idx,
                                end_idx=end_idx,
                                variables=variables,
                                out_variables=out_variables,
                                shuffle=True,
                                multi_dataset_training=True,
                            ),
                            min_pred_range=min_pred_range,
                            max_pred_range=max_pred_range,
                            random_lead_time=random_lead_time,
                            hrs_each_step=hrs_each_step,
                            history=self.hparams.history,
                            window=self.hparams.window
                        ),
                        transforms=transforms,
                        output_transforms=output_transforms,
                        subsample=subsample
                    ),
                    buffer_size,
                )
            self.dict_data_train = dict_data_train
            
    def train_dataloader(self):
        if not torch.distributed.is_initialized():
            raise NotImplementedError("Only support distributed training")
        else:
            node_rank = int(os.environ["NODE_RANK"])
            # assert that number of datasets is the same as number of nodes
            num_nodes = os.environ.get("NODES", None)
            if num_nodes is not None:
                num_nodes = int(num_nodes)
                assert num_nodes == len(self.dict_data_train.keys())

            for idx, k in enumerate(self.dict_data_train.keys()):
                if idx == node_rank:
                    data_train = self.dict_data_train[k]
                    break

        # This assumes that the number of datapoints are going to be the same for all datasets
        return DataLoader(
            data_train,
            batch_size=self.hparams.batch_size,
            drop_last=False,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            collate_fn=collate_fn_continuous,
        )

import torch
import numpy as np
import xarray as xr

from climate_learn.data_module.tasks.task import Task
from climate_learn.data_module.tasks.args import DownscalingArgs

class Downscaling(Task):
    args_class = DownscalingArgs

    def __init__(self, task_args):
        super.__init__(
            task_args.dataset_args, 
            task_args.in_vars, 
            task_args.constant_names, 
            task_args.out_vars, 
            task_args.subsample, 
            task_args.split
        )
        highres_dataset_class = task_args.highres_dataset_args.data_class
        self.highres_dataset = highres_dataset_class(task_args.highres_dataset_args)
        
        assert self.in_vars in self.dataset.variables
        assert self.out_vars in self.highres_dataset.variables
        assert self.constant_names in self.dataset.constant_names

    def setup(self):
        super().setup()
        self.highres_dataset.setup()
        inp_data = xr.concat([self.dataset.data_dict[k] for k in self.in_vars], dim="level")
        out_data = xr.concat(
            [self.highres_dataset.data_dict[k] for k in self.out_vars], dim="level"
        )

        self.inp_data = inp_data[::self.subsample].to_numpy().astype(np.float32)
        self.out_data = out_data[::self.subsample].to_numpy().astype(np.float32)

        constants_data = [
            self.dataset.constants[k].to_numpy().astype(np.float32)
            for k in self.constant_names
        ]
        if len(constants_data) > 0:
            self.constants_data = np.stack(constants_data, axis=0)  # 3, 32, 64
        else:
            self.constants_data = None

        assert len(self.inp_data) == len(self.out_data)

        # why is this a single number instead of a tuple
        self.downscale_ratio = self.out_data.shape[-1] // self.inp_data.shape[-1]

        if self.split == "train":
            self.inp_transform = self.get_normalize(self.inp_data)
            self.out_transform = self.get_normalize(self.out_data)
            self.constant_transform = (
                self.get_normalize(np.expand_dims(self.constants_data, axis=0))
                if self.constants_data is not None
                else None
            )
        else:
            self.inp_transform = None
            self.out_transform = None
            self.constant_transform = None

        self.time = self.dataset.data_dict[self.in_vars[0]].time.to_numpy()[::self.subsample].copy()
        self.inp_lon = self.dataset.lon
        self.inp_lat = self.dataset.lat
        self.out_lon = self.highres_dataset.lon
        self.out_lat = self.highres_dataset.lat

        del self.dataset.data_dict
        del self.highres_dataset.data_dict

    def get_climatology(self):
        return torch.from_numpy(self.out_data.mean(axis=0))

    def __getitem__(self, index):
        inp = torch.from_numpy(self.inp_data[index])
        out = torch.from_numpy(self.out_data[index])
        return (
            self.inp_transform(inp),
            self.out_transform(out),
            self.in_vars,
            self.out_vars,
        )

    def __len__(self):
        return len(self.inp_data)

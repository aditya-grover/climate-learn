import torch
import numpy as np
import xarray as xr

from climate_learn.data_module.tasks.task import Task
from climate_learn.data_module.tasks.args import ForecastingArgs

class Forecasting(Task):
    args_class = ForecastingArgs

    def __init__(self, task_args):
        super.__init__(
            task_args.dataset_args, 
            task_args.in_vars, 
            task_args.constant_names, 
            task_args.out_vars, 
            task_args.subsample, 
            task_args.split
        )
        assert self.in_vars in self.dataset.variables
        assert self.out_vars in self.dataset.variables
        assert self.constant_names in self.dataset.constant_names
        
        self.history = task_args.history
        self.window = task_args.window
        self.pred_range = task_args.pred_range

    def setup(self):
        super().setup()
        inp_data = xr.concat([self.dataset.data_dict[k] for k in self.in_vars], dim="level")
        out_data = xr.concat([self.dataset.data_dict[k] for k in self.out_vars], dim="level")
        self.inp_data = inp_data.to_numpy().astype(np.float32)
        self.out_data = out_data.to_numpy().astype(np.float32)

        constants_data = [
            self.dataset.constants[k].to_numpy().astype(np.float32)
            for k in self.constant_names
        ]
        if len(constants_data) > 0:
            self.constants_data = np.stack(constants_data, axis=0)  # 3, 32, 64
        else:
            self.constants_data = None

        assert len(self.inp_data) == len(self.out_data)

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

        self.time = (
            self.dataset.data_dict[self.in_vars[0]]
            .time.to_numpy()[:-self.pred_range:self.subsample]
            .copy()
        )
        # why do we need different lat and lan for input and output for foprecasting
        self.inp_lon = self.dataset.lon
        self.inp_lat = self.dataset.lat
        self.out_lon = self.dataset.lon
        self.out_lat = self.dataset.lat

        del self.dataset.data_dict

    
    def get_climatology(self):
        return torch.from_numpy(self.out_data.mean(axis=0))

    def create_inp_out(self, index):
        inp = []
        for i in range(self.history):
            idx = index + self.window * i
            inp.append(self.inp_data[idx])
        inp = np.stack(inp, axis=0)
        out_idx = index + (self.history - 1) * self.window + self.pred_range
        out = self.out_data[out_idx]
        return inp, out

    def __getitem__(self, index):
        inp, out = self.create_inp_out(index)
        out = self.out_transform(torch.from_numpy(out))  # C, 32, 64
        inp = self.inp_transform(torch.from_numpy(inp))  # T, C, 32, 64
        if self.constants_data is not None:
            constant = (
                torch.from_numpy(self.constants_data)
                .unsqueeze(0)
                .repeat(inp.shape[0], 1, 1, 1)
            )
            constant = self.constant_transform(constant)
            inp = torch.cat((inp, constant), dim=1)
        return inp, out, self.in_vars + self.constant_names, self.out_vars

    def __len__(self):
        return len(self.inp_data) - ((self.history - 1) * self.window + self.pred_range)
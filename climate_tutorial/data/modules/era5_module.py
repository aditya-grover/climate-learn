import os
import glob
import torch
import numpy as np
import xarray as xr

from tqdm.notebook import tqdm
from torch.utils.data import Dataset
from torchvision.transforms import transforms
from climate_tutorial.data import NAME_TO_VAR, DEFAULT_PRESSURE_LEVELS, CONSTANTS, SINGLE_LEVEL_VARS, PRESSURE_LEVEL_VARS


class ERA5(Dataset):
    def __init__(self, root_dir, root_highres_dir, variables, years, split = 'train'):
        super().__init__()
        self.root_dir = root_dir
        self.root_highres_dir = root_highres_dir
        self.variables = variables
        self.years = years
        self.split = split

        self.data_dict = self.load_from_nc(self.root_dir)
        if self.root_highres_dir is not None:
            self.data_highres_dict = self.load_from_nc(self.root_highres_dir)

        self.get_lat_lon()

    def load_from_nc(self, data_dir):
        constant_names = [name for name in self.variables if NAME_TO_VAR[name] in CONSTANTS]
        ps = glob.glob(os.path.join(data_dir, 'constants', '*.nc'))
        all_constants = xr.open_mfdataset(ps, combine='by_coords')
        self.constants = {name: all_constants[NAME_TO_VAR[name]] for name in constant_names}

        non_const_names = [name for name in self.variables if name not in constant_names]
        data_dict = {}
        for name in non_const_names:
            if name in SINGLE_LEVEL_VARS:
                data_dict[name] = []
            elif name in PRESSURE_LEVEL_VARS:
                for level in DEFAULT_PRESSURE_LEVELS:
                    data_dict[f'{name}_{level}'] = []
            else:
                raise NotImplementedError(f'{name} is not either in single-level or pressure-level dict')

        for year in tqdm(self.years):
            for var in non_const_names:
                dir_var = os.path.join(data_dir, var)
                ps = glob.glob(os.path.join(dir_var, f'*{year}*.nc'))
                xr_data = xr.open_mfdataset(ps, combine='by_coords')
                xr_data = xr_data[NAME_TO_VAR[var]]
                # np_data = xr_data.to_numpy()
                if len(xr_data.shape) == 3: # 8760, 32, 64
                    xr_data = xr_data.expand_dims(dim='level', axis=1)
                    data_dict[var].append(xr_data)
                else: # pressure level
                    for level in DEFAULT_PRESSURE_LEVELS:
                        xr_data_level = xr_data.sel(level=[level])
                        data_dict[f'{name}_{level}'].append(xr_data_level)
        
        data_dict = {k: xr.concat(data_dict[k], dim='time') for k in data_dict.keys()}
        # precipitation and solar radiation miss a few data points in the beginning
        len_min = min([data_dict[k].shape[0] for k in data_dict.keys()])
        data_dict = {k: data_dict[k][-len_min:] for k in data_dict.keys()}
        
        return data_dict

    def get_lat_lon(self):
        # lat lon is stored in each of the nc files, just need to load one and extract
        dir_var = os.path.join(self.root_dir, self.variables[0])
        year = self.years[0]
        ps = glob.glob(os.path.join(dir_var, f'*{year}*.nc'))
        xr_data = xr.open_mfdataset(ps, combine='by_coords')
        self.lat = xr_data['lat'].to_numpy()
        self.lon = xr_data['lon'].to_numpy()

    def __getitem__(self, index):
        pass

    def __len__(self):
        pass

class ERA5Forecasting(ERA5):
    def __init__(self, root_dir, root_highres_dir, in_vars, out_vars, history, window, pred_range, years, subsample=1, split='train'):
        print (f'Creating {split} dataset')
        super().__init__(root_dir, root_highres_dir, in_vars, years, split)
        
        self.in_vars = list(self.data_dict.keys())
        self.out_vars = out_vars
        self.history = history
        self.window = window
        self.pred_range = pred_range

        inp_data = xr.concat([self.data_dict[k] for k in self.in_vars], dim='level')
        out_data = xr.concat([self.data_dict[k] for k in self.out_vars], dim='level')

    self.inp_data = inp_data.to_numpy().astype(np.float32)
        self.out_data = out_data.to_numpy().astype(np.float32)

        constants_data = [self.constants[k].to_numpy().astype(np.float32) for k in self.constants.keys()]
        if len(constants_data) > 0:
            self.constants_data = np.stack(constants_data, axis=0) # 3, 32, 64
        else:
            self.constants_data = None

        assert len(self.inp_data) == len(self.out_data)

        self.downscale_ratio = 1

        if split == 'train':
            self.inp_transform = self.get_normalize(self.inp_data)
            self.out_transform = self.get_normalize(self.out_data)
            self.constant_transform = self.get_normalize(np.expand_dims(self.constants_data, axis=0)) if self.constants_data is not None else None
        else:
            self.inp_transform = None
            self.out_transform = None
            self.constant_transform = None

        del self.data_dict

    def get_normalize(self, data):
        mean = np.mean(data, axis=(0, 2, 3))
        std = np.std(data, axis=(0, 2, 3))
        return transforms.Normalize(mean, std)

    def set_normalize(self, inp_normalize, out_normalize, constant_normalize): # for val and test
        self.inp_transform = inp_normalize
        self.out_transform = out_normalize
        self.constant_transform = constant_normalize

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
        out = self.out_transform(torch.from_numpy(out)) # C, 32, 64
        inp = self.inp_transform(torch.from_numpy(inp)) # T, C, 32, 64
        if self.constants_data is not None:
            constant = torch.from_numpy(self.constants_data).unsqueeze(0).repeat(inp.shape[0], 1, 1, 1)
            constant = self.constant_transform(constant)
            inp = torch.cat((inp, constant), dim=1)
        return inp, out, self.in_vars + list(self.constants.keys()), self.out_vars

    def __len__(self):
        return len(self.inp_data) - ((self.history - 1) * self.window + self.pred_range)

class ERA5Downscaling(ERA5):
    def __init__(self, root_dir, root_highres_dir, in_vars, out_vars, pred_range, years, subsample=1, split='train'):
        print (f'Creating {split} dataset')
        super().__init__(root_dir, root_highres_dir, in_vars, years, split)
        
        self.in_vars = in_vars
        self.out_vars = out_vars
        self.pred_range = pred_range

        inp_data = xr.concat([self.data_dict[k] for k in in_vars], dim='level')
        out_data = xr.concat([self.data_highres_dict[k] for k in out_vars], dim='level')

        self.inp_data = inp_data[::subsample].to_numpy().astype(np.float32)
        self.out_data = out_data[::subsample].to_numpy().astype(np.float32)

        assert len(self.inp_data) == len(self.out_data)

        self.downscale_ratio = self.out_data.shape[-1] // self.inp_data.shape[-1]

        if split == 'train':
            self.inp_transform = self.get_normalize(self.inp_data)
            self.out_transform = self.get_normalize(self.out_data)
        else:
            self.inp_transform = None
            self.out_transform = None

        self.time = self.data_dict[in_vars[0]].time.to_numpy()[::subsample].copy()
        self.inp_lon = self.data_dict[in_vars[0]].lon.to_numpy().copy()
        self.inp_lat = self.data_dict[in_vars[0]].lat.to_numpy().copy()
        self.out_lon = self.data_highres_dict[out_vars[0]].lon.to_numpy().copy()
        self.out_lat = self.data_highres_dict[out_vars[0]].lat.to_numpy().copy()

        del self.data_dict
        del self.data_highres_dict

    def get_normalize(self, data):
        mean = np.mean(data, axis=(0, 2, 3))
        std = np.std(data, axis=(0, 2, 3))
        return transforms.Normalize(mean, std)

    def set_normalize(self, inp_normalize, out_normalize): # for val and test
        self.inp_transform = inp_normalize
        self.out_transform = out_normalize

    def get_climatology(self):
        return torch.from_numpy(self.out_data.mean(axis=0))

    def __getitem__(self, index):
        inp = torch.from_numpy(self.inp_data[index])
        out = torch.from_numpy(self.out_data[index])
        return self.inp_transform(inp), self.out_transform(out), self.in_vars, self.out_vars

    def __len__(self):
        return len(self.inp_data)

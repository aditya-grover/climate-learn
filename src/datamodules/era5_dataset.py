import os
import xarray as xr
import glob
from tqdm import tqdm

import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision.transforms import transforms

from src.datamodules import NAME_TO_VAR

class ERA5(Dataset):
    def __init__(self, root_dir, variables, years, partition='train'):
        super().__init__()
        self.root_dir = root_dir
        self.variables = variables
        self.years = years
        self.partition = partition

        self.load_from_nc()
        self.get_lat_lon()

    def load_from_nc(self):
        data_dict = {k: [] for k in self.variables}

        for year in tqdm(self.years):
            for var in self.variables:
                dir_var = os.path.join(self.root_dir, var)
                ps = glob.glob(os.path.join(dir_var, f'*{year}*.nc'))
                xr_data = xr.open_mfdataset(ps, combine='by_coords')
                xr_data = xr_data[NAME_TO_VAR[var]]
                np_data = xr_data.to_numpy()
                if len(np_data.shape) == 3: # 8760, 32, 64
                    np_data = np.expand_dims(np_data, axis=1)
                data_dict[var].append(np_data)
        
        data_dict = {k: np.concatenate(data_dict[k], axis=0) for k in self.variables}
        
        self.data_dict = data_dict

    def get_lat_lon(self):
        # lat lon is stored in each of the nc files, just need to load one and extract
        dir_var = os.path.join(self.root_dir, self.variables[0])
        year = self.years[0]
        ps = glob.glob(os.path.join(dir_var, f'*{year}*.nc'))
        xr_data = xr.open_mfdataset(ps, combine='by_coords')
        self.lat = xr_data['lat'].to_numpy()
        self.lon = xr_data['lon'].to_numpy()

    def __getitem__(self, index):
        np_data = np.concatenate([self.data_dict[var][index] for var in self.data_dict.keys()], axis=0)
        return torch.from_numpy(np_data)

    def __len__(self):
        return self.data_dict[list(self.data_dict.keys())[0]].shape[0]

class ERA5Forecast(ERA5):
    def __init__(self, root_dir, in_vars, out_vars, pred_range, years, partition='train'):
        print (f'Creating {partition} dataset from netCDF files')
        super().__init__(root_dir, in_vars, years, partition)
        
        self.in_vars = in_vars
        self.out_vars = out_vars
        self.pred_range = pred_range

        inp_data = np.concatenate([self.data_dict[k] for k in in_vars], axis=1)
        out_data = np.concatenate([self.data_dict[k] for k in out_vars], axis=1)

        self.inp_data = inp_data[0 : -pred_range]
        self.out_data = out_data[pred_range:]

        assert len(self.inp_data) == len(self.out_data)

        if partition == 'train':
            self.inp_transform = self.get_normalize(self.inp_data)
            self.out_transform = self.get_normalize(self.out_data)
        else:
            self.inp_transform = None
            self.out_transform = None

    def get_normalize(self, data):
        mean = np.mean(data, axis=(0, 2, 3))
        std = np.std(data, axis=(0, 2, 3))
        return transforms.Normalize(mean, std)

    def set_normalize(self, inp_normalize, out_normalize): # for val and test
        self.inp_transform = inp_normalize
        self.out_transform = out_normalize

    def __getitem__(self, index):
        inp = torch.from_numpy(self.inp_data[index])
        out = torch.from_numpy(self.out_data[index])
        return self.inp_transform(inp), self.out_transform(out), self.in_vars, self.out_vars

    def __len__(self):
        return len(self.inp_data)

# dataset = ERA5('/datadrive/datasets/5.625deg', ['2m_temperature', '10m_u_component_of_wind', '10m_v_component_of_wind', 'geopotential'], [1979, 1980])
# for k in dataset.data_dict.keys():
#     print (k)
#     print (dataset.data_dict[k].shape)
# x = dataset[0]
# print (x.shape)
# print (len(dataset))
# print (dataset.normalize_mean)
# print (dataset.normalize_std)

# dataset = ERA5Forecast('/datadrive/datasets/5.625deg', ['2m_temperature', '10m_u_component_of_wind', '10m_v_component_of_wind', 'geopotential'], ['2m_temperature'], 6, [1979, 1980], 'train')
# print (len(dataset))
# x, y = dataset[0]
# print (x.shape)
# print (y.shape)
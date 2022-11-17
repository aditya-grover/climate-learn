import os
import glob
import torch
import numpy as np
import xarray as xr

from tqdm.notebook import tqdm
from torch.utils.data import Dataset
from torchvision.transforms import transforms

class CMIP6(Dataset):
    def __init__(self, root_dir, root_highres_dir, variables, years, data_file, split = 'train'):
        super().__init__()
        self.root_dir = root_dir
        self.root_highres_dir = root_highres_dir
        self.variables = variables
        self.years = years
        self.split = split
        self.data_file = data_file

        self.data_dict = self.load_from_nc(self.data_file)
        if self.root_highres_dir is not None:
            self.data_highres_dict = self.load_from_nc(self.data_file)

        self.get_lat_lon()

    def load_from_nc(self, data_file):
        data_dict = {k: [] for k in self.variables}

        for var in self.variables:
            xr_data = data_file.where(data_file['time'].dt.year.isin(self.years), drop=True)
            
            if len(xr_data.shape) == 3: # 8760, 32, 64
                xr_data = xr_data.expand_dims(dim='plev', axis=1)
            data_dict[var].append(xr_data)
        
        data_dict = {k: xr.concat(data_dict[k], dim='time') for k in self.variables}
        
        return data_dict

    def get_lat_lon(self):
        xr_data = self.data_file
        self.lat = xr_data['lat'].to_numpy()
        self.lon = xr_data['lon'].to_numpy()

    def __getitem__(self, index):
        pass

    def __len__(self):
        pass

class CMIP6Forecasting(CMIP6):
    def __init__(self, root_dir, root_highres_dir, in_vars, out_vars, pred_range, years, data_file, subsample=1, split='train'):
        print (f'Creating {split} dataset')
        super().__init__(root_dir, root_highres_dir, in_vars, years, data_file, split)
        
        print(".1")
        self.in_vars = in_vars
        self.out_vars = out_vars
        self.pred_range = pred_range
        self.data_file = data_file

        inp_data = xr.concat([self.data_dict[k] for k in in_vars], dim='plev')
        out_data = xr.concat([self.data_dict[k] for k in out_vars], dim='plev')

        self.inp_data = inp_data[0:-pred_range:subsample].to_numpy().astype(np.float32)
        self.out_data = out_data[pred_range::subsample].to_numpy().astype(np.float32)

        assert len(self.inp_data) == len(self.out_data)

        self.downscale_ratio = 1

        print(".1")
        if split == 'train':
            self.inp_transform = self.get_normalize(self.inp_data)
            self.out_transform = self.get_normalize(self.out_data)
        else:
            self.inp_transform = None
            self.out_transform = None

        print(".1")
        self.time = self.data_dict[in_vars[0]].time.to_numpy()[:-pred_range:subsample].copy()
        self.inp_lon = self.data_dict[in_vars[0]].lon.to_numpy().copy()
        self.inp_lat = self.data_dict[in_vars[0]].lat.to_numpy().copy()
        self.out_lon = self.data_dict[out_vars[0]].lon.to_numpy().copy()
        self.out_lat = self.data_dict[out_vars[0]].lat.to_numpy().copy()

        del self.data_dict

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

class CMIP6Downscaling(CMIP6):
    def __init__(self, root_dir, root_highres_dir, in_vars, out_vars, pred_range, years, data_file, subsample=1, split='train'):
        print (f'Creating {split} dataset')
        super().__init__(root_dir, root_highres_dir, in_vars, years, data_file, split)
        
        self.in_vars = in_vars
        self.out_vars = out_vars
        self.pred_range = pred_range

        inp_data = xr.concat([self.data_dict[k] for k in in_vars], dim='plev')
        out_data = xr.concat([self.data_highres_dict[k] for k in out_vars], dim='plev')

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

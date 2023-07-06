import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision.transforms import transforms


class NpzDataset(Dataset):
    def __init__(
        self, npz_in_file, npz_out_file, in_transform=None, out_transform=None
    ):
        super().__init__()
        with open(npz_in_file, "rb") as f:
            npz = np.load(f)
            self.in_per_pixel_mean = torch.from_numpy(npz["mean"])
            self.in_per_pixel_std = torch.from_numpy(npz["std"])
            self.in_data = torch.from_numpy(npz["data"])
            self.in_data = self.in_data.unsqueeze(1)
            self.in_total_mean = np.nanmean(npz["data"])
            self.in_total_std = np.nanstd(npz["data"])
        with open(npz_out_file, "rb") as f:
            npz = np.load(f)
            self.out_per_pixel_mean = torch.from_numpy(npz["mean"])
            self.out_per_pixel_std = torch.from_numpy(npz["std"])
            self.out_data = torch.from_numpy(npz["data"])
            self.out_data = self.out_data.unsqueeze(1)
            self.out_total_mean = np.nanmean(npz["data"])
            self.out_total_std = np.nanstd(npz["data"])
        if in_transform is None:
            self.in_transform = transforms.Normalize(
                self.in_total_mean, self.in_total_std
            )
        else:
            self.in_transform = in_transform
        if out_transform is None:
            self.out_transform = transforms.Normalize(
                self.out_total_mean, self.out_total_std
            )
        else:
            self.out_transform = out_transform
        if len(self.in_data) != len(self.out_data):
            raise RuntimeError("length of input and output data do not match")

    def __len__(self):
        return len(self.in_data)

    def __getitem__(self, i):
        x = self.in_transform(self.in_data[i])
        y = self.out_transform(self.out_data[i])
        return x, y

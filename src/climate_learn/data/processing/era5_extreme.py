from argparse import ArgumentParser
import glob
import os

from ..climate_dataset import ERA5Args
from ..task import ForecastingArgs
from ..dataset import MapDataset, MapDatasetArgs

import torch
import numpy as np


parser = ArgumentParser(description="Generates the masks for ERA5 Extreme.")
parser.add_argument("source", help="The directory where the raw ERA5 data is stored.")
parser.add_argument(
    "source_npz", help="The directory where the processed ERA5 data is stored."
)
parser.add_argument("target", help="The directory to save the processed files.")
args = parser.parse_args()

era_args = ERA5Args(args.source, ["2m_temperature"], range(1979, 2015))
forecasting_args = ForecastingArgs(
    in_vars=["era5:2m_temperature"],
    out_vars=["era5:2m_temperature"],
    constants=[],
    history=1,
    window=1,
    pred_range=1,
)

map_dataset_args = MapDatasetArgs(era_args, forecasting_args)
map_dataset = MapDataset(map_dataset_args)
map_dataset.setup()

constants_data = map_dataset.data.get_constants_data()
const_data = map_dataset.task.create_constants_data(constants_data, 0)
data = []
for index in range(map_dataset.length):
    raw_index = map_dataset.task.get_raw_index(index)
    raw_data = map_dataset.data.get_item(raw_index)
    data.append(map_dataset.task.create_inp_out(raw_data, constants_data, 0))


def handle_dict_features(t):
    t = torch.stack(tuple(t.values()))
    if len(t.size()) == 4:
        return torch.transpose(t, 0, 1)
    return t


inp = torch.stack([handle_dict_features(data[i][0]) for i in range(len(data))])
out = torch.stack([handle_dict_features(data[i][1]) for i in range(len(data))])
if const_data != {}:
    const = handle_dict_features(const_data)
else:
    const = None

time_horizon = 7 * 24
window = 1
mean_tensor = []
for i in range(time_horizon, inp.size(0), window):
    mean_tensor.append(torch.mean(inp[i - time_horizon : i], dim=0))
mean_tensor = torch.stack(mean_tensor)

l_mean_tensor = torch.roll(mean_tensor, 1, -1)
r_mean_tensor = torch.roll(mean_tensor, -1, -1)
d_mean_tensor = torch.roll(mean_tensor, 1, -2)
u_mean_tensor = torch.roll(mean_tensor, -1, -2)

ld_mean_tensor = torch.roll(l_mean_tensor, 1, -2)
lu_mean_tensor = torch.roll(l_mean_tensor, -1, -2)
rd_mean_tensor = torch.roll(r_mean_tensor, 1, -2)
ru_mean_tensor = torch.roll(r_mean_tensor, -1, -2)

g_mean_tensor = 4 * mean_tensor
g_mean_tensor += l_mean_tensor + r_mean_tensor + d_mean_tensor + u_mean_tensor
g_mean_tensor += 0.25 * (
    ld_mean_tensor + lu_mean_tensor + rd_mean_tensor + ru_mean_tensor
)
g_mean_tensor = g_mean_tensor / 9

sorted_g_mean_tensor, sorted_args_g_mean_tensor = torch.sort(g_mean_tensor, dim=0)

low_percentile = 0.05
low_threshold_index = int(low_percentile * g_mean_tensor.size(0))

high_percentile = 0.95
high_threshold_index = int(high_percentile * g_mean_tensor.size(0))

low_threshold = sorted_g_mean_tensor[low_threshold_index].numpy()
high_threshold = sorted_g_mean_tensor[high_threshold_index].numpy()

low_threshold = np.squeeze(low_threshold, axis=0)
high_threshold = np.squeeze(high_threshold, axis=0)

file_list = glob.glob(os.path.join(args.source_npz, "*.npz"))
file_list = [f for f in file_list if "climatology" not in f]

years = list(range(2017, 2019))
file_list_by_years = [[] for _ in years]
for file_name in file_list:
    year = int((file_name.split("/")[-1]).split("_")[0])
    year_index = year - years[0]
    file_list_by_years[year_index].append(file_name)


def sort_func(file_name):
    index = int(((file_name.split("/")[-1]).split("_")[1]).split(".")[0])
    return index


for file_list_by_year in file_list_by_years:
    file_list_by_year.sort(key=sort_func, reverse=False)

time_horizon = 7 * 24
for file_list in file_list_by_years:
    yearly_data = {}
    n_instances_in_shard = 0
    for file in file_list:
        data = np.load(file)
        if yearly_data == {}:
            yearly_data = data
            random_key = next(iter(data.keys()))
            n_instances_in_shard = data[random_key].shape[0]
        else:
            yearly_data = {
                k: np.concatenate((yearly_data[k], data[k]), axis=0)
                for k in yearly_data.keys()
            }
            random_key = next(iter(data.keys()))
            assert n_instances_in_shard == data[random_key].shape[0]
    air_temp = yearly_data["2m_temperature"]
    mean_tensor = []
    for i in range(time_horizon, air_temp.shape[0]):
        curr_mean = np.mean(air_temp[i - time_horizon : i], axis=0)
        mean_tensor.append(curr_mean)
    mean_tensor = np.stack(mean_tensor, axis=0)

    l_mean_tensor = np.roll(mean_tensor, 1, -1)
    r_mean_tensor = np.roll(mean_tensor, -1, -1)
    d_mean_tensor = np.roll(mean_tensor, 1, -2)
    u_mean_tensor = np.roll(mean_tensor, -1, -2)

    ld_mean_tensor = np.roll(l_mean_tensor, 1, -2)
    lu_mean_tensor = np.roll(l_mean_tensor, -1, -2)
    rd_mean_tensor = np.roll(r_mean_tensor, 1, -2)
    ru_mean_tensor = np.roll(r_mean_tensor, -1, -2)

    g_mean_tensor = 4 * mean_tensor
    g_mean_tensor += l_mean_tensor + r_mean_tensor + d_mean_tensor + u_mean_tensor
    g_mean_tensor += 0.25 * (
        ld_mean_tensor + lu_mean_tensor + rd_mean_tensor + ru_mean_tensor
    )
    g_mean_tensor = g_mean_tensor / 9

    threshold_instances = np.zeros_like(air_temp[0], dtype=air_temp.dtype)
    air_temp_extreme_mask = np.zeros_like(air_temp, dtype=air_temp.dtype)
    for i in range(time_horizon, air_temp.shape[0]):
        curr_g_mean = g_mean_tensor[i - time_horizon]
        curr_mask = np.logical_or(
            curr_g_mean < low_threshold, curr_g_mean > high_threshold
        ).astype(air_temp.dtype)
        air_temp_extreme_mask[i] = curr_mask
        threshold_instances += curr_mask
    n_instances = np.min(threshold_instances)
    yearly_data["2m_temperature_extreme_mask"] = air_temp_extreme_mask

    for shard_id, file in enumerate(file_list):
        start_index = shard_id * n_instances_in_shard
        end_index = start_index + n_instances_in_shard
        new_file_name = os.path.join(args.target, file.split("/")[-1])
        sharded_data = {
            k: yearly_data[k][start_index:end_index] for k in yearly_data.keys()
        }
        np.savez(new_file_name, **sharded_data)

    print(
        air_temp_extreme_mask.sum(),
        air_temp.shape[0] * air_temp.shape[-1] * air_temp.shape[-2],
    )

newfile_list = glob.glob(os.path.join(args.target, "*.npz"))
newfile_list = [f for f in newfile_list if "climatology" not in f]
years = list(range(2017, 2019))
newfile_list_by_years = [[] for _ in years]
for file_name in newfile_list:
    year = int((file_name.split("/")[-1]).split("_")[0])
    year_index = year - years[0]
    newfile_list_by_years[year_index].append(file_name)
for newfile_list_by_year in newfile_list_by_years:
    newfile_list_by_year.sort(key=sort_func, reverse=False)

for newfile_list, file_list in zip(newfile_list_by_years, file_list_by_years):
    for new_file, file in zip(newfile_list, file_list):
        new_data = np.load(new_file)
        data = np.load(file)
        for k in new_data.keys():
            if k == "2m_temperature_extreme_mask":
                continue
            else:
                assert (new_data[k] == data[k]).all()

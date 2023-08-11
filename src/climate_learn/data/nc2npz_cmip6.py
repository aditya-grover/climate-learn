import glob
import os

import click
import numpy as np
import xarray as xr
from tqdm import tqdm

from climate_dataset.era5.constants import DEFAULT_PRESSURE_LEVELS, NAME_TO_VAR, CONSTANTS, VAR_TO_NAME

HOURS_PER_YEAR = 7300  # 7304 --> 7300 timesteps per file in CMIP6


def nc2np(path, variables, use_all_levels, years, save_dir, num_shards_per_year, partition):
    os.makedirs(os.path.join(save_dir, partition), exist_ok=True)
    normalize_mean = {}
    normalize_std = {}
    climatology = {}

    constants_path = os.path.join(path, "constants.nc")
    constants_are_downloaded = os.path.isfile(constants_path)
    yhours = 1460

    if constants_are_downloaded:
        constants = xr.open_mfdataset(
            constants_path, combine="by_coords", parallel=True
        )
        constant_fields = [VAR_TO_NAME[v] for v in CONSTANTS if v in VAR_TO_NAME.keys()]
        constant_values = {}
        for f in constant_fields:
            constant_values[f] = np.expand_dims(
                constants[NAME_TO_VAR[f]].to_numpy(), axis=(0, 1)
            ).repeat(yhours, axis=0)
            if partition == "train":
                normalize_mean[f] = constant_values[f].mean(axis=(0, 2, 3))
                normalize_std[f] = constant_values[f].std(axis=(0, 2, 3))

    for year in tqdm(years):
        np_vars = {}

        start_year = int(year.split('_')[0][:4])
        end_year = start_year + 5
        start_hours = 0
        for y in range(start_year, end_year):

            if partition == 'train':
                if y >= 2012:
                    continue
            elif partition == 'val':
                if y < 2012 or y > 2012:
                    continue
                start_hours = yhours*2
            elif partition == 'test':
                if y <= 2012:
                    continue
                if y == 2013:
                    start_hours = yhours*3 + 4
            else:
                print('Invalid partition')
                exit()
            end_hours = start_hours + yhours
            # constant variables
            if constants_are_downloaded:
                for f in constant_fields:
                    np_vars[f] = constant_values[f]


            for var in variables:
                print(var)
                code = NAME_TO_VAR[var]
                ps = glob.glob(os.path.join(path, var, f"*{year}*.nc"))
                ds = xr.open_mfdataset(ps, combine="by_coords", parallel=True)  # dataset for a single variable

                if len(ds[code].shape) == 3:  # surface level variables
                    ds[code] = ds[code].expand_dims("val", axis=1)
                    # remove the last 24 hours if this year has 366 days
                    # np_vars[code] = ds[code].to_numpy()[:HOURS_PER_YEAR]
                    np_vars[var] = ds[code][start_hours:end_hours].to_numpy()
                    
                    var_mean_yearly = np_vars[var].mean(axis=(0, 2, 3))
                    var_std_yearly = np_vars[var].std(axis=(0, 2, 3))

                    clim_yearly = np_vars[var].mean(axis=0)
                    if var not in climatology:
                        climatology[var] = [clim_yearly]
                    else:
                        climatology[var].append(clim_yearly)
                    if partition == 'train':
                        if var not in normalize_mean:
                            normalize_mean[var] = [var_mean_yearly]
                            normalize_std[var] = [var_std_yearly]
                        else:
                            normalize_mean[var].append(var_mean_yearly)
                            normalize_std[var].append(var_std_yearly)
                else:  # multiple-level variables, only use a subset
                    assert len(ds[code].shape) == 4
                    all_levels = ds["plev"][:].to_numpy() / 100  # 92500 --> 925
                    all_levels = all_levels.astype(int)
                    if use_all_levels:
                        all_levels = np.intersect1d(all_levels, ALL_LEVELS)
                    else:
                        all_levels = np.intersect1d(all_levels, DEFAULT_PRESSURE_LEVELS)
                    for level in all_levels:
                        ds_level = ds.sel(plev=[level * 100.0])
                        # level = int(level / 100) # 92500 --> 925

                        # remove the last 24 hours if this year has 366 days
                        # np_vars[f"{var}_{level}"] = ds_level[code].to_numpy()[:HOURS_PER_YEAR]
                        np_vars[f"{var}_{level}"] = ds_level[code][start_hours:end_hours].to_numpy()

                        var_mean_yearly = np_vars[f"{var}_{level}"].mean(axis=(0, 2, 3))
                        var_std_yearly = np_vars[f"{var}_{level}"].std(axis=(0, 2, 3))

                        clim_yearly = np_vars[f"{var}_{level}"].mean(axis=0)
                        if f"{var}_{level}" not in climatology:
                            climatology[f"{var}_{level}"] = [clim_yearly]
                        else:
                            climatology[f"{var}_{level}"].append(clim_yearly)

                        if partition == 'train':
                            if var not in normalize_mean:
                                normalize_mean[f"{var}_{level}"] = [var_mean_yearly]
                                normalize_std[f"{var}_{level}"] = [var_std_yearly]
                            else:
                                normalize_mean[f"{var}_{level}"].append(var_mean_yearly)
                                normalize_std[f"{var}_{level}"].append(var_std_yearly)


            # assert HOURS_PER_YEAR % num_shards_per_year == 0
            assert yhours % num_shards_per_year == 0
            # num_hrs_per_shard = HOURS_PER_YEAR // num_shards_per_year
            num_hrs_per_shard = yhours // num_shards_per_year
            for shard_id in range(num_shards_per_year):
                start_id = shard_id * num_hrs_per_shard
                end_id = start_id + num_hrs_per_shard
                sharded_data = {k: np_vars[k][start_id:end_id] for k in np_vars.keys()}
                np.savez(
                    # os.path.join(save_dir, "train", f"{year}_{shard_id}.npz"),
                    os.path.join(save_dir, partition, f'{y}_{shard_id}.npz'),
                    **sharded_data,
                )
            
            if y % 4 == 0 and y != 1900:
                start_hours = end_hours + 4
            else:
                start_hours = end_hours

    climatology = {k: np.mean(v, axis=0) for k, v in climatology.items()}
    np.savez(
        os.path.join(save_dir, partition, "climatology.npz"),
        **climatology,
    )

    if partition == 'train':
        for var in normalize_mean.keys():
            normalize_mean[var] = np.stack(normalize_mean[var], axis=0)
            normalize_std[var] = np.stack(normalize_std[var], axis=0)

        for var in normalize_mean.keys():  # aggregate over the years
            mean, std = normalize_mean[var], normalize_std[var]
            # var(X) = E[var(X|Y)] + var(E[X|Y])
            variance = (std**2).mean(axis=0) + (mean**2).mean(axis=0) - mean.mean(axis=0) ** 2
            std = np.sqrt(variance)
            # E[X] = E[E[X|Y]]
            mean = mean.mean(axis=0)
            normalize_mean[var] = mean
            normalize_std[var] = std

        np.savez(os.path.join(save_dir, "normalize_mean.npz"), **normalize_mean)
        np.savez(os.path.join(save_dir, "normalize_std.npz"), **normalize_std)


@click.command()
@click.argument("path", type=click.Path(exists=True))
@click.option(
    "--variables",
    "-v",
    type=click.STRING,
    multiple=True,
    default=[
        "2m_temperature",
        "10m_u_component_of_wind",
        "10m_v_component_of_wind",
        "geopotential",
        "u_component_of_wind",
        "v_component_of_wind",
        "temperature",
        # "relative_humidity",
        "specific_humidity",
        "toa_incident_solar_radiation",
    ],
)
@click.option("--all_levels", type=bool, default=False)
@click.option("--num_shards", type=int, default=10)
def main(
    path,
    variables,
    all_levels,
    num_shards,
):
    assert HOURS_PER_YEAR % num_shards == 0
    year_strings = [f"{y}01010600-{y+5}01010000" for y in range(1850, 2015, 5)]  # hard code for cmip6

    if all_levels:
        postfix = "_all_levels"
    else:
        postfix = ""

    if len(variables) <= 3:  # small dataset for testing new models
        yearly_datapath = os.path.join(os.path.dirname(path), f"{os.path.basename(path)}_equally_small_np" + postfix)
    else:
        yearly_datapath = os.path.join(os.path.dirname(path), f"{os.path.basename(path)}_equally_np" + postfix)
    yearly_datapath = os.path.join(os.path.dirname(path), f"1.40625_npz")
    os.makedirs(yearly_datapath, exist_ok=True)

    nc2np(path, variables, all_levels, year_strings, yearly_datapath, num_shards, "val")


if __name__ == "__main__":
    main()
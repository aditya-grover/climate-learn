import glob
import os

import click
import numpy as np
import xarray as xr
from tqdm import tqdm

from datamodules import ALL_LEVELS, DEFAULT_PRESSURE_LEVELS, NAME_TO_VAR

HOURS_PER_YEAR = 8760  # 365-day year


def nc2np(path, variables, use_all_levels, years, save_dir, partition, num_shards_per_year):
    os.makedirs(os.path.join(save_dir, partition), exist_ok=True)

    if partition == "train":
        normalize_mean = {}
        normalize_std = {}

    constants = xr.open_mfdataset(os.path.join(path, "constants.nc"), combine="by_coords", parallel=True)
    constant_fields = ["land_sea_mask", "orography", "lattitude"]
    constant_values = {}
    for f in constant_fields:
        constant_values[f] = np.expand_dims(constants[NAME_TO_VAR[f]].to_numpy(), axis=(0, 1)).repeat(
            HOURS_PER_YEAR, axis=0
        )
        if partition == "train":
            normalize_mean[f] = constant_values[f].mean(axis=(0, 2, 3))
            normalize_std[f] = constant_values[f].std(axis=(0, 2, 3))

    for year in tqdm(years):
        np_vars = {}

        # constant variables
        for f in constant_fields:
            np_vars[NAME_TO_VAR[f]] = constant_values[f]

        # non-constant fields
        for var in variables:
            ps = glob.glob(os.path.join(path, var, f"*{year}*.nc"))
            ds = xr.open_mfdataset(ps, combine="by_coords", parallel=True)  # dataset for a single variable
            code = NAME_TO_VAR[var]

            if len(ds[code].shape) == 3:  # surface level variables
                ds[code] = ds[code].expand_dims("val", axis=1)
                # remove the last 24 hours if this year has 366 days
                np_vars[code] = ds[code].to_numpy()[:HOURS_PER_YEAR]

                if partition == "train":  # compute mean and std of each var in each year
                    var_mean_yearly = np_vars[code].mean(axis=(0, 2, 3))
                    var_std_yearly = np_vars[code].std(axis=(0, 2, 3))
                    if var not in normalize_mean:
                        normalize_mean[var] = [var_mean_yearly]
                        normalize_std[var] = [var_std_yearly]
                    else:
                        normalize_mean[var].append(var_mean_yearly)
                        normalize_std[var].append(var_std_yearly)

            else:  # multiple-level variables, only use a subset
                assert len(ds[code].shape) == 4
                all_levels = ds["level"][:].to_numpy()
                if use_all_levels:
                    all_levels = np.intersect1d(all_levels, ALL_LEVELS)
                else:
                    all_levels = np.intersect1d(all_levels, DEFAULT_PRESSURE_LEVELS[code])
                for level in all_levels:
                    ds_level = ds.sel(level=[level])
                    level = int(level)
                    # remove the last 24 hours if this year has 366 days
                    np_vars[f"{code}_{level}"] = ds_level[code].to_numpy()[:HOURS_PER_YEAR]

                    if partition == "train":  # compute mean and std of each var in each year
                        var_mean_yearly = np_vars[f"{code}_{level}"].mean(axis=(0, 2, 3))
                        var_std_yearly = np_vars[f"{code}_{level}"].std(axis=(0, 2, 3))
                        if var not in normalize_mean:
                            normalize_mean[f"{var}_{level}"] = [var_mean_yearly]
                            normalize_std[f"{var}_{level}"] = [var_std_yearly]
                        else:
                            normalize_mean[f"{var}_{level}"].append(var_mean_yearly)
                            normalize_std[f"{var}_{level}"].append(var_std_yearly)

        assert HOURS_PER_YEAR % num_shards_per_year == 0
        num_hrs_per_shard = HOURS_PER_YEAR // num_shards_per_year
        for shard_id in range(num_shards_per_year):
            start_id = shard_id * num_hrs_per_shard
            end_id = start_id + num_hrs_per_shard
            sharded_data = {k: np_vars[k][start_id:end_id] for k in np_vars.keys()}
            np.savez(
                os.path.join(save_dir, partition, f"{year}_{shard_id}.npz"),
                **sharded_data,
            )

    if partition == "train":
        for var in normalize_mean.keys():
            if var not in constant_fields:
                normalize_mean[var] = np.stack(normalize_mean[var], axis=0)
                normalize_std[var] = np.stack(normalize_std[var], axis=0)

        for var in normalize_mean.keys():  # aggregate over the years
            if var not in constant_fields:
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
        "toa_incident_solar_radiation",
        "total_precipitation",
        "geopotential",
        "u_component_of_wind",
        "v_component_of_wind",
        "temperature",
        "relative_humidity",
        "specific_humidity",
    ],
)
@click.option("--all_levels", type=bool, default=True)
@click.option("--start_train_year", type=int, default=1979)
@click.option("--start_val_year", type=int, default=2013)
@click.option("--start_test_year", type=int, default=2015)
@click.option("--end_year", type=int, default=2017)
@click.option("--num_shards", type=int, default=12)
def main(
    path,
    variables,
    all_levels,
    start_train_year,
    start_val_year,
    start_test_year,
    end_year,
    num_shards,
):
    assert start_val_year > start_train_year and start_test_year > start_val_year and end_year > start_test_year
    train_years = range(start_train_year, start_val_year)
    val_years = range(start_val_year, start_test_year)
    test_years = range(start_test_year, end_year)

    if all_levels:
        postfix = "_all_levels"
    else:
        postfix = ""

    if len(variables) <= 3:  # small dataset for testing new models
        yearly_datapath = os.path.join(os.path.dirname(path), f"{os.path.basename(path)}_equally_small_np" + postfix)
    else:
        yearly_datapath = os.path.join(os.path.dirname(path), f"{os.path.basename(path)}_equally_np" + postfix)
    os.makedirs(yearly_datapath, exist_ok=True)

    nc2np(path, variables, all_levels, train_years, yearly_datapath, "train", num_shards)
    nc2np(path, variables, all_levels, val_years, yearly_datapath, "val", num_shards)
    nc2np(path, variables, all_levels, test_years, yearly_datapath, "test", num_shards)


if __name__ == "__main__":
    main()

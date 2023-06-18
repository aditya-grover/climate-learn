import os
from glob import glob

import click
import xarray as xr

from regrid import regrid

# x = xr.open_dataset('/datadrive/climate_bench/train_val/inputs_ssp126.nc')
# x['CO2'] = x['CO2'].expand_dims(dim={'latitude': 96, 'longitude': 144}, axis=(1,2))
# x['CH4'] = x['CH4'].expand_dims(dim={'latitude': 96, 'longitude': 144}, axis=(1,2))

# ddeg_out = 5.625

# y = regrid(x, ddeg_out)

# print (y)


@click.command()
@click.argument("path", type=click.Path(exists=True))
@click.option("--save_path", type=str)
@click.option("--ddeg_out", type=float, default=5.625)
def main(path, save_path, ddeg_out):
    if not os.path.exists(save_path):
        os.makedirs(save_path, exist_ok=True)

    list_simu = [
        "hist-GHG.nc",
        "hist-aer.nc",
        "historical.nc",
        "ssp126.nc",
        "ssp370.nc",
        "ssp585.nc",
        "ssp245.nc",
    ]
    ps = glob(os.path.join(path, f"*.nc"))
    ps_ = []
    for p in ps:
        for simu in list_simu:
            if simu in p:
                ps_.append(p)
    ps = ps_

    constant_vars = ["CO2", "CH4"]
    for p in ps:
        x = xr.open_dataset(p)
        if "input" in p:
            for v in constant_vars:
                x[v] = x[v].expand_dims(
                    dim={"latitude": 96, "longitude": 144}, axis=(1, 2)
                )
        x_regridded = regrid(x, ddeg_out, reuse_weights=False)
        x_regridded.to_netcdf(os.path.join(save_path, os.path.basename(p)))


if __name__ == "__main__":
    main()

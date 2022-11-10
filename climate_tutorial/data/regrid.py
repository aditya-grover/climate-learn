# copied from https://github.com/pangeo-data/WeatherBench/blob/master/src/regrid.py

import argparse
import xarray as xr
import numpy as np
import xesmf as xe
from glob import glob
import os
from pathlib import Path

def regrid(
        ds_in,
        dataset,
        resolution,
        method='bilinear',
        reuse_weights=True
):
    """
    Regrid horizontally.
    :param ds_in: Input xarray dataset
    :param resolution: Output resolution
    :param method: Regridding method
    :param reuse_weights: Reuse weights for regridding
    :return: ds_out: Regridded dataset
    """
    if dataset is "cmip6":
        print("dropped vars")
        ds_in = ds_in.drop_vars(['lon_bnds', 'lat_bnds'])
        if hasattr(ds_in, 'plev_bnds'):
            ds_in = ds_in.drop(('plev_bnds'))
        if hasattr(ds_in, 'time_bnds'):
            ds_in = ds_in.drop(('time_bnds'))
    

    if 'latitude' in ds_in.coords:
        ds_in = ds_in.rename({'latitude': 'lat', 'longitude': 'lon'})

    # Create output grid
    grid_out = xr.Dataset(
        {
            'lat': (['lat'], np.arange(-90+resolution/2, 90, resolution)),
            'lon': (['lon'], np.arange(0, 360, resolution)),
        }
    )

    # Create regridder
    regridder = xe.Regridder(
        ds_in, grid_out, method, periodic=True, reuse_weights=reuse_weights,
    )

    # Hack to speed up regridding of large files
    ds_list = []
    chunk_size = 500
    if hasattr(ds_in, 'time'):
        n_chunks = len(ds_in.time) // chunk_size + 1
        for i in range(n_chunks+1):
            ds_small = ds_in.isel(time=slice(i*chunk_size, (i+1)*chunk_size))
            ds_list.append(regridder(ds_small, keep_attrs=True).astype('float32'))
        ds_out = xr.concat(ds_list, dim='time')
    else:
        ds_out = regridder(ds_in, keep_attrs=True).astype('float32')

    # Set attributes since they get lost during regridding
    for var in ds_out:
        ds_out[var].attrs =  ds_in[var].attrs
    ds_out.attrs.update(ds_in.attrs)

    # # Regrid dataset
    # ds_out = regridder(ds_in)
    return ds_out


def regrider(
        root,
        source,
        dataset,
        variable,
        resolution,
        method='bilinear',
        reuse_weights=True,
        custom_fn=None,
        file_ending='nc',
        is_grib=False
):
    """
    :param root, source, dataset, variabe: Combined to get input files path
    :param resolution: Output resolution
    :param method: Regridding method
    :param reuse_weights: Reuse weights for regridding
    :param custom_fn: If not None, use custom file name. Otherwise infer from parameters.
    :param file_ending: Default = nc
    """
    input_fns = os.path.join(root, dataset, "pre-regrided", variable)
    output_dir = os.path.join(root, dataset, f"{resolution}deg", variable)

    print(f"Regridding {dataset} {variable} data in {input_fns} to {resolution}deg")

    # if(os.path.exists(output_dir)):
    #     raise Exception("Directory already exists")
    #     return
    # Make sure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    print(output_dir)

    # Get files for starred expressions
    # if '*' in input_fns[0]:
    #     input_fns = sorted(glob(input_fns[0]))

 
    # Loop over input files
    files = Path(input_fns).glob('*')
    for fn in files:
        print("\n hi")
        ds_in = xr.open_dataset(fn, engine='cfgrib') if is_grib else xr.open_dataset(fn)
        
        fn_out = (
            custom_fn or
            '_'.join(str(fn).split('/')[-1][:-3].split('_')[-1:]) + '_' + str(resolution) + 'deg.' + file_ending
        )
        if os.path.exists(output_dir + '/' + fn_out):
            print(output_dir + '/' + fn_out + " already generated")
            continue
        ds_out = regrid(ds_in, dataset, float(resolution), method, reuse_weights)


        print(f"Saving file: {output_dir + '/' + fn_out}")
        ds_out.to_netcdf(output_dir + '/' + fn_out)
        ds_in.close(); ds_out.close()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--root',
        type=str,
        nargs='+',
        help="Root of dataset directory",
        required=True
    )
    parser.add_argument(
        '--source',
        type=str,
        nargs='+',
        help="Source of dataset",
        required=True
    )
    parser.add_argument(
        '--dataset',
        type=str,
        help="Datset to regrid",
        required=True
    )
    parser.add_argument(
        '--variable',
        type=str,
        help="Variable to regrid",
        required=True
    )
    parser.add_argument(
        '--resolution',
        type=str,
        help="Output resolution",
        required=True
    )
    parser.add_argument(
        '--reuse_weights',
        type=int,
        help="Reuse weights for regridding. 0 or 1 (default)",
        default=1  
    )
    parser.add_argument(
        '--custom_fn',
        type=str,
        help="If not None, use custom file name. Otherwise infer from parameters.",
        default=None
    )
    parser.add_argument(
        '--file_ending',
        type=str,
        help="File ending. Default = nc",
        default='nc'
    )
    parser.add_argument(
        '--is_grib',
        type=int,
        help="Input is .grib file. 0 (default) or 1",
        default=0
    )
    args = parser.parse_args()

    regrider(
        root=args.root,
        source=args.source,
        dataset=args.dataset,
        variable=args.variable,
        resolution=args.resolution,
        reuse_weights=args.reuse_weights,
        custom_fn=args.custom_fn,
        file_ending=args.file_ending,
        is_grib=args.is_grib
    )

if(__name__ == "__main__"):
    main()
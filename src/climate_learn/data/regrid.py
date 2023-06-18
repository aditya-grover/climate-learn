import argparse
import xarray as xr
import numpy as np
import xesmf as xe
from glob import glob
import os


def regrid(
    ds_in, ddeg_out, method="bilinear", reuse_weights=True, cmip=False, rename=None
):
    """
    Regrid horizontally.
    :param ds_in: Input xarray dataset
    :param ddeg_out: Output resolution
    :param method: Regridding method
    :param reuse_weights: Reuse weights for regridding
    :return: ds_out: Regridded dataset
    """
    # import pdb; pdb.set_trace()
    # Rename to ESMF compatible coordinates
    if "latitude" in ds_in.coords:
        ds_in = ds_in.rename({"latitude": "lat", "longitude": "lon"})
    if cmip:
        ds_in = ds_in.drop(("lat_bnds", "lon_bnds"))
        if hasattr(ds_in, "plev_bnds"):
            ds_in = ds_in.drop(("plev_bnds"))
        if hasattr(ds_in, "time_bnds"):
            ds_in = ds_in.drop(("time_bnds"))
    if rename is not None:
        ds_in = ds_in.rename({rename[0]: rename[1]})

    # Create output grid
    grid_out = xr.Dataset(
        {
            "lat": (["lat"], np.arange(-90 + ddeg_out / 2, 90, ddeg_out)),
            "lon": (["lon"], np.arange(0, 360, ddeg_out)),
        }
    )

    # Create regridder
    regridder = xe.Regridder(
        ds_in, grid_out, method, periodic=True, reuse_weights=reuse_weights
    )

    # Hack to speed up regridding of large files
    ds_list = []
    chunk_size = 500
    # if hasattr(ds_in, 'time'):
    #     n_chunks = len(ds_in.time) // chunk_size + 1
    #     print ('n chunks', n_chunks)
    #     for i in range(n_chunks+1):
    #         print (i)
    #         ds_small = ds_in.isel(time=slice(i*chunk_size, (i+1)*chunk_size))
    #         print (ds_small)
    #         print ('regridding')
    #         ds_list.append(regridder(ds_small, keep_attrs=True).astype('float32'))
    #         print ('done')
    #     ds_out = xr.concat(ds_list, dim='time')
    # else:
    ds_out = regridder(ds_in, keep_attrs=True).astype("float32")

    # # Set attributes since they get lost during regridding
    # for var in ds_out:
    #     ds_out[var].attrs =  ds_in[var].attrs
    # ds_out.attrs.update(ds_in.attrs)

    if rename is not None:
        if rename[0] == "zg":
            ds_out["z"] *= 9.807
        if rename[0] == "rsdt":
            ds_out["tisr"] *= 60 * 60
            ds_out = ds_out.isel(time=slice(1, None, 12))
            ds_out = ds_out.assign_coords(
                {"time": ds_out.time + np.timedelta64(90, "m")}
            )

    # # Regrid dataset
    # ds_out = regridder(ds_in)
    return ds_out


def main(
    input_fns,
    output_dir,
    ddeg_out,
    method="bilinear",
    reuse_weights=True,
    custom_fn=None,
    file_ending="nc",
    cmip=False,
    rename=None,
):
    """
    :param input_fns: Input files. Can use *. If more than one, loop over them
    :param output_dir: Output directory
    :param ddeg_out: Output resolution
    :param method: Regridding method
    :param reuse_weights: Reuse weights for regridding
    :param custom_fn: If not None, use custom file name. Otherwise infer from parameters.
    :param file_ending: Default = nc
    """

    # Make sure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    # Get files for starred expressions
    if "*" in input_fns[0]:
        input_fns = sorted(glob(input_fns[0]))
    # Loop over input files
    for fn in input_fns:
        print(f"Regridding file: {fn}")
        ds_in = xr.open_dataset(fn)
        ds_out = regrid(ds_in, ddeg_out, method, reuse_weights, cmip, rename)
        fn_out = (
            custom_fn
            or "_".join(fn.split("/")[-1][:-3].split("_")[:-1])
            + "_"
            + str(ddeg_out)
            + "deg."
            + file_ending
        )
        print(f"Saving file: {output_dir + '/' + fn_out}")
        ds_out.to_netcdf(output_dir + "/" + fn_out)
        ds_in.close()
        ds_out.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_fns",
        type=str,
        nargs="+",
        help="Input files (full path). Can use *. If more than one, loop over them",
        required=True,
    )
    parser.add_argument(
        "--output_dir", type=str, help="Output directory", required=True
    )
    parser.add_argument(
        "--ddeg_out", type=float, help="Output resolution", required=True
    )
    parser.add_argument(
        "--reuse_weights",
        type=int,
        help="Reuse weights for regridding. 0 or 1 (default)",
        default=1,
    )
    parser.add_argument(
        "--custom_fn",
        type=str,
        help="If not None, use custom file name. Otherwise infer from parameters.",
        default=None,
    )
    parser.add_argument(
        "--file_ending", type=str, help="File ending. Default = nc", default="nc"
    )
    parser.add_argument(
        "--cmip", type=int, help="Is CMIP data. 0 or 1 (default)", default=0
    )
    parser.add_argument(
        "--rename", type=str, nargs="+", help="Rename var in dataset", default=None
    )
    args = parser.parse_args()

    main(
        input_fns=args.input_fns,
        output_dir=args.output_dir,
        ddeg_out=args.ddeg_out,
        reuse_weights=args.reuse_weights,
        custom_fn=args.custom_fn,
        file_ending=args.file_ending,
        cmip=args.cmip,
        rename=args.rename,
    )

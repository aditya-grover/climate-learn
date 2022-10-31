import os
import cdsapi
import argparse
import subprocess

NAME_TO_CMIP = {
    "geopotential": "zg",
    "u_component_of_wind": "ua",
    "v_component_of_wind": "va",
    "temperature": "ta",
    "relative_humidity": "r",
    "specific_humidity": "hus",
    "geopotential_500": "z",
}

months = [str(i).rjust(2, "0") for i in range(1, 13)]
days = [str(i).rjust(2, "0") for i in range(1, 32)]
times = [str(i).rjust(2, "0") + ":00" for i in range(0, 24)]


def _download_copernicus(root, dataset, variable, year, pressure = False, api_key = None):
    if(dataset not in ["era5"]):
        raise Exception("Dataset not supported")

    if(api_key is not None):
        content = f"url: https://cds.climate.copernicus.eu/api/v2\nkey: {api_key}"
        open(f"{os.environ['HOME']}/.cdsapirc", "w").write(content)

    path = os.path.join(root, dataset, variable, f"{variable}_{year}_0.25deg.nc")
    print(f"Downloading {dataset} {variable} data for year {year} from copernicus to {path}")
    if(os.path.exists(path)):
        return
    os.makedirs(os.path.dirname(path), exist_ok = True)

    download_args = {
        "product_type": "reanalysis",
        "format": "netcdf",
        "variable": variable,
        "year": str(year),
        "month": months,
        "day": days,
        "time": times,
    }

    client = cdsapi.Client()

    if not pressure:
        client.retrieve(
            "reanalysis-era5-single-levels",
            download_args,
            path,
        )
    else:
        download_args["pressure_level"] = [1000, 850, 500, 50]
        client.retrieve(
            "reanalysis-era5-pressure-levels",
            download_args,
            path,
        )

def _download_esgf(root, dataset, variable, resolution = "1.40625"):
    if (dataset not in ["cmip6"]):
        raise Exception("Dataset not supported")

    path = os.path.join(root, dataset, resolution, variable)
    print(f"Downloading {dataset} {variable} data for {resolution} resolution from weatherbench to {path}")
    if(os.path.exists(path)):
        raise Exception("Directory already exists")
        return
    os.makedirs(os.path.dirname(path), exist_ok = True)


    year_strings = [f'{y}01010600-{y+5}01010000' for y in range(1850, 2015, 5)]
    for yr in year_strings:
        url = (
            "https://esgf-data1.llnl.gov/thredds/fileServer/css03_data/CMIP6/CMIP/MPI-M/MPI-ESM1-2-HR/historical/r1i1p1f1/6hrPlevPt/"
            "/{variable}/gn/v20190815/{variable}_6hrPlevPt_MPI-ESM1-2-HR_historical_r1i1p1f1_gn_{yr_string}.nc"
        ).format(yr_string = yr, variable = NAME_TO_CMIP[variable])

        subprocess.check_call(["wget", "--no-check-certificate", url, "-P", path])
    # TODO: regrid to resolution
    # https://github.com/pangeo-data/WeatherBench/blob/master/snakemake_configs_CMIP/MPI-ESM/Snakefile
    # https://github.com/pangeo-data/WeatherBench/blob/master/src/regrid.py


def _download_weatherbench(root, dataset, variable, resolution = "1.40625"):
    if(dataset not in ["era5", "cmip6"]):
        raise Exception("Dataset not supported")

    path = os.path.join(root, dataset, resolution, variable)
    print(f"Downloading {dataset} {variable} data for {resolution} resolution from weatherbench to {path}")
    if(os.path.exists(path)):
        return
    os.makedirs(os.path.dirname(path), exist_ok = True)

    if(dataset == "era5"):
        url = (
            "https://dataserv.ub.tum.de/s/m1524895"
            "/download?path=%2F{resolution}deg%2F{variable}&files={variable}_{resolution}deg.zip"
        ).format(resolution = resolution, variable = variable)
    elif(dataset == "cmip6"):
        url = (
            "https://dataserv.ub.tum.de/s/m1524895"
            "/download?path=%2FCMIP%2FMPI-ESM%2F{resolution}deg%2F{variable}&files={variable}_{resolution}deg.zip"
        ).format(resolution = resolution, variable = variable)
    
    subprocess.check_call(["wget", "--no-check-certificate", url, "-O", path + ".zip"])
    subprocess.check_call(["unzip", path + ".zip", "-d", path])

def download(source, **kwargs):
    if("root" not in kwargs or kwargs["root"] is None):
        kwargs["root"] = ".climate_tutorial"

    kwargs["root"] = os.path.join(kwargs["root"], f"data/{source}")

    if(source == "copernicus"):
        _download_copernicus(**kwargs)
    elif(source == "weatherbench"):
        _download_weatherbench(**kwargs)
    elif(source == "esgf"):
        _download_esgf(**kwargs)

def main():
    parser = argparse.ArgumentParser()

    subparsers = parser.add_subparsers(dest = "source")

    subparser = subparsers.add_parser("copernicus")
    subparser.add_argument("--root", type = str, default = None)
    subparser.add_argument("--variable", type = str, required = True)
    subparser.add_argument("--dataset", type = str, choices = ["era5"], required = True)
    subparser.add_argument("--year", type = int, required = True)
    subparser.add_argument("--pressure", action = "store_true", default = False)
    subparser.add_argument("--api_key", type = str, default = None)

    subparser = subparsers.add_parser("weatherbench")
    subparser.add_argument("--root", type = str, default = None)
    subparser.add_argument("--variable", type = str, required = True)
    subparser.add_argument("--dataset", type = str, choices = ["era5", "cmip6"], required = True)
    subparser.add_argument("--resolution", type = str, default = "1.40625")

    args = parser.parse_args()
    download(**vars(args))

if(__name__ == "__main__"):
    main()

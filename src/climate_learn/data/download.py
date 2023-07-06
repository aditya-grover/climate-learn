# Standard library
from argparse import ArgumentParser
from ftplib import FTP
import os
import re
import requests
from zipfile import ZipFile

# Third party
import cdsapi
from tqdm import tqdm, trange


def download_copernicus_era5(dst, variable, year, pressure=False, api_key=None):
    if api_key is not None:
        content = f"url: https://cds.climate.copernicus.eu/api/v2\nkey: {api_key}"
        home_dir = os.environ["HOME"]
        with open(os.path.join(home_dir, ".cdsapirc"), "w") as f:
            f.write(content)
    os.makedirs(dst, exist_ok=True)
    client = cdsapi.Client()
    download_args = {
        "product_type": "reanalysis",
        "format": "netcdf",
        "variable": variable,
        "year": str(year),
        "month": [str(i).rjust(2, "0") for i in range(1, 13)],
        "day": [str(i).rjust(2, "0") for i in range(1, 32)],
        "time": [str(i).rjust(2, "0") + ":00" for i in range(0, 24)],
    }
    if pressure:
        src = "reanalysis-era5-pressure-levels"
        download_args["pressure_level"] = [1000, 850, 500, 50]
    else:
        src = "reanalysis-era5-single-levels"
    client.retrieve(src, download_args, dst / f"{variable}_{year}_0.25deg.nc")


def download_mpi_esm1_2_hr(dst, variable, years=(1850, 2015)):
    os.makedirs(dst, exist_ok=True)
    year_strings = [f"{y}01010600-{y+5}01010000" for y in range(*years, 5)]
    inst = "MPI-M"
    src = "MPI-ESM1-2-HR"
    exp = "historical"
    for yr in tqdm(year_strings):
        remote_fn = f"{variable}_6hrPlevPt_{src}_{exp}_r1i1p1f1_gn_{yr}.nc"
        url = (
            "https://esgf-data1.llnl.gov/thredds/fileServer/css03_data/CMIP6/"
            f"CMIP/{inst}/{src}/{exp}/r1i1p1f1/6hrPlevPt/{variable}/gn/"
            f"v20190815/{remote_fn}"
        )
        resp = requests.get(url, verify=False, stream=True)
        local_fn = os.path.join(dst, remote_fn)
        with open(local_fn, "wb") as file:
            for chunk in resp.iter_content(chunk_size=1024):
                file.write(chunk)


def download_weatherbench(dst, dataset, variable, resolution=5.625):
    os.makedirs(dst, exist_ok=True)
    if dataset not in ["era5", "cmip6"]:
        raise RuntimeError("Dataset not supported")
    url = "https://dataserv.ub.tum.de/s/m1524895/download?path=%2F"
    res = f"{resolution}deg"
    if dataset == "era5":
        ext = ".nc" if variable == "constants" else ".zip"
        remote_fn = f"{variable}_{res}{ext}"
        url = f"{url}{res}%2F{variable}&files={remote_fn}"
    elif dataset == "cmip6":
        ext = ".zip"
        remote_fn = f"{variable}_{res}{ext}"
        url = f"{url}CMIP%2FMPI-ESM%2F{res}%2F{variable}&files={remote_fn}"
    resp = requests.get(url, verify=False, stream=True)
    if variable == "constants":
        local_fn = os.path.join(dst, "constants.nc")
    else:
        local_fn = os.path.join(dst, remote_fn)
    # TODO: add a progress wheel to indicate it is running.
    # I don't think a progress bar with tqdm is doable since the total size
    # of the file is not known a priori.
    with open(local_fn, "wb") as file:
        for chunk in resp.iter_content(chunk_size=1024):
            file.write(chunk)
    if ext == ".zip":
        with ZipFile(local_fn) as myzip:
            myzip.extractall(dst)
        os.unlink(local_fn)


def download_prism(dst, variable, years=(1981, 2023)):
    os.makedirs(dst, exist_ok=True)
    ftp = FTP("prism.oregonstate.edu")
    ftp.login()
    for year in trange(*years):
        ftp.cwd(f"/daily/{variable}/{year}")
        for remote_fn in tqdm(ftp.nlst(), leave=False):
            local_fn = os.path.join(dst, remote_fn)
            with open(local_fn, "wb") as file:
                ftp.retrbinary(f"RETR {remote_fn}", file.write)
            subdir_name = re.search(r"\d{8}", remote_fn)[0]
            subdir_path = os.path.join(dst, subdir_name)
            os.mkdir(subdir_path)
            with ZipFile(local_fn) as myzip:
                myzip.extractall(path=subdir_path)
            os.unlink(local_fn)
    ftp.quit()


def main():
    parser = ArgumentParser(description="ClimateLearn's download utility.")
    subparsers = parser.add_subparsers(
        help="Data provider to download from.", dest="provider"
    )
    copernicus_era5 = subparsers.add_parser("copernicus-era5")
    mpi_esm1_2_hr = subparsers.add_parser("mpi_esm1_2_hr")
    weatherbench = subparsers.add_parser("weatherbench")
    prism = subparsers.add_parser("prism")

    copernicus_era5.add_argument("dst", help="Destination to store downloaded files.")
    copernicus_era5.add_argument("var", help="Variable to download.")
    copernicus_era5.add_argument("year", type=int)
    copernicus_era5.add_argument("--pressure", action="store_true", default=False)
    copernicus_era5.add_argument("--api_key")

    mpi_esm1_2_hr.add_argument("dst", help="Destination to store downloaded files.")
    mpi_esm1_2_hr.add_argument("var", help="Variable to download.")
    mpi_esm1_2_hr.add_argument("--start", type=int, default=1850)
    mpi_esm1_2_hr.add_argument("--end", type=int, default=2015)

    weatherbench.add_argument("dst", help="Destination to store downloaded files.")
    weatherbench.add_argument("dataset", choices=["era5", "cmip6"])
    weatherbench.add_argument("var")
    weatherbench.add_argument("--res", type=float, default=5.625)

    prism.add_argument("dst", help="Destination to store downloaded files.")
    prism.add_argument("var", help="Variable to download.")
    prism.add_argument("--start", type=int, default=1981)
    prism.add_argument("--end", type=int, default=2023)

    args = parser.parse_args()

    if args.provider == "copernicus_era5":
        download_copernicus_era5(
            args.dst, args.var, args.year, args.pressure, args.api_key
        )
    elif args.provider == "mpi_esm1_2_hr":
        download_mpi_esm1_2_hr(args.dst, args.var, (args.start, args.end))
    elif args.provider == "weatherbench":
        download_weatherbench(args.dst, args.var, args.res)
    elif args.provider == "prism":
        download_prism(args.dst, args.var, (args.start, args.end))


if __name__ == "__main__":
    main()

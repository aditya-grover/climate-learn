# Standard library
import argparse
import os
import subprocess

# Local application
from .constants import NAME_TO_CMIP

# Third party
import cdsapi


months = [str(i).rjust(2, "0") for i in range(1, 13)]
days = [str(i).rjust(2, "0") for i in range(1, 32)]
times = [str(i).rjust(2, "0") + ":00" for i in range(0, 24)]

# TODO: write exceptions in the docstrings
# TODO: figure out how to better specify legal args for dataset, variable,
#   and resolution
# TODO: for download ESGF, do we have to download all the years?
# TODO: can main even be run without runtime warning? maybe we should get rid of it


def _download_copernicus(root, dataset, variable, year, pressure=False, api_key=None):
    """Downloads data from the Copernicus Climate Data Store (CDS).
        Data is stored at `root/dataset/variable/` as NetCDF4 (`.nc`) files.
        Skips the download if a file of the expected naming convention already
        exists at the download destination. More info:
        https://cds.climate.copernicus.eu/cdsapp#!/home

    :param root: The root data directory.
    :type root: str
    :param dataset: The dataset to download. Currently, only "era5" is
        supported.
    :type dataset: str
    :param variable: The variable to download from the specified dataset.
    :type variable: str
    :param pressure: Whether to download data from different pressure levels
        instead of single-level. Defaults to `False`.
    :type pressure: bool, optional
    :param api_key: An API key for accessing CDS. Defaults to `None`. See here
        for more info: https://cds.climate.copernicus.eu/api-how-to.
    :type api_key: str, optional
    """
    if dataset not in ["era5"]:
        raise Exception("Dataset not supported")

    if api_key is not None:
        content = f"url: https://cds.climate.copernicus.eu/api/v2\nkey: {api_key}"
        open(f"{os.environ['HOME']}/.cdsapirc", "w").write(content)

    path = os.path.join(root, dataset, variable, f"{variable}_{year}_0.25deg.nc")
    print(
        f"Downloading {dataset} {variable} data for year {year} from copernicus to {path}"
    )

    if os.path.exists(path):
        return

    os.makedirs(os.path.dirname(path), exist_ok=True)

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


def _download_esgf(
    root,
    dataset,
    variable,
    institutionID="MPI-M",
    sourceID="MPI-ESM1-2-HR",
    exprID="historical",
):
    """Downloads data from the Earth System Grid Federation (ESGF).
        Data is stored at `root/dataset/pre-regrided/variable/` as a NetCDF4
        (`.nc`) file. Skips the download if a file of the expected naming
        convention already exists at the download destination. More info:
        https://esgf-node.llnl.gov/projects/cmip6/

    :param root: The root data directory.
    :type root: str
    :param dataset: The dataset to download. Currently, only "cmip6" is
        supported.
    :type dataset: str
    :param variable: The variable to download from the specified dataset.
    :type variable: str
    :param instituionID: TODO
    :type institutionID: str, optional
    :param sourceID: TODO
    :type sourceID: str, optional
    :param exprID: TODO
    :type exprID: str, optional
    """
    if dataset not in ["cmip6"]:
        raise Exception("Dataset not supported")

    path = os.path.join(root, dataset, "pre-regrided", variable)
    print(f"Downloading {dataset} {variable} data from esgf to {path}")

    os.makedirs(os.path.dirname(path), exist_ok=True)

    year_strings = [f"{y}01010600-{y+5}01010000" for y in range(1850, 2015, 5)]
    for yr in year_strings:
        file_name = ("{var}_6hrPlevPt_{sourceID}_{exprID}_r1i1p1f1_gn_{yr}.nc").format(
            var=NAME_TO_CMIP[variable], yr=yr, sourceID=sourceID, exprID=exprID
        )

        file_path = os.path.join(path, file_name)
        if os.path.exists(file_path):
            print(file_name, "exists")

        else:
            url = (
                "https://esgf-data1.llnl.gov/thredds/fileServer/css03_data/CMIP6/CMIP/{institutionID}/{sourceID}/{exprID}/r1i1p1f1/6hrPlevPt/"
                "{variable}/gn/v20190815/{file}"
            ).format(
                yr_string=yr,
                variable=NAME_TO_CMIP[variable],
                file=file_name,
                institutionID=institutionID,
                sourceID=sourceID,
                exprID=exprID,
            )
            subprocess.check_call(["wget", "--no-check-certificate", url, "-P", path])


def _download_weatherbench(root, dataset, variable, resolution="1.40625"):
    """Downloads data from WeatherBench.
        Data is stored at `root/dataset/resolution/variable/` as NetCDF4
        (`.nc`) files. Skips the download if a file of the expected naming
        convention already exists at the download destination. More info:
        https://mediatum.ub.tum.de/1524895

    :param root: The root data directory
    :type root: str
    :param dataset: The dataset to download. Currently, "era5" and "cmip6" are
        supported.
    :type dataset: str
    :param variable: The variable to download from the specified dataset.
    :type variable: str
    :param resolution:  The desired data resolution in degrees. Can be
        "1.40625", "2.8125", and "5.625". Default is "1.40625".
    :type resolution: str, optional
    """
    if dataset not in ["era5", "cmip6"]:
        raise Exception("Dataset not supported")

    path = os.path.join(root, dataset, resolution, variable)
    print(
        f"Downloading {dataset} {variable} data for {resolution} resolution from weatherbench to {path}"
    )
    if os.path.exists(path):
        return
    os.makedirs(os.path.dirname(path), exist_ok=True)

    if dataset == "era5":
        if variable != "constants":
            url = (
                "https://dataserv.ub.tum.de/s/m1524895"
                "/download?path=%2F{resolution}deg%2F{variable}&files={variable}_{resolution}deg.zip"
            ).format(resolution=resolution, variable=variable)
        elif variable == "constants":
            url = (
                "https://dataserv.ub.tum.de/s/m1524895"
                "/download?path=%2F{resolution}deg%2Fconstants&files=constants_{resolution}deg.nc"
            ).format(resolution=resolution)
    elif dataset == "cmip6":
        url = (
            "https://dataserv.ub.tum.de/s/m1524895"
            "/download?path=%2FCMIP%2FMPI-ESM%2F{resolution}deg%2F{variable}&files={variable}_{resolution}deg.zip"
        ).format(resolution=resolution, variable=variable)

    if variable != "constants":
        subprocess.check_call(
            ["wget", "--no-check-certificate", url, "-O", path + ".zip"]
        )
        subprocess.check_call(["unzip", path + ".zip", "-d", path])
    else:
        subprocess.check_call(["mkdir", path])
        subprocess.check_call(
            [
                "wget",
                "--no-check-certificate",
                url,
                "-O",
                os.path.join(path, "constants.nc"),
            ]
        )


def download(source, **kwargs):
    r"""Download interface.

    :param source: The data source to download from: "copernicus",
        "weatherbench", or "esgf".
    :param type: str
    :param \**kwargs: arguments to pass to the source-specific download
        function: :py:func:`_download_copernicus`,
        :py:func:`_download_weatherbench`, :py:func:`_download_esgf`
    """

    # TODO: this was appropriate for the Colab tutorial, but should we
    # keep it for future releases?
    if "root" not in kwargs or kwargs["root"] is None:
        kwargs["root"] = ".climate_tutorial"

    kwargs["root"] = os.path.join(kwargs["root"], f"data/{source}")

    if source == "copernicus":
        _download_copernicus(**kwargs)
    elif source == "weatherbench":
        _download_weatherbench(**kwargs)
    elif source == "esgf":
        _download_esgf(**kwargs)


def main():
    parser = argparse.ArgumentParser()

    subparsers = parser.add_subparsers(dest="source")

    subparser = subparsers.add_parser("copernicus")
    subparser.add_argument("--root", type=str, default=None)
    subparser.add_argument("--variable", type=str, required=True)
    subparser.add_argument("--dataset", type=str, choices=["era5"], required=True)
    subparser.add_argument("--year", type=int, required=True)
    subparser.add_argument("--pressure", action="store_true", default=False)
    subparser.add_argument("--api_key", type=str, default=None)

    subparser = subparsers.add_parser("weatherbench")
    subparser.add_argument("--root", type=str, default=None)
    subparser.add_argument("--variable", type=str, required=True)
    subparser.add_argument(
        "--dataset", type=str, choices=["era5", "cmip6"], required=True
    )
    subparser.add_argument("--resolution", type=str, default="5.625")

    subparser = subparsers.add_parser("esgf")
    subparser.add_argument("--root", type=str, default=None)
    subparser.add_argument("--variable", type=str, required=True)
    subparser.add_argument("--dataset", type=str, choices=["era5"], required=True)
    subparser.add_argument("--resolution", type=str, default="5.625")
    subparser.add_argument("--institutionID", type=str, default="MPI-M")
    subparser.add_argument("--sourceID", type=str, default="MPI-ESM1-2-HR")
    subparser.add_argument("--exprID", type=str, default="historical")

    args = parser.parse_args()
    download(**vars(args))


if __name__ == "__main__":
    main()

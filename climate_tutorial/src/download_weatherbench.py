import argparse
import os
import subprocess


def download_weatherbench(root, dataset, variable, resolution):
    if not os.path.exists(root):
        os.makedirs(root, exist_ok=True)
    if dataset == "era5":
        url = (
            "https://dataserv.ub.tum.de/s/m1524895"
            "/download?path=%2F{resolution}deg%2F{variable}&files={variable}_{resolution}deg.zip"
        ).format(resolution=resolution, variable=variable)
    else:  # cmip6
        url = (
            "https://dataserv.ub.tum.de/s/m1524895"
            "/download?path=%2FCMIP%2FMPI-ESM%2F{resolution}deg%2F{variable}&files={variable}_{resolution}deg.zip"
        ).format(resolution=resolution, variable=variable)
    # cmd = ["wget", "--no-check-certificate", f'"{url}"', "-O", os.path.join(root, variable + ".zip")]
    # print (" ".join(cmd))
    subprocess.check_call(["wget", "--no-check-certificate", url, "-O", os.path.join(root, variable + ".zip")])
    subprocess.check_call(["unzip", os.path.join(root, variable + ".zip"), "-d", os.path.join(root, variable)])


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--root", type=str, default="")
    parser.add_argument("--dataset", type=str, choices=["era5", "cmip6"])
    parser.add_argument("--variable", type=str, required=True)
    parser.add_argument("--resolution", type=str, default="1.40625")

    args = parser.parse_args()

    download_weatherbench(args.root, args.dataset, args.variable, args.resolution)


if __name__ == "__main__":
    main()

import argparse
import os
import subprocess


def download_weatherbench(args):
    resolution = args.resolution
    variable = args.variable
    if args.dataset == "era5":
        url = (
            "https://dataserv.ub.tum.de/s/m1524895"
            "/download?path=%2F{resolution}deg%2F{variable}&files={variable}_{resolution}deg.zip"
        ).format(resolution=resolution, variable=variable)
    else:  # cmip6
        url = (
            "https://dataserv.ub.tum.de/s/m1524895"
            "/download?path=%2FCMIP%2FMPI-ESM%2F{resolution}deg%2F{variable}&files={variable}_{resolution}deg.zip"
        ).format(resolution=resolution, variable=variable)
    cmd = ["wget", "--no-check-certificate", f'"{url}"', "-O", os.path.join(args.root, variable + ".zip")]
    # print (" ".join(cmd))
    subprocess.run(["wget", "--no-check-certificate", url, "-O", os.path.join(args.root, variable + ".zip")])


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--root", type=str, default="/datadrive/1.40625deg")
    parser.add_argument("--dataset", type=str, choices=["era5", "cmip6"])
    parser.add_argument("--variable", type=str, required=True)
    parser.add_argument("--resolution", type=str, default="1.40625")

    args = parser.parse_args()

    if not os.path.exists(args.root):
        os.makedirs(args.root, exist_ok=True)

    download_weatherbench(args)


if __name__ == "__main__":
    main()

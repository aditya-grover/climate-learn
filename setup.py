import os
import setuptools
import pkg_resources

setuptools.setup(
    name = "climate_learn",
    py_modules = ["climate_learn"],
    version = "0.0.1",
    author = "Hritik Bansal, Shashank Goel, Tung Nguyen, Aditya Grover",
    author_email = "hbansal@ucla.edu, shashankgoel@ucla.edu, tungnd@ucla.edu, agrover@ucla.edu",
    description = "Climate Learn",
    long_description = open("README.md", "r").read(),
    long_description_content_type = "text/markdown",
    url = "https://github.com/aditya-grover/climate-learn",
    packages = setuptools.find_packages(),
    install_requires = [
        "cdsapi",
        "dask",
        "importlib-metadata==4.13.0",
        "lightning",
        "matplotlib",
        "netcdf4",
        "rich",
        "scikit-learn",
        "timm",
        "torch",
        "wandb"
    ],
    extras_require = {
        "docs": [
            "nbsphinx",
            "sphinx==5.3.0",
            "sphinx_rtd_theme==1.1.1"
        ]
    },
    classifiers = [
        "Development Status :: In Progress"
        "Programming Language :: Python :: 3",
        "License :: MIT License",
        "Operating System :: OS Independent",
    ],
)

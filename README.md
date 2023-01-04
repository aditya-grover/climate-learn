<h1 align="center">ClimateLearn</h1>

[![Documentation Status](https://readthedocs.org/projects/climatelearn/badge/?version=latest)](https://climatelearn.readthedocs.io/en/latest/?badge=latest)

**ClimateLearn** is a Python library for accessing state-of-the-art climate data and machine learning models in a standardized, straightforward way. This library provides access to multiple datasets, a zoo of baseline approaches, and a suite of metrics and visualizations for large-scale benchmarking of statistical downscaling and temporal forecasting methods.

## Getting Started

### Quickstart
Please refer to this [Google Colab notebook](https://colab.research.google.com/drive/1GMT_CnxL1o4Za1Uc3Gf7u_tm_M5ECoZo?usp=sharing).

### Local Installation

**conda** is required. We recommend installing [**miniconda**](https://docs.conda.io/en/latest/miniconda.html). 

First, create a conda environment.
```console
$ conda create -n cl_env python=3.8 -y
```

Then, install the conda-only dependencies.
```console
$ conda install -n cl_env xesmf=0.7.0 -c conda-forge -y
``` 

Then, install the rest of the library with pip.
```console
$ conda activate cl_env
$ pip install -e .
```

## About Us
ClimateLearn is managed by the Machine Intelligence Group at UCLA, headed by [Professor Aditya Grover](https://aditya-grover.github.io).

## Citing ClimateLearn
If you use ClimateLearn, please see the `CITATION.cff` file or use the citation prompt provided by GitHub in the sidebar.

<h1 align="center">ClimateLearn</h1>

[![Documentation Status](https://readthedocs.org/projects/climatelearn/badge/?version=latest)](https://climatelearn.readthedocs.io/en/latest/?badge=latest)
[![Google Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1WiNEK1BHsiGzo_bT9Fcm8lea2H_ghNfa)

**ClimateLearn** is a Python library for accessing state-of-the-art climate data and machine learning models in a standardized, straightforward way. This library provides access to multiple datasets, a zoo of baseline approaches, and a suite of metrics and visualizations for large-scale benchmarking of statistical downscaling and temporal forecasting methods. 

## Getting Started

### Quickstart
Please refer to this [Google Colab](https://colab.research.google.com/drive/1WiNEK1BHsiGzo_bT9Fcm8lea2H_ghNfa).

For additional information, some key features of ClimateLearn were previewed at a spotlight tutorial in the "Tackling Climate Change with Machine Learning" Workshop at the Neural Information Processing Systems 2022 Conference. The slides and recorded talk can be found here: https://www.climatechange.ai/papers/neurips2022/114.

### Documentation
Find us on [ReadTheDocs](https://climatelearn.readthedocs.io/).

### Local Installation

**conda** is required. We recommend installing [**miniconda**](https://docs.conda.io/en/latest/miniconda.html). 

1. Clone the repository from GitHub. 
    ```
    $ git clone https://github.com/aditya-grover/climate-learn.git
    ```

2. Create a conda environment `cl_env` and install the conda-only dependencies.
    ```console
    $ conda create -n cl_env xesmf==0.7.0 -c conda-forge -y
    $ conda activate cl_env
    ```

3. Install the rest of the dependencies with pip.
    ```console
    $ cd climate-learn
    $ pip install -e .
    ```

### Integrations

- [Weights & Biases](https://wandb.ai/site)

## About Us
ClimateLearn is managed by the Machine Intelligence Group at UCLA, headed by [Professor Aditya Grover](https://aditya-grover.github.io).

## Citing ClimateLearn
If you use ClimateLearn, please see the `CITATION.cff` file or use the citation prompt provided by GitHub in the sidebar.

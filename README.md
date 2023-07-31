<h1 align="center">ClimateLearn</h1>

[![Documentation Status](https://readthedocs.org/projects/climatelearn/badge/?version=latest)](https://climatelearn.readthedocs.io/en/latest/?badge=latest)
[![CI Build Status](https://github.com/aditya-grover/climate-learn/actions/workflows/ci.yaml/badge.svg)](https://github.com/aditya-grover/climate-learn/actions/workflows/ci.yaml)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Google Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1LcecQLgLtwaHOwbvJAxw9UjCxfM0RMrX?usp=sharing)

**ClimateLearn** is a Python library for accessing state-of-the-art climate data and machine learning models in a standardized, straightforward way. This library provides access to multiple datasets, a zoo of baseline approaches, and a suite of metrics and visualizations for large-scale benchmarking of statistical downscaling and temporal forecasting methods. For further context on our past motivation and future plans, check out our announcement [blog post](https://aditya-grover.github.io/blog/2023/climate-learn/). Also, check out our [arxiv preprint](https://arxiv.org/abs/2307.01909) that showcases the flexibility of ClimateLearn in performing benchmarking and analysis on the robustness and transfer performance of deep learning models.

## Usage

[**Python 3.8+**](https://www.python.org/) is required. The xESMF package has to be installed separately since one of its dependencies, ESMpy, is available only through Conda.
```
conda install -c conda-forge xesmf
pip install climate-learn
```

### Quickstart
We have a quickstart notebook in the `notebooks` folder titled `Quickstart.ipynb`. It is intended for use in Google Colab and can be launched by clicking the Google Colab badge above or this link: https://colab.research.google.com/drive/1LcecQLgLtwaHOwbvJAxw9UjCxfM0RMrX?usp=sharing.

We also previewed some key features of ClimateLearn at a spotlight tutorial in the "Tackling Climate Change with Machine Learning" Workshop at the Neural Information Processing Systems 2022 Conference. The slides and recorded talk can be found on [Climate Change AI's website](https://www.climatechange.ai/papers/neurips2022/114).

### Documentation
Find us on [ReadTheDocs](https://climatelearn.readthedocs.io/).

## About Us
ClimateLearn is managed by the Machine Intelligence Group at UCLA, headed by [Professor Aditya Grover](https://aditya-grover.github.io).

## Contributing
Contributions are welcome! See our [contributing guide](https://github.com/aditya-grover/climate-learn/blob/main/CONTRIBUTING.md).

## Citing ClimateLearn
<!-- If you use ClimateLearn, please see the [`CITATION.cff`](https://github.com/aditya-grover/climate-learn/blob/main/CITATION.cff) file or use the citation prompt provided by GitHub in the sidebar. -->
If you use ClimateLearn in your research, please cite our paper:
```
@article{nguyen2023climatelearn,
  title={ClimateLearn: Benchmarking Machine Learning for Weather and Climate Modeling},
  author={Nguyen, Tung and Jewik, Jason and Bansal, Hritik and Sharma, Prakhar and Grover, Aditya},
  journal={arXiv preprint arXiv:2307.01909},
  year={2023}
}
```
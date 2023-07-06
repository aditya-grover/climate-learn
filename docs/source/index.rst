ClimateLearn
============

What is ClimateLearn?
---------------------

**ClimateLearn** is a Python library for accessing state-of-the-art climate 
data and machine learning models in a standardized, straightforward way. This 
library provides access to multiple datasets, a zoo of baseline approaches, 
and a suite of metrics and visualizations for large-scale benchmarking of 
statistical downscaling and temporal forecasting methods.

.. note::

   This project is under active development. The API might undergo extensive
   changes in the near future.

Getting Started
---------------
`Python 3.8+ <https://www.python.org/>`_ is required. The xESMF package has
to be installed separately since one of its dependencies, ESMpy, is available
only through Conda.

.. code-block:: shell

   conda install -c conda-forge xesmf
   pip install climate-learn

We have a quickstart notebook in the ``notebooks`` folder titled
``Quickstart.ipynb`` that walks through an example usage of ClimateLearn for
weather forecasting from downloading the data through visualizing the
predictions of a trained model. It is intended for use in Google Colab and can
be launched by clicking
`this link <https://colab.research.google.com/drive/1LcecQLgLtwaHOwbvJAxw9UjCxfM0RMrX?usp=sharing>`_.

.. toctree::
   :caption: User Guide
   :maxdepth: 2

   user-guide/tasks_and_datasets
   user-guide/models
   user-guide/metrics
   user-guide/visualizations

.. toctree::
   :caption: Development Guide
   :maxdepth: 1

   development-guide/for-developers
   development-guide/for-maintainers

Why did we build ClimateLearn?
------------------------------

In recent years, there has been a growing interest in the application of
ML-based methods for weather and climate modeling. While there are some
leaderboard benchmarks, such as WeatherBench, ClimateBench, and FloodNet, that
propose datasets and baselines for specific tasks in climate science, a
holistic software ecosystem that encompasses the entire data, modeling, and
evaluation pipeline across several tasks is lacking. Hence, we built
ClimateLearn to standardize datasets, model implementations, and evaluation
protocols for rigorous and reproducible data-driven climate science.

About Us
--------
ClimateLearn is built and maintained by the Machine Intelligence Group at UCLA,
headed by `Professor Aditya Grover <https://aditya-grover.github.io/>`_.
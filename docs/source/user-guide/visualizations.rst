Visualizations
==============

Suppose that you have loaded a data module and a model module from CliamteLearn
as such:

.. code-block:: python
    :linenos:

    from climate_learn.data import DataModule
    from climate_learn.models import load_model
    from climate_learn.utils import visualize, visualize_mean_bias

    data_module = DataModule(...)
    model_module = load_model(...)

The ``climate_learn.utils`` module provides methods to visualize data and model
outputs. Both methods will work for both the downscaling and forecasting tasks,
provided that both ``model_module`` and ``data_module`` are initialized for the
same task.

To produce visualizations of initial condition, ground truth, prediction, and
bias, do the following. 

.. code-block:: python
    :linenos:

    from climate_learn.utils import visualize
    visualize(model_module, data_module)

To produce visualizations of a model's mean bias, do the following.

.. code-block:: python
    :linenos:

    from climate_learn.utils import visualize_mean_bias
    visualize_mean_bias(model_module, data_module)
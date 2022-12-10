Visualizations
==============

Suppose that you have loaded a data module and a model module from ClimateLearn:

.. code-block:: python
    :linenos:

    from climate_learn.data import DataModule
    from climate_learn.models import load_model

    data_module = DataModule(...)
    model_module = load_model(...)

The ``climate_learn.utils`` module provides functions to visualize data and
model outputs. These functions assume ``model_module`` and ``data_module``
are initialized for the same task (both forecasting or both downscaling).

To produce visualizations of initial condition, ground truth, prediction, and
bias [#]_, do the following. 

.. code-block:: python
    :linenos:

    from climate_learn.utils import visualize
    visualize(model_module, data_module)

By default, ``visualize`` will pick 2 random dates in the test dataset. You can
change the number of dates it selects to ``n`` dates by passing ``split=n`` as
a parameter. Alternatively, you can specify exact dates by passing a list of
datetime strings formatted as ``YYYY-mm-dd:HH`` (*e.g.*, "2017-06-10:12"). See
:ref:`climate_learn.utils` for further details. 

To produce visualizations of a model's mean bias on the test dataset, do the
following.

.. code-block:: python
    :linenos:

    from climate_learn.utils import visualize_mean_bias
    visualize_mean_bias(model_module, data_module)

.. [#] Bias is defined as *predicted* minus *observed*.

Example
-------

The following can be run in Google Colab.

.. nbinput:: ipython3
    :execution-count: 1

    %%capture
    !pip install git+https://github.com/aditya-grover/climate-learn.git

.. nbinput:: ipython3
    :execution-count: 2

    # Download WeatherBench 2m_temperature data to Google Drive
    from google.colab import drive
    from climate_learn.data import download

    drive.mount("/content/drive")    
    download(
        root="/content/drive/MyDrive/Climate/.climate_tutorial",
        source="weatherbench",
        variable="2m_temperature",
        dataset="era5", 
        resolution="5.625"
    )

.. nbinput:: ipython3
    :execution-count: 3

    # Load data module for forecasting task
    from climate_learn.utils.datetime import Year, Days, Hours
    from climate_learn.data import DataModule

    data_module = DataModule(
        dataset = "ERA5",
        task = "forecasting",
        root_dir = "/content/drive/MyDrive/Climate/.climate_tutorial/data/weatherbench/era5/5.625/",
        in_vars = ["2m_temperature"],
        out_vars = ["2m_temperature"],
        train_start_year = Year(1979),
        val_start_year = Year(2015),
        test_start_year = Year(2017),
        end_year = Year(2018),
        pred_range = Days(3),
        subsample = Hours(6),
        batch_size = 128,
        num_workers = 1
    )

.. nbinput:: ipython3
    :execution-count: 4

    # Load ResNet model
    from climate_learn.models import load_model

    model_kwargs = {
        "in_channels": len(data_module.hparams.in_vars),
        "out_channels": len(data_module.hparams.out_vars),
        "n_blocks": 4
    }

    optim_kwargs = {
        "lr": 1e-4,
        "weight_decay": 1e-5,
        "warmup_epochs": 1,
        "max_epochs": 5,
    }

    model_module = load_model(
        name="resnet",
        task="forecasting",
        model_kwargs=model_kwargs,
        optim_kwargs=optim_kwargs
    )

.. nbinput:: ipython3
    :execution-count: 5

    # Visualize ResNet model performance on two dates in the test set
    from climate_learn.utils import visualize
    visualize(model_module, data_module, samples=["2017-06-01:12", "2017-08-01:18"])

.. nboutput::
    :execution-count: 5

    .. image:: images/visualize.png
        :alt: Visualizations produced by ``utils.visualize``.

.. nbinput:: ipython3
    :execution-count: 6

    # Visualize ResNet model mean bias across the entire test set
    from climate_learn.utils import visualize_mean_bias
    visualize_mean_bias(model_module, data_module)

.. nboutput::
    :execution-count: 6

    .. image:: images/visualize_mean_bias.png
        :alt: Mean bias visualization produced by ``utils.visualize_mean_bias``.

*Note:* These visualizations were produced using a trained ResNet model, but
training is omitted from this example. Please see :ref:`models-reference` for
model training.

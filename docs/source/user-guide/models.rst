Models
======
ClimateLearn's model modules are configurable based on the problem
(forecasting, using the ``ForecastLitModule`` module, and downscaling,
using the ``DownscaleLitModule`` module) and the desired model archiecture.
Currently, three deep neural network architectures are supported:

#. Convolutional neural networks: the CNN is a widely used architecture for visual recognition tasks. A constrained version of the standard neural network, CNNs capitalize on knowledge of the input's structure as an image. ClimateLearn suports two popular variants of CNNs:

    #. ResNet: ResNets are a popular variant of CNNs that have been used to achieve weather forecasting for variables such as temperature and geopotential.

    #. U-Net: U-Nets are a CNN varriant that entails both downsampling and upsampling convolutions. Their implementationin ClimateLearn allows users to benchmark U-Net for climate modeling tasks.

#. Vision transformers: the utility of ViTs for representing climate variables is largely under-explored, so ClimateLearn provides an implementation for benchmarking transformers.

Initialization
--------------

Models are initialized by the ``load_model`` function in the
``climate_learn.models`` module, which accepts an input for the desired
architecture (``"vit"``, ``"resnet"``, or ``"unet"``) and the desired
task (``"forecasting"`` or ``"downscaling"``). Below is an example of 
initializing a ViT model for temporal forecasting.

.. code-block:: python
    :linenos:

    from climate_learn.models import load_model
    model_module = load_model(name="vit", task="forecasting")

Training
--------

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
Models
======
ClimateLearn's model modules are configurable based on the problem
(forecasting, using the ``ForecastLitModule`` module, and downscaling,
using the ``DownscaleLitModule`` module) and the desired model archiecture.
Currently, three deep neural network architectures are supported:

#. Convolutional neural networks: the CNN is a widely used architecture for visual recognition tasks. A constrained version of the standard neural network, CNNs capitalize on knowledge of the input's structure as an image. ClimateLearn suports two popular variants of CNNs:

    a. ResNet: ResNets are a popular variant of CNNs [#]_ that have been used to achieve weather forecasting for variables such as temperature and geopotential [#]_.
    
    b. U-Net: U-Nets are a CNN variant that entails both downsampling and upsampling convolutions. Their development and popularity in the biomedical space [#]_ paved the way for ClimateLearn's implementation, allowing users to benchmark U-Nets for climate modeling tasks.

#. Vision transformers: ViTs are the latest contemporary to CNNs for visual recognition [#]_. The utility of ViTs for representing climate variables is largely under-explored, but has been used for short-range temperature forecasting [#]_. 

.. [#] `Deep Residual Learning for Image Recognition <https://arxiv.org/abs/1512.03385/>`_
.. [#] `Data-driven medium-range weather prediction with a Resnet pretrained on climate simulations: A new model for WeatherBench <https://arxiv.org/abs/2008.08626/>`_
.. [#] `U-Net: Convolutional Networks for Biomedical Image Segmentation <https://arxiv.org/abs/1505.04597/>`_
.. [#] `An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale <https://arxiv.org/abs/2010.11929/>`_
.. [#] `TENT: Tensorized Encoder Transformer for Temperature Forecasting <https://arxiv.org/abs/2106.14742/>`_


Initialization
--------------

Models are initialized by the ``load_model`` function in the
``climate_learn.models`` module, which accepts an input for the desired
architecture (``"vit"``, ``"resnet"``, or ``"unet"``) and the desired
task (``"forecasting"`` or ``"downscaling"``). The function also accepts
optional keyword arguments for the model and task optimizer specifically.
Below is an example of initializing a ViT model for temporal forecasting.

.. code-block:: python
    :linenos:

    from climate_learn.models import load_model
    model_kwargs = {
        "n_blocks": 4
    }
    optim_kwargs = {
        "lr": 1e-4,
    }
    model_module = load_model(
        name="vit",
        task="forecasting",
        model_kwargs=None,
        optim_kwargs=optim_kwargs
    )

Training
--------

The ``climate_learn.training`` module provides a ``Trainer`` class for
fitting and testing models. The trainer is initialized with parameters
such as the seed, the accelerator, and the maximimum number of epochs.

The trainer has two functions, ``fit`` and ``test``, used for fitting
and testing the argument model on the argument data module. Each
function assumes ``model_module`` and ``data_module`` are initialized
for the same task (both forecasting or both downscaling). See
:doc:`Metrics <metrics>` for more information on the metrics
on which the model is tested in ``Trainer.test()``.

Below is an example of fitting and testing a model with a given data module.

.. code-block:: python
    :linenos:

    from climate_learn.training import Trainer

    trainer = Trainer(
        seed = 0,
        accelerator = "gpu",
        precision = 16,
        max_epochs = 5,
    )

    trainer.fit(model_module, data_module)

    trainer.test(model_module, data_module)

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

    # Load U-Net model
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
        name="unet",
        task="forecasting",
        model_kwargs=model_kwargs,
        optim_kwargs=optim_kwargs
    )

.. nbinput:: ipython3
    :execution-count: 5

    from climate_learn.training import Trainer

    # Initialize model trainer
    trainer = Trainer(
        seed = 0,
        accelerator = "gpu",
        precision = 16,
        max_epochs = 5,
    )

.. nbinput:: ipython3
    :execution-count: 6
    trainer.fit(model_module, data_module)

.. nbinput:: ipython3
    :execution-count: 7
    trainer.test(model_module, data_module)

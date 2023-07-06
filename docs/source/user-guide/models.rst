Models
======

ClimateLearn supports a variety of baselines and deep learning models, as shown
in the table below.

+---------------+--------------------+----------------+----------------------------------+
| Type          | Model              | Relevant Tasks | Notes                            |
+===============+====================+================+==================================+
| Baseline      | Climatology        | Forecasting    |                                  |
|               +--------------------+----------------+----------------------------------+
|               | Persistence        | Forecasting    |                                  |
|               +--------------------+----------------+----------------------------------+
|               | Interpolation      | Downscaling    | Nearest, bilinear are available. |
|               +--------------------+----------------+----------------------------------+
|               | Linear regression  | | Forecasting  | | Not practical for hi-res data, |
|               |                    | | Downscaling  | | or data with many variables.   |
|               |                    | | Projection   |                                  |
+---------------+--------------------+----------------+----------------------------------+
| Deep learning | ResNet             | | Forecasting  |                                  |
|               |                    | | Downscaling  |                                  |
|               |                    | | Projection   |                                  |
|               +--------------------+----------------+----------------------------------+
|               | U-net              | | Forecasting  |                                  |
|               |                    | | Downscaling  |                                  |
|               |                    | | Projection   |                                  |
|               +--------------------+----------------+----------------------------------+
|               | Vision transformer | | Forecasting  |                                  |
|               |                    | | Downscaling  |                                  |
|               |                    | | Projection   |                                  |
+---------------+--------------------+----------------+----------------------------------+

Loading a Model
---------------

In order to construct a model, ClimateLearn requires an instantiated data
module to determine the number of input and output channels. Suppose this
data module is called ``dm``. Then, one can load baselines by name as such:

.. code-block:: python

    import climate_learn as cl

    climatology = cl.load_forecasting_module(
        data_module=dm,
        architecture="climatology"
    )
    interpolation = cl.load_downscaling_module(
        data_module=dm,
        architecture="nearest-interpolation"
    )

We also currently provide one deep learning architecture, with its associated
optimizer and learning rate scheduler, by
`Rasp & Theurey (2020) <https://arxiv.org/abs/2008.08626>`_.

.. code-block:: python

    import climate_learn as cl

    resnet = cl.load_forecasting_module(
        data_module=dm,
        architecture="rasp-theurey-2020"
    )

.. note::

    Our goal for the future is to implement as many architectures from the
    literature as we can find for fair comparison and benchmarking. If you
    would like to contribute, please open an 
    `issue on our GitHub repository <https://github.com/aditya-grover/climate-learn/issues>`_.

ClimateLearn also supports customization of the provided architectures in two
ways. First, one can specify the customization in the loading function itself.

.. code-block:: python

    import climate_learn as cl

    model = cl.load_forecasting_module(
        data_module=dm,
        model="resnet",
        model_kwargs={"n_blocks": 4, "history": 5},
        optim="adamw",
        optim_kwargs={"lr": 5e-4},
        sched="linear-warmup-cosine-annealing",
        sched_kwargs={"warmup_epochs": 5, "max_epochs": 50}
    )

Second, one can insantiate the model, optimizer, and learning rate scheduler
directly. Note that one can mix directly instantiated and ClimateLearn-provided
options.

.. code-block:: python

    import climate_learn as cl
    from torch.optim import SGD
    from torch.optim.lr_scheduler import ReduceLROnPlateau

    model = cl.load_forecasting_module(
        data_module=dm,
        model=cl.models.hub.ResNet(...),
        optim=SGD(...),
        sched=ReduceLROnPlateau(...)
    )
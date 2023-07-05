Tasks and Datasets
==================

ClimateLearn supports multiple tasks and datasets for weather and climate
modeling. First, we introduce the tasks to motivate the choice of datasets.
Then, we describe the datasets that are available through ClimateLearn and
show code examples of how to download the data. Finally, we show how to process
the data with ClimateLearn and prepare them for use with your machine learning
models.

Tasks
-----

**Weather forecasting** is the task of predicting the weather at a future time
step :math:`t + \Delta t` given the weather conditions at the current step
:math:`t` and optionally steps preceding :math:`t`. A ML model receives an
input of shape :math:`C\times H\times W` and predicts an output of shape
:math:`C'\times H\times W`. :math:`C` and :math:`C'` denote the number of input
and output channels, respectively, which contain variables such as geopotential,
temperature, and humidity. :math:`H` and :math:`W` denote the spatial coverage
and resolution of the data, which depend on the region studied and how densely
we grid it.

**Downscaling** Due to their high computational cost, existing climate models
often use large grid cells, leading to low-resolution predictions. While useful
for understanding large-scale climate trends, these do not provide sufficient
detail to analyze local phenomena and design regional policies. The process of
correcting biases in climate model outputs and mapping them to higher
resolutions is known as downscaling. ML models for downscaling are trained to
map an input of shape :math:`C\times H\times W` to a higher resolution output
:math:`C'\times H'\times W'`, where :math:`H'\gt H` and :math:`W'\gt W`.

**Climate projection** aims to obtain long-term predictions of the climate under
different forcings, *e.g.*, greenhouse gas emissions. For instance, one might
want to predict the annual mean distributions of variables such as surface
temperature and precipitation given levels of atmospheric carbon dioxide and
methane.

ERA5 Dataset
------------

**ERA5** is a reanalysis dataset maintained by the European Center for
Medium-Range Weather Forecasting (ECMWF). In its raw format, ERA5 contains
hourly data from 1979 to the current time on a grid with cells of width and
height :math:`0.25^\circ` of the Earth, with different climate variables at
37 different pressure levels plus the planet's surface. This corresponds to
nearly 400,000 data samples, each a matrix of shape :math:`721\times 1440`.
Since this is too big for most deep learning models, ClimateLearn supports
downloading a smaller, pre-processed version of ERA5 data from WeatherBench.

.. _weatherbench-era5-download:

Downloading from WeatherBench
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

ClimateLearn provides ERA5 data through two sources. One source is
`WeatherBench <https://mediatum.ub.tum.de/1524895>`_.

.. code-block:: python

    import climate_learn as cl

    root_directory = "/home/user/climate-learn"
    variable = "2m_temperature"
    cl.data.download_weatherbench(
        dst=f"{root_directory}/{variable}",
        dataset="era5",
        variable=variable,
        resolution=5.625  # optional, default is 5.625
    )

Note that ERA5 has both single-level and pressure-level variables. WeatherBench
provides temperature at 850 hPa and geopotential at 500 hPa separate from
temperature at all pressure levels and geopotential at all pressure levels. We
recommend you to download these variables as such:

.. code-block:: python

    import climate_learn as cl

    root_directory = "/home/user/climate-learn"
    cl.data.download_weatherbench(
        f"{root_directory}/temperature",
        dataset="era5",
        variable="temperature_850",
        resolution=5.625  # optional, default is 5.625
    )
    cl.data.download_weatherbench(
        f"{root_directory}/geopotential",
        dataset="era5",
        variable="geopotential_500",
        resolution=5.625  # optional, default is 5.625
    )

.. _weatherbench-era5-reference:

WeatherBench Quick Reference
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The following variables can be downloaded from WeatherBench at 1.40625, 2.8125,
and 5.625 degree resolutions. The temporal coverage is 1978 to 2018 at hourly
intervals, and the pressure levels are 50, 250, 500, 600, 700, 850, and 925 hPa.

+-----------------+----------------------------------+----------------------------------+
| Type            | Variable                         |               Notes              |
+=================+==================================+==================================+
| Single-level    | ``2m_temperature``               |                                  |
|                 +----------------------------------+----------------------------------+
|                 | ``10m_u_component_of_wind``      |                                  |
|                 +----------------------------------+----------------------------------+
|                 | ``10m_v_component_of_wind``      |                                  |
|                 +----------------------------------+----------------------------------+
|                 | ``geopotential_500``             | Extracted from ``geopotential``. |
|                 +----------------------------------+----------------------------------+
|                 | ``land-sea mask``                | Download as ``constants``.       |
|                 +----------------------------------+----------------------------------+
|                 | ``mean_sea_level_pressure``      |                                  |
|                 +----------------------------------+----------------------------------+
|                 | ``orography``                    | Download as ``constants``.       |
|                 +----------------------------------+----------------------------------+
|                 | ``surface_pressure``             |                                  |
|                 +----------------------------------+----------------------------------+
|                 | ``temperature_850``              | Extracted from ``temperature``.  |
|                 +----------------------------------+----------------------------------+
|                 | ``toa_incident_solar_radiation`` |                                  |
|                 +----------------------------------+----------------------------------+
|                 | ``total_cloud_cover``            |                                  |
|                 +----------------------------------+----------------------------------+
|                 | ``total_precipitation``          |                                  |
+-----------------+----------------------------------+----------------------------------+
| Pressure levels | ``geopotential``                 |                                  |
|                 +----------------------------------+----------------------------------+
|                 | ``potential_vorticity``          |                                  |
|                 +----------------------------------+----------------------------------+
|                 | ``relative_humidity``            |                                  |
|                 +----------------------------------+----------------------------------+
|                 | ``specific_humidity``            |                                  |
|                 +----------------------------------+----------------------------------+
|                 | ``temperature``                  |                                  |
|                 +----------------------------------+----------------------------------+
|                 | ``u_component_of_wind``          |                                  |
|                 +----------------------------------+----------------------------------+
|                 | ``v_component_of_wind``          |                                  |
|                 +----------------------------------+----------------------------------+
|                 | ``vorticity``                    |                                  |
+-----------------+----------------------------------+----------------------------------+

Downloading from Copernicus
^^^^^^^^^^^^^^^^^^^^^^^^^^^

While we generally recommend using WeatherBench, ClimateLearn also provides
access to ERA5 data through
`Copernicus <https://cds.climate.copernicus.eu/cdsapp#!/search?type=dataset>`_.
Copernicus ERA5 data is not pre-processed and requires an API key, which can be
obtained by following the instructions at this link: https://cds.climate.copernicus.eu/api-how-to.
Once you have the API key, the following code will download ERA5 data from
Copernicus. The API key only needs to be provided on the first function call.

.. code-block:: python

    import climate_learn as cl

    root_directory = "/home/user/climate-learn"
    variable = "2m_temperature"
    year = 2000
    cl.data.download_copernicus_era5(
        dst=f"{root_directory}/{variable}",
        variable=variable,
        year=year,
        pressure=False, # optional, default is False
        api_key={YOUR_API_KEY_HERE} # optional, only required on first call
    )

We refer to the Copernicus documentation for ERA5 data on
`single levels <https://cds.climate.copernicus.eu/cdsapp#!/dataset/reanalysis-era5-single-levels?tab=overview>`_
and
`pressure levels <https://cds.climate.copernicus.eu/cdsapp#!/dataset/reanalysis-era5-pressure-levels?tab=overview>`_
for details about available years and variables.

CMIP6 Data Collection
---------------------

**CMIP6** is a collection of simulated data from the Coupled Model
Intercomparison Project Phase 6 (CMIP6), an international effort across
different climate modeling groups to compare and evaluate their global climate
models. ClimateLearn facilitates access to data produced by the MPI-ESM1.2-HR
model of CMIP6 as it contains similar climate variables as those represented in
ERA5. MPI-ESM1.2-HR provides data from 1850 to 2015 at 6 hour intervals on a
grid with cells of width and height :math:`1^\circ`. Since this corresponds to
data that is too big for most deep learning models, ClimateLearn provides
a smaller version of the raw MPI-ESM1.2-HR data.

.. _weatherbench-cmip6-download:

Downloading from WeatherBench
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Besides providing ERA5 data, `WeatherBench <https://mediatum.ub.tum.de/1524895>`_
also provides data from MPI-ESM1.2-HR of CMIP6.

.. code-block:: python

    import climate_learn as cl

    root_directory = "/home/user/climate-learn"
    variable = "temperature"
    cl.data.download_weatherbench(
        dst=f"{root_directory}/{variable}",
        dataset="cmip6",
        variable=variable,
        resolution=5.625  # optional, default is 5.625
    )

.. _weatherbench-cmip6-reference:

WeatherBench Quick Reference
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The following variables can be downloaded from WeatherBench at 2.8125 and
5.625 degree resolutions. The temporal coverage is 1850 to 2015 at hourly
intervals, and the pressure levels are 50, 250, 500, 600, 700, 850, and 925 hPa
(same as ERA5 provided by WeatherBench).

+-------------------------+
| Variable                |
+=========================+
| ``geopotential``        |
+-------------------------+
| ``specific_humidity``   |
+-------------------------+
| ``temperature``         |
+-------------------------+
| ``u_component_of_wind`` |
+-------------------------+
| ``v_component_of_wind`` |
+-------------------------+

Downloading from the ESGF
^^^^^^^^^^^^^^^^^^^^^^^^^

While we generally recommend using WeatherBench, ClimateLearn also provides
access to the CMIP6 data through the
`Earth System Grid Federation (ESGF) <https://esgf-node.llnl.gov/search/cmip6/>`_.

.. code-block:: python

    import climate_learn as cl

    root_directory = "/home/user/climate-learn"
    variable = "tas"
    cl.data.download_mpi_esm1_2_hr(
        dst=f"{root_directory}/{variable}",
        variable=variable,
        years=(1850, 2015), # optional, (1850, 2015) is the default range
    )

ESGF Quick Reference
^^^^^^^^^^^^^^^^^^^^

The following data can be downloaded from ESGF at 100km resolution, or about
:math:`0.8^\circ`. The temporal coverage is 1850 to 2015 (non-inclusive end)
at 6 hour intervals.

+-----------+------------------------------------------+
| Variable  | Long Name                                |
+===========+==========================================+
| ``ps``    | Surface air pressure                     |
+-----------+------------------------------------------+
| ``tsl``   | Temperature of soil                      |
+-----------+------------------------------------------+
| ``tas``   | Near-surface air temperature             |
+-----------+------------------------------------------+
| ``huss``  | Near-surface specific humidity           |
+-----------+------------------------------------------+
| ``vas``   | Northward near-surface wind              |
+-----------+------------------------------------------+
| ``uas``   | Eastward near-surface wind               |
+-----------+------------------------------------------+
| ``mrsos`` | Moisture in upper portion of soil column |
+-----------+------------------------------------------+
| ``mrsol`` | Total water content of soil layer        |
+-----------+------------------------------------------+
| ``ta``    | Air temperature                          |
+-----------+------------------------------------------+
| ``hus``   | Specific humidity                        |
+-----------+------------------------------------------+
| ``va``    | Northward wind                           |
+-----------+------------------------------------------+
| ``psl``   | Sea level pressure                       |
+-----------+------------------------------------------+
| ``ua``    | Eastward wind                            |
+-----------+------------------------------------------+
| ``zg``    | Geopotential height                      |
+-----------+------------------------------------------+

PRISM Dataset
-------------

**PRISM** is a dataset of various observed atmospheric variables like
precipitation and temperature over the conterminous United States at varying
spatial and temporal resolutions from 1895 to present day. It is maintained
by the PRISM Climtae Group at Oregon State University. At the highest publicly
available resolution, PRISM contains daily data on a grid with cells of width
and height 4 km (approximately :math:`0.03^\circ`). Since this also corresponds
to data that is too big for most deep learning models, ClimateLearn provides
a regridded version of raw PRISM data to :math:`0.75^\circ` resolution.

.. code-block:: python

    import climate_learn as cl

    root_directory = "/home/user/climate-learn"
    variable = "tmax"
    cl.data.download_prism(
        dst=f"{root_directory}/{variable}",
        variable=variable,
        years=(1981, 2023), # optional, (1981, 2023) is the default range
    )

The temporal coverage for the data ClimateLearn facilitates access to is 1981
to present year (inclusive) at daily intervals. We refer to the documentation
for descriptions of the available variables:
https://prism.oregonstate.edu/documents/PRISM_datasets.pdf.

.. note::

    The script at `climate_learn/data/download.py` can be run standalone to
    download data as well.

Data Processing
---------------

From WeatherBench
^^^^^^^^^^^^^^^^^

The following assumes you have downloaded ERA5 data from Weatherbench to the
directory ``/home/user/climate-learn/``.

.. code-block:: python

    from climate_learn.data.processing.nc2npz import convert_nc2npz

    convert_nc2npz(
        root_dir="/home/user/climate-learn",
        save_dir="/home/user/climate-learn/processed",
        variables=["temperature", "geopotential"],
        start_train_year=1979,
        start_val_year=2015,
        start_test_year=2017,
        end_year=2018,
        num_shards=16
    )

If you also have constants data downloaded, the above code snippet will handle
it automatically. You do not have to specify ``constants`` for the ``variables``
argument.

Extreme ERA5 Dataset
^^^^^^^^^^^^^^^^^^^^

**Extreme-ERA5** is a subset of ERA5 that we have curated to evaluate
forecasting performance for extreme weather events. Specifically, we consider
events where individual climate variables exceed critical values locally.
Heat waves and cold snaps are examples of such events that are intuitively
familiar. To generate the extreme ERA5 dataset, ClimateLearn requires ERA5
data downloaded from WeatherBench. Then, run the script at
``src/climate_learn/data/processing/era5_extreme.py``.

From PRISM
^^^^^^^^^^

Use the scripts at
``src/climate_learn/data/processing/era5_cropped.py`` and
``src/climate_learn/data/processing/prism.py``.

.. note::

    Currently, ClimateLearn normalizes all data to :math:`\mathcal{N}(0,1)`. We
    recognize that this might not be the best transform for every variable. For
    example, it is unreasonable to model precipitation according to a Gaussian
    distribution. In the future, we will add support for different transforms in
    data processing.

Loading Data
------------

Once data has been downloaded and processed, it can be loaded into PyTorch
dataloaders for forecasting and downscaling. Legal arguments to the ``task``
parameter are ``direct-forecasting``, ``iterative-forecasting``,
``continuous-forecasting``, and ``downscaling``.

.. code:: python

    import climate_learn as cl
    dm = cl.data.IterDataModule(
        task,
        inp_root_dir,
        out_root_dir,
        in_vars,
        out_vars,
        src="era5",
        history=3,
        window=6,
        pred_range=args.pred_range,
        subsample=6,
        batch_size=128,
        num_workers=8,
    )

One can also load data for climate projection.

.. code:: python

    import climate_learn as cl
    dm = cl.data.ClimateBenchDataModule(
        root_dir,
        variables,
        out_variables,
        train_ratio=0.9,
        history=10,
        batch_size=16,
        num_workers=1,
    )
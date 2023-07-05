Tasks and Datasets
==================

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

Datasets
--------

**ERA5** is a reanalysis dataset maintained by the European Center for
Medium-Range Weather Forecasting (ECMWF). In its raw format, ERA5 contains
hourly data from 1979 to the current time on a grid with cells of width and
height :math:`0.25^\circ` of the Earth, with different climate variables at
37 different pressure levels plus the planet's surface. This corresponds to
nearly 400,000 data samples, each a matrix of shape :math:`721\times 1440`.
Since this is too big for most deep learning models, ClimateLearn supports
downloading a smaller, pre-processed version of ERA5 data from WeatherBench.

**Extreme-ERA5** is a subset of ERA5 that we have curated to evaluate
forecasting performance for extreme weather events. Specifically, we consider
events where individual climate variables exceed critical values locally.
Heat waves and cold snaps are examples of such events that are intuitively
familiar.

**CMIP6** is a collection of simulated data from the Coupled Model
Intercomparison Project Phase 6 (CMIP6), an international effort across
different climate modeling groups to compare and evaluate their global climate
models. ClimateLearn facilitates access to data produced by the MPI-ESM1.2-HR
model of CMIP6 as it contains similar climtae variables as those represented in
ERA5. MPI-ESM1.2-HR provides data from 1850 to 2015 at 6 hour intervals on a
grid with cells of width and height :math:`1^\circ`. Since this corresponds to
data that is too big for most deep learning models, ClimateLearn provides
a smaller version of the raw MPI-ESM1.2-HR data.

**PRISM** is a dataset of various observed atmospheric variables like
precipitation and temperature over the conterminous United States at varying
spatial and temporal resolutions from 1895 to present day. It is maintained
by the PRISM Climtae Group at Oregon State University. At the highest publicly
available resolution, PRISM contains daily data on a grid with cells of width
and height 4 km (approximately :math:`0.03^\circ`). Since this also corresponds
to data that is too big for most deep learning models, ClimateLearn provides
a regridded version of raw PRISM data to :math:`0.75^\circ` resolution.

Code Examples
-------------

The following examples assume you have imported ClimateLearn as ``cl``.

Downloading ERA5 Data
^^^^^^^^^^^^^^^^^^^^^

ClimateLearn provides ERA5 data through two sources. One source is
`WeatherBench <https://mediatum.ub.tum.de/1524895>`_.

.. code-block:: python

    root_directory = "/home/user/climate-learn"
    variable = "2m_temperature"

    cl.data.download_weatherbench(
        dst=f"{root_directory}/{variable}",
        dataset="era5",
        variable=variable
    )

    
Note that ERA5 has both single-level and pressure-level variables. WeatherBench
provides temperature at 850 hPa and geopotential at 500 hPa separate from
temperature at all pressure levels and geopotential at all pressure levels. We
recommend you to download these variables as such:

.. code-block:: python

    cl.data.download_weatherbench(
        f"{root_directory}/temperature",
        dataset="era5",
        variable="temperature_850"
    )

    cl.data.download_weatherbench(
        f"{root_directory}/geopotential",
        dataset="era5",
        variable="geopotential_500"
    )
    
While we generally recommend using WeatherBench, ClimateLearn also provides
access to ERA5 data through
`Copernicus <https://cds.climate.copernicus.eu/cdsapp#!/search?type=dataset>`_.
Copernicus ERA5 data is not pre-processed and requires an API key, which can be
obtained by following the instructions at this link: https://cds.climate.copernicus.eu/api-how-to.
Once you have the API key, the following code will download ERA5 data from
Copernicus. The API key only needs to be provided on the first function call.

.. code-block:: python

    cl.data.download_copernicus_era5(
        dst=f"{root_directory}/{variable}",
        variable=variable,
        year=year,
        api_key={YOUR_API_KEY_HERE}
    )
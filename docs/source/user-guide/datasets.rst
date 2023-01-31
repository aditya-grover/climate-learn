Datasets
========

The package currently supports two climate datasets, ERA5 and CMIP6. The datasets can be downloaded through the package via multiple sources. 


ERA5
------------------
The ERA5 dataset provides hourly estimates of a large number of atmospheric, land and oceanic climate variables. [#]_ It can be downloaded through Copernicus or via the WeatherBench data repository [#]_.

Copernicus
^^^^^^^^^^^^^^

.. code-block:: python

    download(source = "copernicus", variable = "2m_temperature", dataset = "era5", year = 1979, api_key = api_key)

Though it depends on the variable, the average download time for a single variable via Copernicus is around 25 minutes. The API key can be geenrated on the `cds website <https://cds.climate.copernicus.eu/api-how-to>`_.

Weatherbench 
^^^^^^^^^^^^^^^

.. code-block:: python

    download(root = path, source = "weatherbench", variable = "2m_temperature", dataset = "era5", resolution = "5.625")

The authors of the weatherbench paper have made the ERA5 dataset readily available in three resolutions: 1.4062, 2.8125, and 5.625 degrees. The average download time for a single variable is around 5 minutes. 


CMIP6
------------------
The CMIP6 dataset contains simulation data from a variety of climate models.

Weatherbench 
^^^^^^^^^^^^^^

.. code-block:: python

    download(root = path, source = "weatherbench", variable = "2m_temperature", dataset = "cmip6", resolution = "5.625")

The authors of the WeatherBench paper have made a regridded historical climate run in CMIP6 readily available in the same data repository as the ERA5 data. The average download time for a single variable is around 5 minutes. The available resolutions are 1.4062 and 5.625 degrees.

ESGF
^^^^^^^^^^^^^^

.. code-block:: python

    download(root = path, dataset = "cmip6", variable = "temperature", resolution = "5.625", institutionID="MPI-M", sourceID="MPI-ESM1-2-HR", exprID="historical")

The CMIP6 data is also available through the Earth System Grid Federation (ESGF)'s own servers. It takes several hours to download a single variable, but as long as the server is available, any variable of any simulation can be downloaded. 


.. [#] `This part is quoted from ECMWF <https://www.ecmwf.int/en/forecasts/datasets/reanalysis-datasets/era5>`_
.. [#] `link to data repository <https://mediatum.ub.tum.de/1524895>`_
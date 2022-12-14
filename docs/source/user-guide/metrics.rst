Metrics
=======

Currently, there are 12 metrics supported in ClimateLearn, including commonly used metrics like :ref:`Mean Squared Error<Mean Squared Error>` (MSE), :ref:`Root Mean Squared Error<Root Mean Squared Error>` (RMSE), and :ref:`Pearson Correlation Coefficient<Pearson Correlation Coefficient>`, for forecasting and downscaling tasks. Part of these metrics are applied as loss functions in the training, validation, and test steps according to specific types of method used. The rest are used as evaluation metrics in the test steps.


Mean Squared Error
------------------
.. code-block:: python

    def mse(pred, y, vars, mask=None, transform_pred=False, transform=None, lat=None, log_steps=None, log_days=None, log_day=None, clim=None)

The `mse <https://github.com/aditya-grover/climate-learn/blob/main/climate_learn/models/modules/utils/metrics.py#L7>`_ function computes `mean square error <https://en.wikipedia.org/wiki/Mean_squared_error>`_, a risk metric corresponding to the expected value of the squared error or loss. This is used as default training loss function in downscaling task.


Root Mean Squared Error
-----------------------
.. code-block:: python

    def rmse(pred, y, vars, mask=None, transform_pred=False, transform=None, lat=None, log_steps=None, log_days=None, log_day=None, clim=None)

The `rmse <https://github.com/aditya-grover/climate-learn/blob/main/climate_learn/models/modules/utils/metrics.py#L275>`_ function computes `root-mean-square error <https://en.wikipedia.org/wiki/Root-mean-square_deviation>`_, the square root of the second sample moment of the differences between predicted values and observed values or the quadratic mean of these differences. 

.. math:: \sqrt{\frac{1}{N_{lat}N_{lon}} \sum_{N_{lat}}\sum_{N_{lon}}(prediction - truth)^2 }

The size of ``pred`` and ``y`` being ``[N, C, H, W]``, and the mean is computed over sampling points of the grid of size ``H * W``, with ``N`` being the size of ensemble, ``C`` being the number of channels. This is used in the validation and test steps's upsampling in the downscaling task.


Latitude-Weighted RMSE
----------------------

.. code-block:: python

    def lat_weighted_rmse(pred, y, vars, mask=None, transform_pred=True, transform=None, lat=None, log_steps=None, log_days=None, log_day=None, clim=None)

The `lat_weighted_rmse <https://github.com/aditya-grover/climate-learn/blob/main/climate_learn/models/modules/utils/metrics.py#L169>`_ function is similar to the regular :ref:`RMSE<Root Mean Squared Error>`, but is given a weight for every pixel in the grid map according to the latitude value on the earth. 

.. math:: \sqrt{\frac{1}{N_{lat}N_{lon}} \sum^{N_{lat}}_j \sum^{N_{lon}}_k L(j)(prediction - truth)^2 }

Pixels near the equator are given more weight because the earth is curved leading to less area towards the pole.

.. math:: L(j) = \frac{cos(lat(j))}{ \frac{1}{N_{lat}} \sum_j^{N_{lat}} cos(lat(j))}

This metric is being used to evaluate all the baseline methods, and the validation/test steps of deterministic methods.


Anomaly Correlation Coefficient
-------------------------------

.. code-block:: python

    def lat_weighted_acc(pred, y, vars, mask=None, transform_pred=True, transform=None, lat=None, log_steps=None, log_days=None, log_day=None, clim=None)

The `lat_weighted_acc <https://github.com/aditya-grover/climate-learn/blob/main/climate_learn/models/modules/utils/metrics.py#L234>`_ is an weighted version of Anomaly Correlation Coefficient (ACC). 

.. math:: ACC = \frac{\overline{(pred - clim)(truth - clim)}}{\sqrt{\overline{(pred - clim)^2} \space \overline{(truth - clim)^2}}}

ACC is one of the most widely used measures in the verification of spatial fields. It is the spatial correlation between a forecast anomaly relative to climatology, and a verifying analysis anomaly relative to climatology. ACC represents a measure of how well the forecast anomalies have represented the observed anomalies and shows how well the predicted values from a forecast model "fit" with the real-life data [#]_. This metric is used in the test step of forecasting task for deterministic method.


Pearson Correlation Coefficient
-------------------------------

.. code-block:: python

    def pearson(pred, y, vars, mask=None, transform_pred=False, transform=None, lat=None, log_steps=None, log_days=None, log_day=None, clim=None)

The `pearson <https://github.com/aditya-grover/climate-learn/blob/main/climate_learn/models/modules/utils/metrics.py#L299>`_ (PCC) is a measure of linear correlation between two sets of data. It is the ratio between the covariance of two variables and the product of their standard deviations. It is calculated using `scipy.stats.pearsonr(x, y) <https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.pearsonr.html>`_ with the ``pred`` and ``truth`` as input. 

.. math:: PCC = \frac{\sum (x - m_x)(y - m_y)}{\sqrt{\sum (x - m_x)^2 \sum (y - m_y)^2}}

This metric is used in the validation/test steps for downscaling task.

Mean Bias
---------

.. code-block:: python

    def mean_bias(pred, y, vars, mask=None, transform_pred=False, transform=None, lat=None, log_steps=None, log_days=None, log_day=None, clim=None)

The `mean_bias <https://github.com/aditya-grover/climate-learn/blob/main/climate_learn/models/modules/utils/metrics.py#L322>`_ is a function that calculates the absolute difference between spatial mean of predictions and observations. 

.. math:: \sqrt{\frac{1}{N_{lat}N_{lon}} \sum_{N_{lat}}\sum_{N_{lon}}|prediction - truth| }

This metric is used in the :doc:`visualization <visualizations>` to give a direct idea of the difference between prediction and truth value. It is also used in the validation/test steps for downscaling task.

Latitude-Weighted Spread-Skill Ratio
------------------------------------

.. code-block:: python

    def lat_weighted_spread_skill_ratio(pred, y, vars, mask=None, transform_pred=True, transform=None, lat=None, log_steps=None, log_days=None, log_day=None, clim=None)

`lat_weighted_spread_skill_ratio <https://github.com/aditya-grover/climate-learn/blob/main/climate_learn/models/modules/utils/metrics.py#L203>`_ is a latitude-weighted version of spread-skill ratio, which is a first-order measure of the reliability of the ensemble.

.. math:: \sqrt{\frac{1}{N_{lat}N_{lon}} \sum^{N_{lat}}_j \sum^{N_{lon}}_k L(j)var(f_{j,k}) }

where :math:`var(f_{j,k})` is the variance in the ensemble dimension.
This metric is being used as one of the validation loss function for the parametric prediction of probabilistic neural networks.


Latitude-Weighted CRPS
----------------------

.. code-block:: python

    def crps_gaussian(pred, y, vars, mask=None, transform_pred=None, transform=None, lat=None, log_steps=None, log_days=None, log_day=None, clim=None)

The `crps_gaussian <https://github.com/aditya-grover/climate-learn/blob/main/climate_learn/models/modules/utils/metrics.py#L109>`_ calculates the latitude-weighted Continuous Ranked Probability Score, in order to evaluate the calibration and sharpness of the ensemble. CRPS is a measure of how good forecasts are in matching observed outcomes [#]_. Where:

- CRPS = 0 the forecast is wholly accurate;
- CRPS = 1 the forecast is wholly inaccurate.

This metric is being used as one of the train/validation loss function for the probabilistic method.


.. [#] `This part is quoted from ECMWF <https://confluence.ecmwf.int/display/FUG/Anomaly+Correlation+Coefficient>`_
.. [#] `This part is quoted from ECMWF appendices <https://confluence.ecmwf.int/display/FUG/12.B+Statistical+Concepts+-+Probabilistic+Data#:~:text=The%20Continuous%20Ranked%20Probability%20Score>`_
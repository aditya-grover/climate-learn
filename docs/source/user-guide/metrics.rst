Metrics
=======

ClimateLearn provides the following metrics for deterministic predictions.

- Mean squared error (MSE)
- Mean squared error skill score (MSESS)
- Mean absolute error (MAE)
- Root mean squared error (RMSE)
- Anomaly correlation coefficient (ACC)
- Pearson's correlation coefficient (Pearson)
- Mean bias
- Normalized root mean squared error (NRMSEs)
- Normalized root mean squared error in global mean (NRMSEg)

For probabilistic forecasts, the library provides the following metrics.

- Gaussian continuous ranked probability score (Gaussian CRPS)
- Gaussian spread
- Gaussian spread-skill ratio (Gaussian SSR)

We refer to the following sources for the definitions and motivations for these
metrics:

- https://geo.libretexts.org/Bookshelves/Meteorology_and_Climate_Science/Practical_Meteorology_(Stull)/20%3A_Numerical_Weather_Prediction_(NWP)/20.7%3A_Forecast_Quality_and_Verfication
- https://repository.library.noaa.gov/view/noaa/48746
- https://arxiv.org/abs/2205.00865
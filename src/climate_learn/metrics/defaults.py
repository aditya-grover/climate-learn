# Standard library
from dataclasses import dataclass
from functools import partial
from typing import Callable, List

# Local application
from .functional import *


@dataclass
class ForecastingMetrics:
    TRAIN_DEFAULT: List[Callable] = [
        mse
    ]
    VAL_DEFAULT: List[Callable] = [
        lat_mse,
        lat_rmse,
        partial(denormalized, metric=lat_acc)
    ]
    TEST_DEFAULT: List[Callable] = [
        lat_rmse,
        partial(denormalized, metric=lat_acc)
    ]
    

@dataclass
class DownscalingMetrics:
    TRAIN_DEFAULT: List[Callable] = [
        mse
    ]
    VAL_DEFAULT: List[Metric] = [
        rmse,
        pearson,
        mean_bias
    ]
    TEST_DEFAULT: List[Metric] = [
        rmse,
        pearson,
        mean_bias
    ]
from climate_learn.data.climate_dataset.args import (
    ClimateDatasetArgs,
    ERA5Args,
    StackedClimateDatasetArgs,
)


class TestStackedClimateDatasetArgsInstantiation:
    def test_initialization(self):
        data_args = []
        data_arg1 = ClimateDatasetArgs(
            variables=["random_variable_1", "random_variable_2"],
            name="my_climate_dataset",
        )
        data_arg2 = ERA5Args(
            root_dir="my_data_path",
            variables=["geopotential", "2m_temperature"],
            years=range(2010, 2015),
            name="era5",
        )
        data_args.append(data_arg1)
        data_args.append(data_arg2)
        StackedClimateDatasetArgs(data_args=data_args)

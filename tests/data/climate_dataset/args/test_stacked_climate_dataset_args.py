from climate_learn.data.climate_dataset.args import (
    ClimateDatasetArgs,
    ERA5Args,
    StackedClimateDatasetArgs,
)


class TestStackedClimateDatasetArgsInstantiation:
    def test_initialization(self):
        data_args = []
        data_arg1 = ClimateDatasetArgs(
            variables=["random_variable_1", "random_variable_2"], split="train"
        )
        data_arg2 = ERA5Args(
            root_dir="my_data_path",
            variables=["random_variable_1", "random_variable_2"],
            years=range(2010, 2015),
            split="train",
        )
        data_args.append(data_arg1)
        data_args.append(data_arg2)
        StackedClimateDatasetArgs(data_args=data_args)

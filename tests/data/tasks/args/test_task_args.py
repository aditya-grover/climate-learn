from climate_learn.data.tasks.args import TaskArgs
from climate_learn.data.climate_dataset.args import ClimateDatasetArgs


class TestTaskArgsInstantiation:
    def test_initialization(self):
        temp_data_args = ClimateDatasetArgs(
            variables=["random_variable_1", "random_variable_2"], split="train"
        )
        TaskArgs(
            temp_data_args,
            in_vars=["random_variable_1"],
            out_vars=["random_variable_2"],
        )

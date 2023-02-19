from climate_learn.data_module.tasks import Task
from climate_learn.data_module.data.args import DataArgs
from climate_learn.data_module.tasks.args import TaskArgs


class TestTaskInstantiation:
    def test_initialization(self):
        temp_data_args = DataArgs(
            variables=["random_variable_1", "random_variable_2"], split="Train"
        )
        Task(
            TaskArgs(
                temp_data_args,
                in_vars=["random_variable_1"],
                out_vars=["random_variable_2"],
            )
        )

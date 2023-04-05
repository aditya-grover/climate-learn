from climate_learn.data.task import TaskArgs, Task


class TestTaskInstantiation:
    def test_initialization(self):
        Task(
            TaskArgs(
                in_vars=["random_variable_1"],
                out_vars=["random_variable_2"],
                constant_names=["random_constant"],
                subsample=3,
            )
        )

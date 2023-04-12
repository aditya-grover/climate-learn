from climate_learn.data.task.args import TaskArgs


class TestTaskArgsInstantiation:
    def test_initialization(self):
        TaskArgs(
            in_vars=["random_variable_1"],
            out_vars=["random_variable_2"],
            constants=["random_constant"],
            subsample=3,
        )

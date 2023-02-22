from climate_learn.data_module.data.args import DataArgs


class TestDataArgsInstantiation:
    def test_initialization(self):
        DataArgs(variables=["random_variable_1", "random_variable_2"], split="train")

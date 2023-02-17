class DataArgs:
    data_class = "Data"

    def __init__(self, variables, split="train"):
        self.variables = variables
        self.split = split

    def setup(self, data_module_args):
        pass

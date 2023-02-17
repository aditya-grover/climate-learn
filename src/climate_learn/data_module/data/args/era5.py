from climate_learn.data_module.data.args import DataArgs
from climate_learn.data_module.data import ERA5

class ERA5Args(DataArgs):
    data_class = ERA5

    def __init__(self, root_dir, variables, years, split="train"):
        super().__init__(variables, split)
        self.root_dir = root_dir
        self.years = years

    def setup(self, data_module_args):
        if self.split == "train":
            self.years = range(data_module_args.train_start_year, data_module_args.val_start_year)
        elif self.split == "val":
            self.years = range(data_module_args.val_start_year, data_module_args.test_start_year)
        elif self.split == "test":
            self.years = range(data_module_args.test_start_year, data_module_args.end_year + 1)
        else:
            raise ValueError(" Invalid split")
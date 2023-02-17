from torch.utils.data import Dataset
from climate_learn.data_module.data.args import DataArgs


class Data(Dataset):
    args_class = DataArgs

    def __init__(self, data_args):
        super().__init__()
        self.variables = data_args.variables
        self.split = data_args.split

    def setup(self):
        pass

    def __getitem__(self, index):
        pass

    def __len__(self):
        pass

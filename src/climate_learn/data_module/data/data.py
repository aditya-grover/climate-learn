from typing import Callable, Sequence
from torch.utils.data import Dataset
from climate_learn.data_module.data.args import DataArgs


class Data(Dataset):
    args_class: Callable[..., DataArgs] = DataArgs

    def __init__(self, data_args: DataArgs) -> None:
        super().__init__()
        self.variables: Sequence[str] = data_args.variables
        self.split: str = data_args.split

    def setup(self) -> None:
        pass

    def __getitem__(self, index: int) -> None:
        pass

    def __len__(self) -> None:
        pass


DataArgs._data_class = Data

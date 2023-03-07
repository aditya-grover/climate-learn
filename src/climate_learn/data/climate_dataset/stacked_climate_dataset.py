from typing import Callable, Sequence
from climate_learn.data.climate_dataset.args import StackedClimateDatasetArgs
from climate_learn.data.climate_dataset import ClimateDataset


class StackedClimateDataset(ClimateDataset):
    _args_class: Callable[..., StackedClimateDatasetArgs] = StackedClimateDatasetArgs

    def __init__(self, data_args: StackedClimateDatasetArgs) -> None:
        self.climate_datasets = []
        for data_arg in data_args.child_data_args:
            if isinstance(data_arg._data_class, str):
                climate_dataset_class = eval(data_arg._data_class)
            else:
                climate_dataset_class = data_arg._data_class
            self.climate_datasets.append(climate_dataset_class(data_arg))

    def setup(self, style="map") -> None:
        dataset_length = []
        variables_to_update = []
        for climate_dataset in self.climate_datasets:
            length, var_to_update = climate_dataset.setup(style)
            dataset_length.append(length)
            variables_to_update.append(var_to_update)
        assert len(set(dataset_length)) == 1
        return dataset_length[0], variables_to_update

    def get_item(self, index):
        return [dataset.get_item(index) for dataset in self.climate_datasets]

    def get_iteritem(self):
        pass

    def get_constants_data(self):
        return [dataset.constants for dataset in self.climate_datasets]

    def get_metadata(self):
        return [dataset.get_metadata() for dataset in self.climate_datasets]


StackedClimateDatasetArgs._data_class = StackedClimateDataset

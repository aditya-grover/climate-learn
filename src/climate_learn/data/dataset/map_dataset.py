# Standard library
from typing import Callable, Dict, Sequence, Tuple, Union

# Third party
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision.transforms import transforms

# Local application
from ..climate_dataset import ClimateDataset
from ..task import Task
from .args import MapDatasetArgs

Data = Dict[str, torch.tensor]
Transform = Dict[str, transforms.Normalize]


class MapDataset(Dataset):
    _args_class: Callable[..., MapDatasetArgs] = MapDatasetArgs

    def __init__(self, dataset_args: MapDatasetArgs) -> None:
        if isinstance(dataset_args.climate_dataset_args._data_class, str):
            climate_dataset_class: Callable[..., ClimateDataset] = eval(
                dataset_args.climate_dataset_args._data_class
            )
        else:
            climate_dataset_class: Callable[
                ..., ClimateDataset
            ] = dataset_args.climate_dataset_args._data_class
        self.data: ClimateDataset = climate_dataset_class(
            dataset_args.climate_dataset_args
        )

        if isinstance(dataset_args.task_args._task_class, str):
            task_class: Callable[..., Task] = eval(dataset_args.task_args._task_class)
        else:
            task_class: Callable[..., Task] = dataset_args.task_args._task_class
        self.task: Task = task_class(dataset_args.task_args)

        self.length: int = 0
        self.climatology: Union[Data, None] = None

    def setup_transforms(self) -> None:
        constants_data: Data = self.data.get_constants_data()
        const_data: Data = self.task.create_constants_data(
            constants_data, apply_transform=0
        )
        mean_const_data: Data = {k: torch.mean(const_data[k]) for k in const_data}
        std_const_data: Data = {k: torch.std(const_data[k]) for k in const_data}
        const_transforms: Transform = {
            k: transforms.Normalize(mean_const_data[k], std_const_data[k])
            for k in const_data
        }

        for index in range(self.length):
            raw_index: Union[Sequence[int], int] = self.task.get_raw_index(index)
            raw_data: Data = self.data.get_item(raw_index)
            inp_data, out_data = self.task.create_inp_out(
                raw_data, constants_data, apply_transform=0
            )
            if index == 0:
                stacked_inp_data = {k: [] for k in inp_data}
                stacked_out_data = {k: [] for k in out_data}
            for k in inp_data:
                stacked_inp_data[k].append(inp_data[k])
            for k in out_data:
                stacked_out_data[k].append(out_data[k])

        stacked_inp_data: Data = {k: torch.stack(stacked_inp_data[k]) for k in inp_data}
        stacked_out_data: Data = {k: torch.stack(stacked_out_data[k]) for k in out_data}

        # Taking mean over entire histories for forecasting
        mean_inp_data: Data = {
            k: torch.mean(stacked_inp_data[k]) for k in stacked_inp_data
        }
        std_inp_data: Data = {
            k: torch.std(stacked_inp_data[k]) for k in stacked_inp_data
        }
        mean_out_data: Data = {
            k: torch.mean(stacked_out_data[k]) for k in stacked_out_data
        }
        std_out_data: Data = {
            k: torch.std(stacked_out_data[k]) for k in stacked_out_data
        }

        inp_transforms: Transform = {
            k: transforms.Normalize(mean_inp_data[k], std_inp_data[k])
            for k in stacked_inp_data
        }
        out_transforms: Transform = {
            k: transforms.Normalize(mean_out_data[k], std_out_data[k])
            for k in stacked_out_data
        }

        self.task.set_normalize(inp_transforms, out_transforms, const_transforms)
        self.climatology: Data = {
            k: torch.mean(stacked_out_data[k], dim=0) for k in stacked_out_data
        }

    def setup(self) -> None:
        data_len, variables_to_update = self.data.setup(style="map")
        self.length = self.task.setup(data_len, variables_to_update)
        self.setup_transforms()

    def get_metadata(self) -> Dict[str, Union[np.ndarray, None]]:
        return self.data.get_metadata()

    def get_climatology(self) -> Union[Data, None]:
        return self.climatology

    def get_data(self) -> Tuple[torch.tensor, torch.tensor, Union[torch.tensor, None]]:
        constants_data: Data = self.data.get_constants_data()
        const_data: Data = self.task.create_constants_data(constants_data)
        data = []
        for index in range(self.length):
            raw_index: Union[Sequence[int], int] = self.task.get_raw_index(index)
            raw_data: Data = self.data.get_item(raw_index)
            data.append(self.task.create_inp_out(raw_data, constants_data))

        def handle_dict_features(t: Data) -> torch.tensor:
            ## Hotfix for the models to work with dict style data
            t = torch.stack(tuple(t.values()))
            ## Handles the case for forecasting input as it has history in it
            ## TODO: Come up with an efficient solution instead of if condition
            if len(t.size()) == 4:
                return torch.transpose(t, 0, 1)
            return t

        inp: torch.tensor = torch.stack(
            [handle_dict_features(data[i][0]) for i in range(len(data))]
        )
        out: torch.tensor = torch.stack(
            [handle_dict_features(data[i][1]) for i in range(len(data))]
        )
        if const_data != {}:
            const: torch.tensor = handle_dict_features(const_data)
        else:
            const = None
        return inp, out, const

    def get_time(self) -> Dict[str, np.ndarray]:
        time_indices: Sequence[int] = [
            self.task.get_time_index(index) for index in range(self.length)
        ]
        time_dict: Dict[str, Union[np.ndarray, None]] = self.data.get_time()
        for key in time_dict.keys():
            if not isinstance(time_dict[key], np.ndarray):
                raise RuntimeError(f"Data hasn't been loaded into the memory yet.")
        return {key: time_dict[key][time_indices] for key in time_dict.keys()}

    def get_transforms(self) -> Tuple[Transform, Transform, Transform]:
        return self.task.get_transforms()

    def set_normalize(
        self,
        inp_transforms: Transform,
        out_transforms: Transform,
        const_transforms: Transform,
    ) -> None:
        self.task.set_normalize(inp_transforms, out_transforms, const_transforms)

    def __getitem__(self, index: int) -> Tuple[Data, Data, Data]:
        constants_data: Data = self.data.get_constants_data()
        const_data: Data = self.task.create_constants_data(constants_data)
        raw_index: Union[Sequence[int], int] = self.task.get_raw_index(index)
        raw_data: Data = self.data.get_item(raw_index)
        inp_data, out_data = self.task.create_inp_out(raw_data, constants_data)
        return inp_data, out_data, const_data

    def __len__(self) -> int:
        return self.length


MapDatasetArgs._data_class = MapDataset

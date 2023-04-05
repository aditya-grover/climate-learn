from typing import Callable, Dict, Sequence, Tuple, Union
from torch.utils.data import Dataset
import torch
import numpy
from torchvision.transforms import transforms
from climate_learn.data.dataset.args import MapDatasetArgs
from climate_learn.data.climate_dataset import ClimateDataset
from climate_learn.data.tasks import Task


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

    def setup(self) -> None:
        data_len, variables_to_update = self.data.setup(style="map")
        #### TODO: Come up with better way to extract lat and lon
        ## HotFix (StackedClimateDataset returns a list instead of dict)
        metadata = self.data.get_metadata()
        if isinstance(metadata, list):  # For downscaling
            self.lat: numpy.ndarray = metadata[0]["lat"]
            self.lon: numpy.ndarray = metadata[0]["lon"]
            self.out_lat: numpy.ndarray = metadata[1]["lat"]
            self.out_lon: numpy.ndarray = metadata[1]["lon"]
        else:
            self.lat: numpy.ndarray = metadata["lat"]
            self.lon: numpy.ndarray = metadata["lon"]
            self.out_lat: numpy.ndarray = metadata["lat"]
            self.out_lon: numpy.ndarray = metadata["lon"]
        self.length: int = self.task.setup(data_len, variables_to_update)
        self.setup_transforms()

    def setup_transforms(self) -> None:
        constants_data: Dict[str, torch.tensor] = self.data.get_constants_data()
        for index in range(self.length):
            raw_index = self.task.get_raw_index(index)
            raw_data = self.data.get_item(raw_index)
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

        stacked_inp_data = {k: torch.stack(stacked_inp_data[k]) for k in inp_data}
        stacked_out_data = {k: torch.stack(stacked_out_data[k]) for k in out_data}

        # Taking mean over entire histories for forecasting
        mean_inp_data = {k: torch.mean(stacked_inp_data[k]) for k in stacked_inp_data}
        std_inp_data = {k: torch.std(stacked_inp_data[k]) for k in stacked_inp_data}
        mean_out_data = {k: torch.mean(stacked_out_data[k]) for k in stacked_out_data}
        std_out_data = {k: torch.std(stacked_out_data[k]) for k in stacked_out_data}

        self.inp_transforms = {
            k: transforms.Normalize(mean_inp_data[k], std_inp_data[k])
            for k in stacked_inp_data
        }
        self.out_transforms = {
            k: transforms.Normalize(mean_out_data[k], std_out_data[k])
            for k in stacked_out_data
        }

        self.task.set_normalize(self.inp_transforms, self.out_transforms)
        self.climatology = {
            k: torch.mean(stacked_out_data[k], dim=0) for k in stacked_out_data
        }

    def get_climatology(self) -> Dict[str, torch.tensor]:
        return self.climatology

    def set_normalize(
        self,
        inp_transforms: Dict[str, transforms.Normalize],
        out_transforms: Dict[str, transforms.Normalize],
    ) -> None:
        self.task.set_normalize(inp_transforms, out_transforms)

    def __getitem__(
        self, index: int
    ) -> Tuple[Dict[str, torch.tensor], Dict[str, torch.tensor]]:
        raw_index: Union[Sequence[int], int] = self.task.get_raw_index(index)
        raw_data: Dict[str, torch.tensor] = self.data.get_item(raw_index)
        constants_data: Dict[str, torch.tensor] = self.data.get_constants_data()
        return self.task.create_inp_out(raw_data, constants_data)

    def __len__(self) -> int:
        return self.length


MapDatasetArgs._data_class = MapDataset

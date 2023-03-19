from torch.utils.data import Dataset
import torch
from torchvision.transforms import transforms
from climate_learn.data.climate_dataset import ClimateDatasetArgs
from climate_learn.data.tasks import TaskArgs


class MapDataset(Dataset):
    def __init__(self, climate_dataset_args: ClimateDatasetArgs, task_args: TaskArgs):
        if isinstance(climate_dataset_args._data_class, str):
            climate_dataset_class = eval(climate_dataset_args._data_class)
        else:
            climate_dataset_class = climate_dataset_args._data_class
        self.data = climate_dataset_class(climate_dataset_args)

        if isinstance(task_args._task_class, str):
            task_class = eval(task_args._task_class)
        else:
            task_class = task_args._task_class
        self.task = task_class(task_args)
        self.setup()

    def setup(self):
        print("Setting up Data")
        data_len, variables_to_update = self.data.setup(style="map")
        #### TODO: Come up with better way to extract lat and lon
        ## HotFix (StackedClimateDataset returns a list instead of dict)
        metadata = self.data.get_metadata()
        if isinstance(metadata, list):  # For downscaling
            self.lat = metadata[0]["lat"]
            self.lon = metadata[0]["lon"]
            self.out_lat = metadata[1]["lat"]
            self.out_lon = metadata[1]["lon"]
        else:
            self.lat = metadata["lat"]
            self.lon = metadata["lon"]
        print("Setting up Task")
        self.length = self.task.setup(data_len, variables_to_update)
        print("Calculating Transforms for the task")
        self.setup_transforms()

    def setup_transforms(self):
        constants_data = self.data.get_constants_data()
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

        print("Stacking data before getting Transforms for the task")
        stacked_inp_data = {k: torch.stack(stacked_inp_data[k]) for k in inp_data}
        stacked_out_data = {k: torch.stack(stacked_out_data[k]) for k in out_data}

        # Taking mean over entire histories for forecasting
        print("Calculating mean and std over the entire data")
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

    def get_climatology(self):
        return self.climatology

    def set_normalize(self, inp_transforms, out_transforms):
        self.task.set_normalize(inp_transforms, out_transforms)

    def __getitem__(self, index):
        raw_index = self.task.get_raw_index(index)
        raw_data = self.data.get_item(raw_index)
        constants_data = self.data.get_constants_data()
        return self.task.create_inp_out(raw_data, constants_data)

    def __len__(self) -> int:
        return self.length

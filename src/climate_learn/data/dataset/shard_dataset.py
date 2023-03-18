import numpy as np
from torch.utils.data import IterableDataset
import torch
from torchvision.transforms import transforms
from climate_learn.data.climate_dataset import ClimateDatasetArgs
from climate_learn.data.tasks import TaskArgs


class ShardedDataset(IterableDataset):
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
        self.n_chunks = 5

    def get_setup_args(self, seed):
        setup_args = {}
        worker_info = torch.utils.data.get_worker_info()
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            rank = torch.distributed.get_rank()
            world_size = torch.distributed.get_world_size()
            worker_info = torch.utils.data.get_worker_info()
            if worker_info is not None:
                # Inside one of the worker process
                num_workers = worker_info.num_workers
                rank = rank * num_workers + worker_info.id
                world_size = world_size * num_workers
        else:
            if worker_info is not None:
                # Not in a distributed setting; Inside one of the worker process
                rank = worker_info.id
                world_size = worker_info.num_workers
            else:
                # Not in a distributed setting; Inside main process
                rank = 0
                world_size = 1
        setup_args["world_size"] = world_size
        setup_args["rank"] = rank
        setup_args["seed"] = seed
        setup_args["n_chunks"] = self.n_chunks
        return setup_args

    def setup_transforms(self):
        constants_data = self.data.get_constants_data()
        transform_data = []
        chunks_iterated_till_now = 0
        while chunks_iterated_till_now < self.n_chunks:
            data_len = self.data.load_chunk(chunks_iterated_till_now)
            length = self.task.setup(data_len)
            # TODO: shuffline logic can go here
            for index in range(length):
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

            mean_inp_data = {
                k: torch.mean(stacked_inp_data[k]) for k in stacked_inp_data
            }
            std_inp_data = {k: torch.std(stacked_inp_data[k]) for k in stacked_inp_data}
            mean_out_data = {
                k: torch.mean(stacked_out_data[k]) for k in stacked_out_data
            }
            std_out_data = {k: torch.std(stacked_out_data[k]) for k in stacked_out_data}

            transform_data.append(
                [mean_inp_data, std_inp_data, mean_out_data, std_out_data, length]
            )
            chunks_iterated_till_now += 1

        ### Using https://math.stackexchange.com/a/37131 to calculate mean and std from chunk mean and std
        # Done to prevent any numerical overflow
        length_gcd = np.gcd.reduce([chunk_data[4] for chunk_data in transform_data])
        reduced_total_length = torch.tensor(0.0)
        for chunk_data in transform_data:
            chunk_data[4] = torch.tensor(chunk_data[4] / length_gcd)
            reduced_total_length += chunk_data[4]

        mean_inp_data = {k: torch.tensor(0.0) for k in transform_data[0][0].keys()}
        mean_out_data = {k: torch.tensor(0.0) for k in transform_data[0][2].keys()}

        for chunk_data in transform_data:
            chunk_length = chunk_data[4]
            mean_inp_data = {
                k: mean_inp_data[k]
                + (chunk_data[0][k] * chunk_length) / reduced_total_length
                for k in mean_inp_data
            }
            mean_out_data = {
                k: mean_out_data[k]
                + (chunk_data[2][k] * chunk_length) / reduced_total_length
                for k in mean_out_data
            }

        std_inp_data = {k: torch.tensor(0.0) for k in transform_data[0][1].keys()}
        std_out_data = {k: torch.tensor(0.0) for k in transform_data[0][3].keys()}

        for chunk_data in transform_data:
            chunk_length = chunk_data[4]
            chunk_std_inp = {
                k: torch.square(chunk_data[1][k])
                + torch.square(chunk_data[0][k] - mean_inp_data[k])
                for k in std_inp_data
            }
            chunk_std_out = {
                k: torch.square(chunk_data[3][k])
                + torch.square(chunk_data[2][k] - mean_out_data[k])
                for k in std_out_data
            }
            std_inp_data = {
                k: std_inp_data[k]
                + (chunk_std_inp[k] * chunk_length) / reduced_total_length
                for k in std_inp_data
            }
            std_out_data = {
                k: std_out_data[k]
                + (chunk_std_out[k] * chunk_length) / reduced_total_length
                for k in std_out_data
            }

        std_inp_data = {k: torch.sqrt(std_inp_data[k]) for k in std_inp_data.keys()}
        std_out_data = {k: torch.sqrt(std_out_data[k]) for k in std_out_data.keys()}

        # Taking mean over entire histories for forecasting
        ## TODO: Need some sort of communication from all the process and then form assert
        ## for Standard deviation some sort of formula would need to be used for effective comm

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

    def setup(self):
        print("Setting up Data")
        setup_args = self.get_setup_args(seed=0)
        data_len, variables_to_update = self.data.setup(
            style="shard", setup_args=setup_args
        )
        #### TODO: Come up with better way to extract lat and lon
        ## HotFix (StackedClimateDataset returns a list instead of dict)
        ## TODO: Need some sort of communication from all the process and then form assert
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
        ## TODO: Need some sort of communication from all the process and then form assert
        _ = self.task.setup(data_len, variables_to_update)
        print("Calculating Transforms for the task")
        self.setup_transforms()

    def __iter__(self):
        ### TODO: get epoch number which then would be used to set seed
        setup_args = self.get_setup_args(seed=0)
        data_len, _ = self.data.setup(style="shard", setup_args=setup_args)
        constants_data = self.data.get_constants_data()
        chunks_iterated_till_now = 0
        while chunks_iterated_till_now < self.n_chunks:
            data_len = self.data.load_chunk(chunks_iterated_till_now)
            length = self.task.setup(data_len)
            # TODO: shuffline logic can go here
            for index in range(length):
                raw_index = self.task.get_raw_index(index)
                raw_data = self.data.get_item(raw_index)
                yield self.task.create_inp_out(raw_data, constants_data)
            chunks_iterated_till_now += 1

# Standard library
import random
from typing import Callable, Dict, Sequence, Tuple, Union

# Third party
import numpy
import torch
from torch.utils.data import IterableDataset
from torchvision.transforms import transforms

# Local application
from climate_learn.data.climate_dataset import ClimateDataset
from climate_learn.data.task import Task
from climate_learn.data.dataset.args import ShardDatasetArgs


class ShardDataset(IterableDataset):
    def __init__(self, dataset_args: ShardDatasetArgs):
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
        self.n_chunks = dataset_args.n_chunks

    def get_setup_args(self, seed: int) -> Dict[str, int]:
        setup_args: Dict[str, int] = {}
        worker_info = torch.utils.data.get_worker_info()
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            rank: int = torch.distributed.get_rank()
            world_size: int = torch.distributed.get_world_size()
            if worker_info is not None:
                # Inside one of the worker process
                sett = "Distributed is on; worker process"
                info = f"worker id: {worker_info.id}, num_workers: {worker_info.num_workers}, rank: {rank}, world_size: {world_size}"
                # print(f"{sett} {info}")
                num_workers: int = worker_info.num_workers
                rank = rank * num_workers + worker_info.id
                world_size = world_size * num_workers
            else:
                sett = "Distributed is on; main process"
                info = f"rank: {rank}, world_size: {world_size}"
                # print(f"{sett} {info}")
        else:
            if worker_info is not None:
                # Not in a distributed setting; Inside one of the worker process
                sett = "Distributed is off; worker process"
                info = f"worker id: {worker_info.id}, num_workers: {worker_info.num_workers}"
                # print(f"{sett} {info}")
                rank: int = worker_info.id
                world_size: int = worker_info.num_workers
            else:
                # Not in a distributed setting; Inside main process
                sett = "Distributed is off; main process"
                info = f"Nothing to look at"
                # print(f"{sett} {info}")
                rank: int = 0
                world_size: int = 1
        setup_args["world_size"] = world_size
        setup_args["rank"] = rank
        setup_args["seed"] = seed
        setup_args["n_chunks"] = self.n_chunks
        return setup_args

    def setup_transforms(self) -> None:
        constants_data: Dict[str, torch.tensor] = self.data.get_constants_data()
        transform_data: Sequence[
            Tuple[
                Dict[str, torch.tensor],
                Dict[str, torch.tensor],
                Dict[str, torch.tensor],
                Dict[str, torch.tensor],
                int,
            ]
        ] = []
        chunks_iterated_till_now: int = 0
        while chunks_iterated_till_now < self.n_chunks:
            data_len: int = self.data.load_chunk(chunks_iterated_till_now)
            length: int = self.task.setup(data_len)
            for index in range(length):
                raw_index: Union[Sequence[int], int] = self.task.get_raw_index(index)
                raw_data: Dict[str, torch.tensor] = self.data.get_item(raw_index)
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
        # GCD used to prevent any numerical overflow
        length_gcd: int = numpy.gcd.reduce(
            [chunk_data[4] for chunk_data in transform_data]
        )
        reduced_total_length: torch.tensor = torch.tensor(0.0)
        for chunk_data in transform_data:
            chunk_data[4] = torch.tensor(chunk_data[4] / length_gcd)
            reduced_total_length += chunk_data[4]

        mean_inp_data: Dict[str, torch.tensor] = {
            k: torch.tensor(0.0) for k in transform_data[0][0].keys()
        }
        mean_out_data: Dict[str, torch.tensor] = {
            k: torch.tensor(0.0) for k in transform_data[0][2].keys()
        }

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

        std_inp_data: Dict[str, torch.tensor] = {
            k: torch.tensor(0.0) for k in transform_data[0][1].keys()
        }
        std_out_data: Dict[str, torch.tensor] = {
            k: torch.tensor(0.0) for k in transform_data[0][3].keys()
        }

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

        self.inp_transforms: Dict[str, transforms.Normalize] = {
            k: transforms.Normalize(mean_inp_data[k], std_inp_data[k])
            for k in stacked_inp_data
        }
        self.out_transforms: Dict[str, transforms.Normalize] = {
            k: transforms.Normalize(mean_out_data[k], std_out_data[k])
            for k in stacked_out_data
        }

        self.task.set_normalize(self.inp_transforms, self.out_transforms)
        self.climatology: Dict[str, torch.tensor] = {
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

    def setup(self) -> None:
        setup_args: Dict[str, int] = self.get_setup_args(seed=0)
        data_len, variables_to_update = self.data.setup(
            style="shard", setup_args=setup_args
        )
        #### TODO: Come up with better way to extract lat and lon
        ## HotFix (StackedClimateDataset returns a list instead of dict)
        ## TODO: Need some sort of communication from all the process and then form assert
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
        ## TODO: Need some sort of communication from all the process and then form assert
        _ = self.task.setup(data_len, variables_to_update)
        self.setup_transforms()
        self.epoch: int = 0

    def __iter__(self):
        ### TODO: get epoch number which then would be used to set seed
        ## Hacky solution for now; Ideally should get that from Trainer
        self.epoch += 1
        setup_args: Dict[str, int] = self.get_setup_args(seed=self.epoch)
        data_len, _ = self.data.setup(style="shard", setup_args=setup_args)
        constants_data: Dict[str, torch.tensor] = self.data.get_constants_data()
        chunks_iterated_till_now: int = 0
        while chunks_iterated_till_now < self.n_chunks:
            data_len: int = self.data.load_chunk(chunks_iterated_till_now)
            length: int = self.task.setup(data_len)

            # TODO: shuffline logic can go here
            indices: Sequence[int] = list(range(length))
            random.Random(self.epoch).shuffle(indices)
            for index in indices:
                raw_index: Union[Sequence[int], int] = self.task.get_raw_index(index)
                raw_data: Dict[str, torch.tensor] = self.data.get_item(raw_index)
                yield self.task.create_inp_out(raw_data, constants_data)
            chunks_iterated_till_now += 1


ShardDatasetArgs._data_class = ShardDataset

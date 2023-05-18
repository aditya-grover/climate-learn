# Standard library
import random
from typing import Callable, Dict, Sequence, Tuple, Union

# Third party
import numpy as np
import torch
from torch.utils.data import IterableDataset
from torchvision.transforms import transforms

# Local application
from ..climate_dataset import ClimateDataset
from ..task import Task
from .args import ShardDatasetArgs

Data = Dict[str, torch.tensor]
Transform = Dict[str, transforms.Normalize]


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
        self.n_chunks: int = dataset_args.n_chunks

        self.climatology: Union[Data, None] = None
        self.epoch: int = 0
        self.shuffle: bool = True
        self.drop_last: bool = False

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
                num_workers: int = worker_info.num_workers
                rank = rank * num_workers + worker_info.id
                world_size = world_size * num_workers
            else:
                sett = "Distributed is on; main process"
                info = f"rank: {rank}, world_size: {world_size}"
        else:
            if worker_info is not None:
                # Not in a distributed setting; Inside one of the worker process
                sett = "Distributed is off; worker process"
                info = f"worker id: {worker_info.id}, num_workers: {worker_info.num_workers}"
                rank: int = worker_info.id
                world_size: int = worker_info.num_workers
            else:
                # Not in a distributed setting; Inside main process
                sett = "Distributed is off; main process"
                info = f"Nothing to look at"
                rank: int = 0
                world_size: int = 1
        setup_args["world_size"] = world_size
        setup_args["rank"] = rank
        setup_args["seed"] = seed
        setup_args["n_chunks"] = self.n_chunks
        if self.drop_last:
            setup_args["drop_last"] = True
        if self.shuffle:
            setup_args["shuffle"] = True
        return setup_args

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
        transform_data: Sequence[Tuple[Data, Data, Data, Data, int]] = []
        chunks_iterated_till_now: int = 0
        while chunks_iterated_till_now < self.n_chunks:
            data_len: int = self.data.load_chunk(chunks_iterated_till_now)
            length: int = self.task.setup(data_len)
            for index in range(length):
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
            stacked_inp_data: Data = {
                k: torch.stack(stacked_inp_data[k]) for k in inp_data
            }
            stacked_out_data: Data = {
                k: torch.stack(stacked_out_data[k]) for k in out_data
            }

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
            climatology_data: Data = {
                k: torch.mean(stacked_out_data[k], dim=0) for k in stacked_out_data
            }

            transform_data.append(
                [
                    mean_inp_data,
                    std_inp_data,
                    mean_out_data,
                    std_out_data,
                    climatology_data,
                    length,
                ]
            )
            chunks_iterated_till_now += 1

        ### Using https://math.stackexchange.com/a/37131 to calculate mean and std from chunk mean and std
        # GCD used to prevent any numerical overflow
        length_gcd: int = np.gcd.reduce(
            [chunk_data[5] for chunk_data in transform_data]
        )
        reduced_total_length: torch.tensor = torch.tensor(0.0)
        for chunk_data in transform_data:
            chunk_data[5] = torch.tensor(chunk_data[5] / length_gcd)
            reduced_total_length += chunk_data[5]

        mean_inp_data: Data = {
            k: torch.tensor(0.0) for k in transform_data[0][0].keys()
        }
        mean_out_data: Data = {
            k: torch.tensor(0.0) for k in transform_data[0][2].keys()
        }
        climatology_data_shape: torch.Size = next(
            iter(transform_data[0][4].values())
        ).size()
        climatology_data: Data = {
            k: torch.zeros(climatology_data_shape) for k in transform_data[0][4].keys()
        }

        for chunk_data in transform_data:
            chunk_length = chunk_data[5]
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
            climatology_data = {
                k: climatology_data[k]
                + (chunk_data[4][k] * chunk_length) / reduced_total_length
                for k in climatology_data
            }

        std_inp_data: Data = {k: torch.tensor(0.0) for k in transform_data[0][1].keys()}
        std_out_data: Data = {k: torch.tensor(0.0) for k in transform_data[0][3].keys()}

        for chunk_data in transform_data:
            chunk_length = chunk_data[5]
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

        std_inp_data: Data = {
            k: torch.sqrt(std_inp_data[k]) for k in std_inp_data.keys()
        }
        std_out_data: Data = {
            k: torch.sqrt(std_out_data[k]) for k in std_out_data.keys()
        }

        # Taking mean over entire histories for forecasting
        ## TODO: Need some sort of communication from all the process and then form assert
        ## for Standard deviation some sort of formula would need to be used for effective comm

        inp_transforms: Transform = {
            k: transforms.Normalize(mean_inp_data[k], std_inp_data[k])
            for k in stacked_inp_data
        }
        out_transforms: Transform = {
            k: transforms.Normalize(mean_out_data[k], std_out_data[k])
            for k in stacked_out_data
        }

        self.task.set_normalize(inp_transforms, out_transforms, const_transforms)
        self.climatology: Data = climatology_data

    def setup(self) -> None:
        setup_args: Dict[str, int] = self.get_setup_args(seed=0)
        del setup_args["shuffle"]
        data_len, variables_to_update = self.data.setup(
            style="shard", setup_args=setup_args
        )
        ## TODO: Need some sort of communication from all the process and then form assert
        _ = self.task.setup(data_len, variables_to_update)
        self.setup_transforms()
        self.epoch = 0

    def get_metadata(self) -> Dict[str, Union[np.ndarray, None]]:
        return self.data.get_metadata()

    def get_climatology(self) -> Union[Data, None]:
        return self.climatology

    def get_data(
        self, entire_data: bool = False
    ) -> Tuple[torch.tensor, torch.tensor, Union[torch.tensor, None]]:
        setup_args: Dict[str, int] = self.get_setup_args(seed=0)
        del setup_args["shuffle"]
        data_len, _ = self.data.setup(style="shard", setup_args=setup_args)
        constants_data: Data = self.data.get_constants_data()
        const_data: Data = self.task.create_constants_data(constants_data)
        ## We want last temporal data if we shard it
        chunks_iterated_till_now: int = self.n_chunks - 1
        if entire_data:
            chunks_iterated_till_now = 0
        data = []
        while chunks_iterated_till_now < self.n_chunks:
            data_len: int = self.data.load_chunk(chunks_iterated_till_now)
            length: int = self.task.setup(data_len)
            indices: Sequence[int] = list(range(length))
            for index in indices:
                raw_index: Union[Sequence[int], int] = self.task.get_raw_index(index)
                raw_data: Data = self.data.get_item(raw_index)
                data.append(self.task.create_inp_out(raw_data, constants_data))
            chunks_iterated_till_now += 1

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

    def get_time(self) -> np.ndarray:
        time_dict: Dict[str, Union[np.ndarray, None]] = self.data.get_time()
        for key in time_dict.keys():
            if not isinstance(time_dict[key], np.ndarray):
                raise RuntimeError(f"Data hasn't been loaded into the memory yet.")
        random_key: str = next(iter(time_dict.keys()))
        data_len: int = len(time_dict[random_key])
        length: int = self.task.setup(data_len)
        time_indices: Sequence[int] = [
            self.task.get_time_index(index) for index in range(length)
        ]
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

    def __iter__(self) -> Tuple[Data, Data, Data]:
        ### TODO: get epoch number which then would be used to set seed
        ## Hacky solution for now; Ideally should get that from Trainer
        self.epoch += 1
        setup_args: Dict[str, int] = self.get_setup_args(seed=self.epoch)
        data_len, _ = self.data.setup(style="shard", setup_args=setup_args)
        constants_data: Data = self.data.get_constants_data()
        const_data: Data = self.task.create_constants_data(constants_data)
        chunks_iterated_till_now: int = 0
        while chunks_iterated_till_now < self.n_chunks:
            data_len: int = self.data.load_chunk(chunks_iterated_till_now)
            length: int = self.task.setup(data_len)
            indices: Sequence[int] = list(range(length))
            random.Random(self.epoch).shuffle(indices)
            for index in indices:
                raw_index: Union[Sequence[int], int] = self.task.get_raw_index(index)
                raw_data: Data = self.data.get_item(raw_index)
                inp_data, out_data = self.task.create_inp_out(raw_data, constants_data)
                yield inp_data, out_data, const_data
            chunks_iterated_till_now += 1


ShardDatasetArgs._data_class = ShardDataset

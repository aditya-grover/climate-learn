import math
import random

import numpy as np
import torch
from torch.utils.data import IterableDataset


class ERA5Npy(IterableDataset):
    def __init__(self, file_list, variables, out_variables, shuffle: bool = False) -> None:
        super().__init__()
        self.file_list = file_list
        self.variables = variables
        self.out_variables = out_variables if out_variables is not None else variables
        self.shuffle = shuffle

    def __iter__(self):
        if self.shuffle:
            random.shuffle(self.file_list)
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is None:
            iter_start = 0
            iter_end = len(self.file_list)
        else:
            if not torch.distributed.is_initialized():
                rank = 0
                world_size = 1
            else:
                rank = torch.distributed.get_rank()
                world_size = torch.distributed.get_world_size()
            num_workers_per_ddp = worker_info.num_workers
            num_shards = num_workers_per_ddp * world_size
            per_worker = int(math.floor(len(self.file_list) / float(num_shards)))
            worker_id = rank * num_workers_per_ddp + worker_info.id
            iter_start = worker_id * per_worker
            # iter_end = min(iter_start + per_worker, len(self.file_list))
            iter_end = iter_start + per_worker

        # print(f"rank {rank}")
        # print(f"world size {world_size}")
        # print(f"len data {len(self.file_list)}")
        # print(f"start {iter_start}, end {iter_end}")

        # count the number of data points this worker holds
        # num_data = 0
        # for idx in range(iter_start, iter_end):
        #     data = np.load(self.file_list[idx])
        #     num_data += data["t2m"].shape[0]

        # print(f"rank {rank}")
        # print(f"{num_data} data points")

        # print("==============================")

        for idx in range(iter_start, iter_end):
            path = self.file_list[idx]
            data = np.load(path)
            yield {k: data[k] for k in self.variables}, self.variables, self.out_variables


class ERA5Forecast(IterableDataset):
    def __init__(
        self, dataset: ERA5Npy, predict_range: int = 6, history: int = 3, interval: int = 6, subsample: int = 1
    ) -> None:
        super().__init__()
        self.dataset = dataset
        self.predict_range = predict_range
        self.history = history
        self.interval = interval
        self.subsample = subsample

    def __iter__(self):
        # TODO: this would not get us stuff across the years
        # i.e. where inputs are from previous years and output from next
        for data, variables, out_variables in self.dataset:
            x = np.concatenate([data[k].astype(np.float32) for k in data.keys()], axis=1)
            x = torch.from_numpy(x)
            y = np.concatenate([data[k].astype(np.float32) for k in out_variables], axis=1)
            y = torch.from_numpy(y)

            inputs = x.unsqueeze(0).repeat_interleave(self.history, dim=0)
            for t in range(self.history):
                inputs[t] = inputs[t].roll(-t * self.interval, dims=0)

            last_idx = -((self.history - 1) * self.interval + self.predict_range)

            outputs = y.roll(last_idx, dims=0)

            inputs = inputs[:, :last_idx].transpose(0, 1)  # N, T, C, H, W
            outputs = outputs[:last_idx]  # N, C, H, W

            inputs = inputs[:: self.subsample]
            outputs = outputs[:: self.subsample]

            yield inputs, outputs, variables, out_variables


class ERA5ForecastMultiStep(IterableDataset):
    def __init__(
        self,
        dataset: ERA5Npy,
        pred_range: int = 6,
        history: int = 3,
        interval: int = 6,
        pred_steps: int = 4,
        subsample: int = 1,
    ) -> None:
        super().__init__()
        self.dataset = dataset
        self.pred_range = pred_range
        self.history = history
        self.interval = interval
        self.pred_steps = pred_steps
        self.subsample = subsample

    def __iter__(self):
        # TODO: this would not get us stuff across the years
        # i.e. where inputs are from previous years and output from next
        for data, variables, out_variables in self.dataset:
            x = np.concatenate([data[k].astype(np.float32) for k in data.keys()], axis=1)
            x = torch.from_numpy(x)
            y = np.concatenate([data[k].astype(np.float32) for k in out_variables], axis=1)
            y = torch.from_numpy(y)

            inputs = x.unsqueeze(0).repeat_interleave(self.history, dim=0)
            for t in range(self.history):
                inputs[t] = inputs[t].roll(-t * self.interval, dims=0)

            outputs = y.unsqueeze(0).repeat_interleave(self.pred_steps, dim=0)
            start_idx = (self.history - 1) * self.interval + self.pred_range
            for t in range(self.pred_steps):
                outputs[t] = outputs[t].roll(-(start_idx + t * self.pred_range), dims=0)

            last_idx = -((self.history - 1) * self.interval + self.pred_steps * self.pred_range)

            inputs = inputs[:, :last_idx].transpose(0, 1)  # N, T1, C, H, W
            outputs = outputs[:, :last_idx].transpose(0, 1)  # N, T2, C, H, W

            inputs = inputs[:: self.subsample]
            outputs = outputs[:: self.subsample]

            yield inputs, outputs, variables, out_variables


class IndividualForecastDataIter(IterableDataset):
    def __init__(self, dataset: ERA5Forecast, transforms: torch.nn.Module, output_transforms: torch.nn.Module):
        super().__init__()
        self.dataset = dataset
        self.transforms = transforms
        self.output_transforms = output_transforms

    def __iter__(self):
        for (inp, out, variables, out_variables) in self.dataset:
            assert inp.shape[0] == out.shape[0]
            for i in range(inp.shape[0]):
                # TODO: should we unsqueeze the first dimension?
                if self.transforms is not None:
                    yield self.transforms(inp[i]), self.output_transforms(out[i]), variables, out_variables
                else:
                    yield inp[i], out[i], variables, out_variables


class ShuffleIterableDataset(IterableDataset):
    def __init__(self, dataset, buffer_size: int) -> None:
        super().__init__()
        assert buffer_size > 0
        self.dataset = dataset
        self.buffer_size = buffer_size

    def __iter__(self):
        buf = []
        for x in self.dataset:
            if len(buf) == self.buffer_size:
                idx = random.randint(0, self.buffer_size - 1)
                yield buf[idx]
                buf[idx] = x
            else:
                buf.append(x)
        random.shuffle(buf)
        while buf:
            yield buf.pop()

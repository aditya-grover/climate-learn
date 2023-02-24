from typing import Any, Callable, Sequence, Tuple
import torch
import numpy as np
import xarray as xr
from torchvision.transforms import transforms

from climate_learn.data.climate_dataset import *
from climate_learn.data.tasks.task import Task
from climate_learn.data.tasks.args import DownscalingArgs


class Downscaling(Task):
    args_class: Callable[..., DownscalingArgs] = DownscalingArgs

    def __init__(self, task_args: DownscalingArgs) -> None:
        super().__init__(task_args)
        if isinstance(task_args.highres_dataset_args._data_class, str):
            highres_dataset_class: Callable[..., ClimateDataset] = eval(
                task_args.highres_dataset_args._data_class
            )
        else:
            highres_dataset_class: Callable[
                ..., ClimateDataset
            ] = task_args.highres_dataset_args._data_class

        self.highres_dataset: ClimateDataset = highres_dataset_class(
            task_args.highres_dataset_args
        )

        assert set(self.in_vars) <= set(self.dataset.variables)
        assert set(self.out_vars) <= set(self.highres_dataset.variables)

    def setup(self) -> None:
        super().setup()
        if len(self.constant_names) > 0:
            assert set(self.constant_names) <= set(self.dataset.constant_names)
        self.highres_dataset.setup()
        inp_data = xr.concat(
            [self.dataset.data_dict[k] for k in self.in_vars], dim="level"
        )
        out_data = xr.concat(
            [self.highres_dataset.data_dict[k] for k in self.out_vars], dim="level"
        )

        self.inp_data: np.ndarray = (
            inp_data[:: self.subsample].to_numpy().astype(np.float32)
        )
        self.out_data: np.ndarray = (
            out_data[:: self.subsample].to_numpy().astype(np.float32)
        )

        constants_data = [
            self.dataset.constants[k].to_numpy().astype(np.float32)
            for k in self.constant_names
        ]
        if len(constants_data) > 0:
            self.constants_data: Union[np.ndarray, None] = np.stack(
                constants_data, axis=0
            )  # 3, 32, 64
        else:
            self.constants_data: Union[np.ndarray, None] = None

        assert len(self.inp_data) == len(self.out_data)

        # why is this a single number instead of a tuple
        self.downscale_ratio: Any = (
            self.out_data.shape[-1] // self.inp_data.shape[-1]
        )  # TODO add stronger typecheck

        if self.split == "train":
            self.inp_transform: Union[transforms.Normalize, None] = self.get_normalize(
                self.inp_data
            )
            self.out_transform: Union[transforms.Normalize, None] = self.get_normalize(
                self.out_data
            )
            self.constant_transform: Union[transforms.Normalize, None] = (
                self.get_normalize(np.expand_dims(self.constants_data, axis=0))
                if self.constants_data is not None
                else None
            )
        else:
            self.inp_transform: Union[transforms.Normalize, None] = None
            self.out_transform: Union[transforms.Normalize, None] = None
            self.constant_transform: Union[transforms.Normalize, None] = None

        self.time: np.ndarray = (
            self.dataset.data_dict[self.in_vars[0]]
            .time.to_numpy()[:: self.subsample]
            .copy()
        )
        self.inp_lon: np.ndarray = self.dataset.lon
        self.inp_lat: np.ndarray = self.dataset.lat
        self.out_lon: np.ndarray = self.highres_dataset.lon
        self.out_lat: np.ndarray = self.highres_dataset.lat

        del self.dataset.data_dict
        del self.highres_dataset.data_dict

    def get_climatology(self) -> torch.Tensor:
        return torch.from_numpy(self.out_data.mean(axis=0))

    def __getitem__(
        self, index
    ) -> Tuple[np.ndarray, np.ndarray, Sequence[str], Sequence[str]]:
        inp = torch.from_numpy(self.inp_data[index])
        out = torch.from_numpy(self.out_data[index])
        return (
            self.inp_transform(inp),
            self.out_transform(out),
            self.in_vars,
            self.out_vars,
        )

    def __len__(self) -> int:
        return len(self.inp_data)


DownscalingArgs._task_class = Downscaling

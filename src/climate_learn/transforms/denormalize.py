# Standard library
from typing import Union

# Local application
from .registry import register
from ..data import DataModule, IterDataModule

# Third party
import torch
from torch.distributions.normal import Normal
from torchvision import transforms
import numpy as np


@register("denormalize")
class Denormalize:
    def __init__(self, data_module: Union[DataModule, IterDataModule]):
        super().__init__()
        norm = data_module.get_out_transforms()
        if norm is None:
            raise RuntimeError("norm was 'None', did you setup the data module?")
        # Hotfix to work with dict style data
        # mean_norm = torch.tensor([norm[k].mean for k in norm.keys()])
        # std_norm = torch.tensor([norm[k].std for k in norm.keys()])
        mean_norm = norm.mean
        std_norm = norm.std
        std_denorm = 1 / std_norm
        mean_denorm = -mean_norm * std_denorm
        self.transform = transforms.Normalize(mean_denorm, std_denorm)

    def __call__(self, x) -> Union[torch.FloatTensor, torch.DoubleTensor]:
        return self.transform(x)


@register("denormalize_gaussian")
class DenormalizeGaussian:
    def __init__(self, data_module: Union[DataModule, IterDataModule]):
        super().__init__()
        norm = data_module.get_out_transforms()
        if norm is None:
            raise RuntimeError("norm was 'None', did you setup the data module?")
        # Hotfix to work with dict style data
        # mean_norm = torch.tensor([norm[k].mean for k in norm.keys()])
        # std_norm = torch.tensor([norm[k].std for k in norm.keys()])
        mean_norm = norm.mean
        std_norm = norm.std
        std_denorm = 1 / std_norm
        mean_denorm = -mean_norm * std_denorm
        self.transform_loc = transforms.Normalize(mean_denorm, std_denorm)
        self.transform_scale = transforms.Normalize(np.zeros_like(mean_denorm), std_denorm)

    def __call__(self, x):
        if isinstance(x, Normal):
            loc, scale = x.loc, x.scale
            loc = self.transform_loc(loc)
            scale = self.transform_scale(scale)
            return Normal(loc, scale)
        else:
            return self.transform_loc(x)
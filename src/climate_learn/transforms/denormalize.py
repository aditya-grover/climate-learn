# Standard library
from typing import Union

# Local application
from .registry import register
from ..data import IterDataModule

# Third party
import torch
from torchvision import transforms


@register("denormalize")
class Denormalize:
    def __init__(self, data_module: IterDataModule):
        norm = data_module.get_out_transforms()
        if norm is None:
            raise RuntimeError("norm was 'None', did you setup the data module?")
        # Hotfix to work with dict style data
        if isinstance(norm, dict):
            mean_norm = torch.tensor([norm[k].mean for k in norm.keys()])
            std_norm = torch.tensor([norm[k].std for k in norm.keys()])
        else:
            mean_norm = norm.mean
            std_norm = norm.std
        std_denorm = 1 / std_norm
        mean_denorm = -mean_norm * std_denorm
        self.transform = transforms.Normalize(mean_denorm, std_denorm)

    def __call__(self, x) -> Union[torch.FloatTensor, torch.DoubleTensor]:
        return self.transform(x)

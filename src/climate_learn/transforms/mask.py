# Standard library
from typing import Union

# Local application
from .registry import register

# Third party
import torch


@register("mask")
class Mask:
    def __init__(self, mask: torch.IntTensor):
        self.mask = mask

    def __call__(self, x) -> Union[torch.FloatTensor, torch.DoubleTensor]:
        res = torch.where(self.mask == 1, x, 0)
        return res

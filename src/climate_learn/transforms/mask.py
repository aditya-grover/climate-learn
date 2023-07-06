# Standard library
from typing import Union

# Local application
from .registry import register

# Third party
import torch


@register("mask")
class Mask:
    def __init__(self, mask: torch.IntTensor, val=0):
        self.mask = mask
        self.val = val

    def __call__(self, x) -> Union[torch.FloatTensor, torch.DoubleTensor]:
        self.mask = self.mask.to(x.device)
        res = torch.where(self.mask == 1, x, self.val)
        return res

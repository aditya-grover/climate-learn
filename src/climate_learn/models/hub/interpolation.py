# Local application
from .utils import register

# Third party
from torch import nn
import torch.nn.functional as F


@register("interpolation")
class Interpolation(nn.Module):
    def __init__(self, size, mode):
        super().__init__()
        self.size = size
        self.mode = mode

    def forward(self, x):
        yhat = F.interpolate(x, self.size, mode=self.mode)
        return yhat

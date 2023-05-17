# Standard library
from typing import Iterable, Optional

# Local application
from .utils import register

# Third party
from torch import nn


@register("persistence")
class Persistence(nn.Module):
    def __init__(self, channels: Optional[Iterable[int]] = None):
        """
        :param channels: The indices of the channels to forward from the input
            to the output.
        """
        super().__init__()
        self.channels = channels

    def forward(self, x):
        # x.shape = [B,T,in_channels,H,W]
        if self.channels:
            yhat = x[:, -1, self.channels]
        else:
            yhat = x[:, -1]
        # yhat.shape = [B,out_channels,H,W]
        return yhat

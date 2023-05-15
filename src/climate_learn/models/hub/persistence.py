# Local application
from .utils import register

# Third party
from torch import nn


@register("persistence")
class Persistence(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        # x.shape = [B,T,C,H,W]
        yhat = x[:, -1]
        # yhat.shape = [B,C,H,W]
        return yhat

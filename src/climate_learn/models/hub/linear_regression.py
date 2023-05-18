# Local application
from .utils import register

# Third party
from torch import nn


@register("linear-regression")
class LinearRegression(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)

    def forward(self, x):
        # x.shape = [B,T,C,H,W]
        batch_size = x.shape[0]
        height = x.shape[3]
        width = x.shape[4]
        # x.shape = [B,T*C*H*W]
        x = x.flatten(1)
        # yhat.shape = [B,C*H*W]
        yhat = self.linear(x)
        yhat = yhat.reshape(batch_size, -1, height, width)
        return yhat

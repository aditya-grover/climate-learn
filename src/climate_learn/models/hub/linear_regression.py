# Third party
from torch import nn


class LinearRegression(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)

    def forward(self, x):
        target_shape = x[:, 0].shape  # not including time dimension
        x = x.flatten(1)  # flatten along all but the batch dimension
        yhat = self.linear(x)
        yhat = yhat.reshape(target_shape)
        return yhat

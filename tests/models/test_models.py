# Standard library
import pytest

# Local application
from climate_learn.models.hub import (
    Climatology,
    Persistence,
    LinearRegression,
    ResNet,
    Unet,
    VisionTransformer,
    Interpolation,
)

# Third party
import torch


class TestForecastingModels:
    num_batches = 32
    history = 3
    num_channels = 2
    width = 32
    height = 64
    x = torch.randn((num_batches, history, num_channels, width, height))
    y = torch.randn((num_batches, num_channels, width, height))

    def test_climatology(self):
        clim = torch.zeros((self.num_channels, self.width, self.height))
        model = Climatology(clim)
        assert model(self.x).shape == self.y.shape

    def test_persistence(self):
        model = Persistence()
        assert model(self.x).shape == self.y.shape

    def test_linear_regresion(self):
        in_ftrs = self.history * self.num_channels * self.width * self.height
        out_ftrs = self.num_channels * self.width * self.height
        model = LinearRegression(in_ftrs, out_ftrs)
        assert model(self.x).shape == self.y.shape

    def test_resnet(self):
        model = ResNet(self.num_channels, self.num_channels, self.history)
        assert model(self.x).shape == self.y.shape

    def test_unet(self):
        model = Unet(self.num_channels, self.num_channels, self.history)
        assert model(self.x).shape == self.y.shape

    def test_vit(self):
        model = VisionTransformer(
            (self.width, self.height),
            self.num_channels,
            self.num_channels,
            self.history,
        )
        assert model(self.x).shape == self.y.shape

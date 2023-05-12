# Local application
from climate_learn.models.hub import (
    Climatology,
    Interpolation,
    LinearRegression,
    Persistence,
    ResNet,
    Unet,
    VisionTransformer,
)

# Third party
import pytest
import torch


class TestForecastingModels:
    num_batches = 32
    history = 3
    num_channels = 2
    width, height = 32, 64
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

    @pytest.mark.skip(reason="ViT is broken, will fix in future PR")
    def test_vit(self):
        model = VisionTransformer(
            (self.width, self.height),
            self.num_channels,
            self.num_channels,
            self.history,
        )
        assert model(self.x).shape == self.y.shape


class TestDownscalingModels:
    num_batches = 32
    history = 3
    num_channels = 2
    out_width, out_height = 32, 64
    downscaling_factor = 0.8
    in_width = int(downscaling_factor * out_width)
    in_height = int(downscaling_factor * out_height)
    x = torch.randn((num_batches, num_channels, in_width, in_height))
    y = torch.randn((num_batches, num_channels, out_width, out_height))

    @pytest.mark.parametrize("mode", ["bilinear", "nearest"])
    def test_interpolation(self, mode):
        model = Interpolation((self.out_width, self.out_height), mode)
        assert model(self.x).shape == self.y.shape

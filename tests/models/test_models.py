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
    out_channels = 1
    height, width = 32, 64
    x = torch.randn((num_batches, history, num_channels, height, width))
    y_same_channels = torch.randn((num_batches, num_channels, height, width))
    y_diff_channels = torch.randn((num_batches, out_channels, height, width))

    def test_climatology(self):
        clim = torch.zeros((self.num_channels, self.height, self.width))
        mean = std = torch.ones_like(clim)
        model = Climatology(clim, mean, std)
        assert model(self.x).shape == self.y_same_channels.shape

    @pytest.mark.parametrize("same_out_channels", [True, False])
    def test_persistence(self, same_out_channels: bool):
        if same_out_channels:
            model = Persistence()
            target = self.y_same_channels
        else:
            model = Persistence(range(self.out_channels))
            target = self.y_diff_channels
        assert model(self.x).shape == target.shape

    @pytest.mark.parametrize("same_out_channels", [True, False])
    def test_linear_regresion(self, same_out_channels: bool):
        in_ftrs = self.history * self.num_channels * self.width * self.height
        if same_out_channels:
            out_ftrs = self.num_channels * self.width * self.height
            target = self.y_same_channels
        else:
            out_ftrs = self.out_channels * self.width * self.height
            target = self.y_diff_channels
        model = LinearRegression(in_ftrs, out_ftrs)
        assert model(self.x).shape == target.shape

    @pytest.mark.parametrize("same_out_channels", [True, False])
    def test_resnet(self, same_out_channels: bool):
        if same_out_channels:
            out_channels = self.num_channels
            target = self.y_same_channels
        else:
            out_channels = self.out_channels
            target = self.y_diff_channels
        model = ResNet(self.num_channels, out_channels, self.history)
        assert model(self.x).shape == target.shape

    @pytest.mark.parametrize("same_out_channels", [True, False])
    def test_unet(self, same_out_channels: bool):
        if same_out_channels:
            out_channels = self.num_channels
            target = self.y_same_channels
        else:
            out_channels = self.out_channels
            target = self.y_diff_channels
        model = Unet(self.num_channels, out_channels, self.history)
        assert model(self.x).shape == target.shape

    @pytest.mark.parametrize("same_out_channels", [True, False])
    def test_vit(self, same_out_channels):
        if same_out_channels:
            out_channels = self.num_channels
            target = self.y_same_channels
        else:
            out_channels = self.out_channels
            target = self.y_diff_channels
        model = VisionTransformer(
            (self.height, self.width),
            self.num_channels,
            out_channels,
            self.history,
        )
        assert model(self.x).shape == target.shape


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

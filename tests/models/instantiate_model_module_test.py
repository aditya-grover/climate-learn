from climate_learn.models.modules import DownscaleLitModule, ForecastLitModule
from climate_learn.models.components import *
import torch


class TestLitModuleInstantiation:
    def test_forecast(self):
        ForecastLitModule(
            ResNet(in_channels=1),
            optimizer=torch.optim.Adam,
        )

    def test_downscale(self):
        DownscaleLitModule(
            Unet(in_channels=1),
            optimizer=torch.optim.AdamW,
        )

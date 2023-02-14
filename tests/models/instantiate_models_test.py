from climate_learn.models.components import *


class TestModelInstantiation:
    def test_vit(self):
        VisionTransformer()

    def test_resnet(self):
        ResNet(in_channels=1)

    def test_unet(self):
        Unet(in_channels=1)

# Local application
from .components.cnn_blocks import PeriodicConv2D, ResidualBlock
from .utils import register

# Third party
import torch
from torch import nn
from torchvision.models.segmentation import deeplabv3_mobilenet_v3_large, DeepLabV3_MobileNet_V3_Large_Weights
from torchvision.models.segmentation import fcn_resnet50, FCN_ResNet50_Weights


@register("resnet_pretrained")
class ResNetPretrained(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        img_size,
        history=1,
        activation="leaky",
        norm: bool = True,
        use_pretrained_weights: bool = False,
        pretrained_model_name: str = "fcn_resnet50",
        padding_mode: str = "zeros",
    ) -> None:
        super().__init__()
        self.in_channels = in_channels * history
        self.out_channels = out_channels
        self.use_pretrained_weights = use_pretrained_weights
        self.img_size = img_size
        self.pretrained_model_name = pretrained_model_name
        self.feature_map_scaling = 2
        self.padding_mode = padding_mode

        if activation == "gelu":
            self.activation = nn.GELU()
        elif activation == "relu":
            self.activation = nn.ReLU()
        elif activation == "silu":
            self.activation = nn.SiLU()
        elif activation == "leaky":
            self.activation = nn.LeakyReLU(0.3)
        else:
            raise NotImplementedError(f"Activation {activation} not implemented")

        self.image_proj = nn.Sequential(
            PeriodicConv2D(
                self.in_channels, 128, kernel_size=7, padding=3
            ),
            nn.BatchNorm2d(128),
            self.activation,
            PeriodicConv2D(
                128, 64, kernel_size=7, padding=3
            ),
        )

        self.backbone = self.load_backbone(use_pretrained_weights)

        if 'deeplabv3' in pretrained_model_name:
            backbone_out_channels = self.backbone[str(len(self.backbone)-1)].out_channels
        elif 'fcn' in pretrained_model_name:
            backbone_out_channels = self.backbone['layer3'][0].conv3.out_channels
        self.head = nn.ModuleList()
        self.head.append(nn.Linear(backbone_out_channels, backbone_out_channels))
        self.head.append(nn.LayerNorm(backbone_out_channels))
        self.head.append(self.activation)
        self.head = nn.Sequential(*self.head)

        self.final = PeriodicConv2D(
            backbone_out_channels // (self.feature_map_scaling ** 2), out_channels, kernel_size=7, padding=3
        )


    def load_backbone(self, pretrained_weights=True):
        if self.pretrained_model_name == "deeplabv3_mobilenet_v3_large":
            if pretrained_weights:
                weights = DeepLabV3_MobileNet_V3_Large_Weights.DEFAULT
                model = deeplabv3_mobilenet_v3_large(weights=weights)
                print('Loaded Pretrained Weights')
            else:
                model = deeplabv3_mobilenet_v3_large()
                print('Randomly Initialized Model')
            model.backbone['0'] = nn.Identity()
        elif self.pretrained_model_name == "fcn_resnet50":
            if pretrained_weights:
                weights = FCN_ResNet50_Weights.DEFAULT
                model = fcn_resnet50(weights=weights)
                print('Loaded Pretrained Weights')
            else:
                model = fcn_resnet50()
                print('Randomly Initialized Model')
            model.backbone['conv1'] = nn.Identity()
            model.backbone['bn1'] = nn.Identity()
            model.backbone['relu'] = nn.Identity()
            model.backbone['maxpool'] = nn.Identity()
            model.backbone['layer3'][1] = nn.Identity()
            model.backbone['layer3'][2] = nn.Identity()
            model.backbone['layer3'][3] = nn.Identity()
            model.backbone['layer3'][4] = nn.Identity()
            model.backbone['layer3'][5] = nn.Identity()
            model.backbone['layer4'] = nn.Identity()
            for name, module in model.named_modules():
                if hasattr(module, 'padding_mode'):
                    module.padding_mode = self.padding_mode
        else:
            raise NotImplementedError(f"Pretrained model {self.pretrained_model_name} not implemented")

        return model.backbone

    def forward(self, x):
        if len(x.shape) == 5:  # x.shape = [B,T,C,H,W]
            x = x.flatten(1, 2)
        # x.shape = [B,T*C,H,W]
        x = self.image_proj(x)
        # x.shape = [B,backbone_in_channels,H,W]
        x = self.backbone(x)['out']
        # x.shape = [B,backbone_out_channels,H//8,W//8]
        x = torch.einsum('nchw->nhwc', x)
        # x.shape = [B,H//8,W//8,backbone_out_channels]
        x = self.head(x)
        # x.shape = [B,H//8,W//8,backbone_out_channels]
        x = x.reshape(shape=(x.shape[0], x.shape[1], x.shape[2], self.feature_map_scaling, self.feature_map_scaling, -1))
        # x.shape = [B,H//8,W//8,8,8,backbone_out_channels//(8*8)]
        x = torch.einsum('nhwpqv->nvhpwq', x)
        # x.shape = [B,backbone_out_channels//(8*8),H//8,8,W//8,8]
        x = x.reshape(x.shape[0], x.shape[1], self.img_size[0], self.img_size[1])
        # x.shape = [B,backbone_out_channels//(8*8),H,W]
        x = self.final(x)
        # x.shape = [B,out_channels,H,W]
        return x

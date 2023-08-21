# Local application
from xml.etree.ElementTree import tostring
from .components.cnn_blocks import PeriodicConv2D, ResidualBlock
from .utils import register

# Third party
from torch import nn
from torch.distributions.normal import Normal
import torch


@register("resnet_gaussian")
class ResNetGaussian(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        history=1,
        hidden_channels=128,
        activation="leaky",
        norm: bool = True,
        dropout: float = 0.1,
        n_blocks: int = 2,
    ) -> None:
        super().__init__()
        self.in_channels = in_channels * history
        self.out_channels = out_channels
        self.hidden_channels = hidden_channels

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

        self.image_proj = PeriodicConv2D(
            self.in_channels, hidden_channels, kernel_size=7, padding=3
        )
        self.blocks = nn.ModuleList(
            [
                ResidualBlock(
                    hidden_channels,
                    hidden_channels,
                    activation=activation,
                    norm=True,
                    dropout=dropout,
                )
                for _ in range(n_blocks)
            ]
        )

        if norm:
            self.norm = nn.BatchNorm2d(hidden_channels)
        else:
            self.norm = nn.Identity()
        self.final = PeriodicConv2D(
            hidden_channels, out_channels*2, kernel_size=7, padding=3
        )

    def forward(self, x):
        if len(x.shape) == 5:  # x.shape = [B,T,C,H,W]
            x = x.flatten(1, 2)
        # x.shape = [B,T*C,H,W]
        x = self.image_proj(x)
        for block in self.blocks:
            x = block(x)
        pred = self.final(self.activation(self.norm(x)))
        mean, std = torch.chunk(pred, 2, dim=1)
        std = torch.exp(std)
        return Normal(mean, std)

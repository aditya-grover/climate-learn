# Local application
from .components.cnn_blocks import PeriodicConv2D, ResidualBlock
from .utils import register

# Third party
from torch import nn


@register("resnet")
class ResNet(nn.Module):
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
            act = nn.GELU()
        elif activation == "relu":
            act = nn.ReLU()
        elif activation == "silu":
            act = nn.SiLU()
        elif activation == "leaky":
            act = nn.LeakyReLU(0.3)
        else:
            raise NotImplementedError(f"Activation {activation} not implemented")

        if norm:
            norm = nn.BatchNorm2d(hidden_channels)
        else:
            norm = nn.Identity()

        self.image_proj = nn.Sequential(
            PeriodicConv2D(
                self.in_channels, hidden_channels, kernel_size=7, padding=3
            ),
            act,
            norm,
            nn.Dropout(dropout)
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

        self.final = PeriodicConv2D(
            hidden_channels, out_channels, kernel_size=3, padding=1
        )

    def forward(self, x):
        if len(x.shape) == 5:  # x.shape = [B,T,C,H,W]
            x = x.flatten(1, 2)
        # x.shape = [B,T*C,H,W]
        x = self.image_proj(x)
        for block in self.blocks:
            x = block(x)
        yhat = self.final(x)
        # yhat.shape = [B,C,H,W]
        return yhat

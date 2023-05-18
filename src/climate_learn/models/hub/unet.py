# Standard library
from typing import Iterable

# Local application
from .utils import register
from .components.cnn_blocks import (
    DownBlock,
    Downsample,
    MiddleBlock,
    PeriodicConv2D,
    UpBlock,
    Upsample,
)

# Third party
import torch
from torch import nn


@register("unet")
class Unet(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        history=1,
        hidden_channels=64,
        activation="leaky",
        norm: bool = True,
        dropout: float = 0.1,
        ch_mults: Iterable[int] = (1, 2, 2, 4),
        is_attn: Iterable[bool] = (False, False, False, False),
        mid_attn: bool = False,
        n_blocks: int = 2,
    ) -> None:
        super().__init__()
        self.prob_type = None
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
            self.in_channels, self.hidden_channels, kernel_size=7, padding=3
        )

        # #### First half of U-Net - decreasing resolution
        down = []
        # Number of channels
        out_channels = in_channels = self.hidden_channels
        # For each resolution
        n_resolutions = len(ch_mults)
        for i in range(n_resolutions):
            # Number of output channels at this resolution
            out_channels = in_channels * ch_mults[i]
            # Add `n_blocks`
            for _ in range(n_blocks):
                down.append(
                    DownBlock(
                        in_channels,
                        out_channels,
                        has_attn=is_attn[i],
                        activation=activation,
                        norm=norm,
                        dropout=dropout,
                    )
                )
                in_channels = out_channels
            # Down sample at all resolutions except the last
            if i < n_resolutions - 1:
                down.append(Downsample(in_channels))

        # Combine the set of modules
        self.down = nn.ModuleList(down)

        # Middle block
        self.middle = MiddleBlock(
            out_channels,
            has_attn=mid_attn,
            activation=activation,
            norm=norm,
            dropout=dropout,
        )

        # #### Second half of U-Net - increasing resolution
        up = []
        # Number of channels
        in_channels = out_channels
        # For each resolution
        for i in reversed(range(n_resolutions)):
            # `n_blocks` at the same resolution
            out_channels = in_channels
            for _ in range(n_blocks):
                up.append(
                    UpBlock(
                        in_channels,
                        out_channels,
                        has_attn=is_attn[i],
                        activation=activation,
                        norm=norm,
                        dropout=dropout,
                    )
                )
            # Final block to reduce the number of channels
            out_channels = in_channels // ch_mults[i]
            up.append(
                UpBlock(
                    in_channels,
                    out_channels,
                    has_attn=is_attn[i],
                    activation=activation,
                    norm=norm,
                    dropout=dropout,
                )
            )
            in_channels = out_channels
            # Up sample at all resolutions except last
            if i > 0:
                up.append(Upsample(in_channels))

        # Combine the set of modules
        self.up = nn.ModuleList(up)

        if norm:
            self.norm = nn.BatchNorm2d(self.hidden_channels)
        else:
            self.norm = nn.Identity()
        self.final = PeriodicConv2D(
            in_channels, self.out_channels, kernel_size=7, padding=3
        )

    def forward(self, x):
        if len(x.shape) == 5:  # x.shape = [B,T,C,H,W]
            x = x.flatten(1, 2)
        # x.shape = [B,T*C,H,W]
        x = self.image_proj(x)
        h = [x]
        for m in self.down:
            x = m(x)
            h.append(x)
        x = self.middle(x)
        for m in self.up:
            if isinstance(m, Upsample):
                x = m(x)
            else:
                # Get the skip connection from first half of U-Net and concatenate
                s = h.pop()
                x = torch.cat((x, s), dim=1)
                x = m(x)
        yhat = self.final(self.activation(self.norm(x)))
        return yhat

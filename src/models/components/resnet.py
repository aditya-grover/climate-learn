from typing import List, Tuple, Union

import torch
from torch import nn

# Large based on https://github.com/labmlai/annotated_deep_learning_paper_implementations/blob/master/labml_nn/diffusion/ddpm/unet.py
# MIT License


class PeriodicPadding2D(nn.Module):
    def __init__(self, pad_width, **kwargs):
        super().__init__(**kwargs)
        self.pad_width = pad_width

    def forward(self, inputs, **kwargs):
        if self.pad_width == 0:
            return inputs
        inputs_padded = torch.cat((
            inputs[:, :, :, -self.pad_width:], inputs, inputs[:, :, :, :self.pad_width]
        ), dim=-1)
        # Zero padding in the lat direction
        inputs_padded = nn.functional.pad(inputs_padded, (0, 0, self.pad_width, self.pad_width))
        return inputs_padded


class PeriodicConv2D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, **kwargs):
        super().__init__(**kwargs)
        self.padding = PeriodicPadding2D(padding)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=0)

    def forward(self, inputs):
        return self.conv(self.padding(inputs))


class PeriodicConvTranspose2D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, **kwargs):
        super().__init__(**kwargs)
        self.padding = PeriodicPadding2D(padding)
        self.conv = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=0)

    def forward(self, inputs):
        return self.conv(self.padding(inputs))


class ResidualBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        activation: str = "leaky",
        norm: bool = False,
        dropout: float = 0.1,
    ):
        super().__init__()
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

        self.conv1 = PeriodicConv2D(in_channels, out_channels, kernel_size=3, padding=1)
        self.conv2 = PeriodicConv2D(out_channels, out_channels, kernel_size=3, padding=1)
        # If the number of input channels is not equal to the number of output channels we have to
        # project the shortcut connection
        if in_channels != out_channels:
            self.shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=(1, 1))
        else:
            self.shortcut = nn.Identity()

        if norm:
            self.norm1 = nn.BatchNorm2d(out_channels)
            self.norm2 = nn.BatchNorm2d(out_channels)
        else:
            self.norm1 = nn.Identity()
            self.norm2 = nn.Identity()

        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor):
        # First convolution layer
        h = self.drop(self.norm1(self.activation(self.conv1(x))))
        # Second convolution layer
        h = self.drop(self.norm2(self.activation(self.conv2(h))))
        # Add the shortcut connection and return
        return h + self.shortcut(x)


class ResNet(nn.Module):
    def __init__(
        self,
        in_channels,
        hidden_channels=128,
        activation="leaky",
        time_history=1,
        time_future=1,
        out_channels=None,
        norm: bool = True,
        dropout: float = 0.1,
        n_blocks: int = 2,
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        if out_channels is None:
            out_channels = in_channels
        self.out_channels = out_channels
        self.time_history = time_history
        self.time_future = time_future
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

        insize = time_history * self.in_channels
        # Project image into feature map
        self.image_proj = PeriodicConv2D(insize, hidden_channels, kernel_size=7, padding=3)

        blocks = []
        for i in range(n_blocks):
            blocks.append(
                ResidualBlock(
                    hidden_channels,
                    hidden_channels,
                    activation=activation,
                    norm=True,
                    dropout=dropout
                )
            )

        self.blocks = nn.ModuleList(blocks)

        if norm:
            self.norm = nn.BatchNorm2d(hidden_channels)
        else:
            self.norm = nn.Identity()
        out_channels = time_future * self.out_channels
        self.final = PeriodicConv2D(hidden_channels, out_channels, kernel_size=7, padding=3)

    def predict(self, x):
        if len(x.shape) == 5:
            x = x.flatten(1, 2)
        x = self.image_proj(x)

        for m in self.blocks:
            x = m(x)

        return self.final(self.activation(self.norm(x)))

    def forward(self, x: torch.Tensor, y: torch.Tensor, out_variables, metric, lat):
        # B, C, H, W
        pred = self.predict(x)
        return [m(pred, y, out_variables, lat) for m in metric], x

    def rollout(self, x, y, variables, out_variables, steps, metric, transform, lat, log_steps, log_days):
        if steps > 1:
            assert len(variables) == len(out_variables)

        preds = []
        for _ in range(steps):
            x = self.predict(x)
            preds.append(x)
        preds = torch.stack(preds, dim=1)

        return [m(preds, y, transform, out_variables, lat, log_steps, log_days) for m in metric], preds


# model = ResNet(in_channels=2, time_history=3, out_channels=2).cuda()
# x = torch.randn((64, 3, 2, 32, 64)).cuda()
# y = model.predict(x)
# print (y.shape)

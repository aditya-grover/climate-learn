import torch
from torch import nn
from .cnn_blocks import PeriodicConv2D, ResidualBlock

# Large based on https://github.com/labmlai/annotated_deep_learning_paper_implementations/blob/master/labml_nn/diffusion/ddpm/unet.py
# MIT License


class ResNet(nn.Module):
    def __init__(
        self,
        in_channels,
        history=1,
        hidden_channels=128,
        activation="leaky",
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

        insize = self.in_channels * history
        # Project image into feature map
        self.image_proj = PeriodicConv2D(
            insize, hidden_channels, kernel_size=7, padding=3
        )

        blocks = []
        for i in range(n_blocks):
            blocks.append(
                ResidualBlock(
                    hidden_channels,
                    hidden_channels,
                    activation=activation,
                    norm=True,
                    dropout=dropout,
                )
            )

        self.blocks = nn.ModuleList(blocks)

        if norm:
            self.norm = nn.BatchNorm2d(hidden_channels)
        else:
            self.norm = nn.Identity()
        out_channels = self.out_channels
        self.final = PeriodicConv2D(
            hidden_channels, out_channels, kernel_size=7, padding=3
        )

    def predict(self, x):
        if len(x.shape) == 5:  # history
            x = x.flatten(1, 2)
        # x.shape [128, 1, 32, 64]
        x = self.image_proj(x)  # [128, 128, 32, 64]

        for m in self.blocks:
            x = m(x)

        pred = self.final(self.activation(self.norm(x)))  # pred.shape [128, 50, 32, 64]

        return pred

    def forward(self, x: torch.Tensor, y: torch.Tensor, out_variables, metric, lat):
        # B, C, H, W
        pred = self.predict(x)
        return ([m(pred, y, out_variables, lat=lat) for m in metric], x)

    def rollout(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        clim,
        variables,
        out_variables,
        steps,
        metric,
        transform,
        lat,
        log_steps,
        log_days,
    ):
        if steps > 1:
            assert len(variables) == len(out_variables)

        preds = []
        for _ in range(steps):
            x = self.predict(x)
            preds.append(x)
        preds = torch.stack(preds, dim=1)
        if len(y.shape) == 4:
            y = y.unsqueeze(1)

        return (
            [
                m(
                    preds,
                    y,
                    out_variables,
                    transform=transform,
                    lat=lat,
                    log_steps=log_steps,
                    log_days=log_days,
                    clim=clim,
                )
                for m in metric
            ],
            x,
        )

    def upsample(self, x, y, out_vars, transform, metric):
        with torch.no_grad():
            pred = self.predict(x)
        return ([m(pred, y, out_vars, transform=transform) for m in metric], x)

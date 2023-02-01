from math import log
from typing import List, Tuple, Union

import torch
from torch import nn
from torch.distributions.normal import Normal
from .cnn_blocks import (
    PeriodicConv2D,
    DownBlock,
    UpBlock,
    MiddleBlock,
    Downsample,
    Upsample,
)

# Large based on https://github.com/labmlai/annotated_deep_learning_paper_implementations/blob/master/labml_nn/diffusion/ddpm/unet.py
# MIT License


class Unet(nn.Module):
    def __init__(
        self,
        in_channels,
        history=1,
        hidden_channels=64,
        activation="leaky",
        out_channels=None,
        upsampling=1,
        norm: bool = True,
        dropout: float = 0.1,
        ch_mults: Union[Tuple[int, ...], List[int]] = (1, 2, 2, 4),
        is_attn: Union[Tuple[bool, ...], List[bool]] = (False, False, False, False),
        mid_attn: bool = False,
        n_blocks: int = 2,
        prob_type: str = None,  # parametric, mcdropout, deter (used for ensembling)
        n_samples: int = 50,  # only used for mcdropout
    ) -> None:
        super().__init__()
        self.prob_type = None
        self.in_channels = in_channels
        if out_channels is None:
            out_channels = in_channels
        self.out_channels = out_channels
        self.hidden_channels = hidden_channels
        self.upsampling = upsampling

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

        assert not prob_type or prob_type in ["parametric", "mcdropout", "deter"]
        self.prob_type = prob_type
        self.n_samples = n_samples

        # Number of resolutions
        n_resolutions = len(ch_mults)

        insize = self.in_channels * history
        n_channels = hidden_channels
        # Project image into feature map
        self.image_proj = PeriodicConv2D(insize, n_channels, kernel_size=7, padding=3)

        # #### First half of U-Net - decreasing resolution
        down = []
        # Number of channels
        out_channels = in_channels = n_channels
        # For each resolution
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
                        mc_dropout=(self.prob_type == "mcdropout"),
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
            mc_dropout=(self.prob_type == "mcdropout"),
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
                        mc_dropout=(self.prob_type == "mcdropout"),
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
                    mc_dropout=(self.prob_type == "mcdropout"),
                )
            )
            in_channels = out_channels
            # Up sample at all resolutions except last
            if i > 0:
                up.append(Upsample(in_channels))

        # Combine the set of modules
        self.up = nn.ModuleList(up)

        upsamplers = []
        if upsampling > 1:
            n_upsamplers = int(log(upsampling, 2))
            for i in range(n_upsamplers - 1):
                upsamplers.append(Upsample(hidden_channels))
                upsamplers.append(self.activation)
            upsamplers.append(Upsample(hidden_channels))
        self.upsamplers = nn.ModuleList(upsamplers)

        if norm:
            self.norm = nn.BatchNorm2d(n_channels)
        else:
            self.norm = nn.Identity()
        out_channels = self.out_channels
        self.final = PeriodicConv2D(in_channels, out_channels, kernel_size=7, padding=3)
        if prob_type == "parametric":
            self.final_std = PeriodicConv2D(
                in_channels, out_channels, kernel_size=7, padding=3
            )

    def predict(self, x):
        if len(x.shape) == 5:  # history
            x = x.flatten(1, 2)
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

        if self.upsampling > 1:
            for m in self.upsamplers:
                x = m(x)

        pred = self.final(self.activation(self.norm(x)))

        if self.prob_type == "parametric":
            std = self.final_std(self.activation(self.norm(x)))
            std = torch.exp(std)
            pred = Normal(pred, std)

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
        mean_transform,
        std_transform,
        log_day,
    ):
        """
        Notes from climate_uncertainty repo merge
        Shared function params before merge:
            x, y, clim, variables, out_variables, metric
        Unique function params for climate_tutorial before merge:
            steps, transform, lat, log_steps, log_days
        Unique function params for climate_uncertainty before merge:
            mean_transform, std_transform, lat, log_day
        """
        if self.prob_type:
            # x: B, C, H, W
            b = x.shape[0]

            if self.prob_type == "mcdropout":
                x = (
                    x.unsqueeze(1).repeat(1, self.n_samples, 1, 1, 1, 1).flatten(0, 1)
                )  # B x n_samples, C, H, W

            pred = self.predict(x)  # Normal if parametric else Tensor

            if self.prob_type == "mcdropout":
                pred = mean_transform(pred)
                pred = pred.unflatten(
                    dim=0, sizes=(b, self.n_samples)
                )  # B, n_samples, C, H, W
                mean = torch.mean(pred, dim=1)
                std = torch.std(pred, dim=1)
                pred = Normal(mean, std)
            elif self.prob_type == "parametric":
                mean, std = pred.loc, pred.scale
                mean = mean_transform(mean)
                std = std_transform(std)
                pred = Normal(mean, std)
            else:
                pred = mean_transform(pred)

            y = mean_transform(y)

            return (
                [
                    m(
                        pred,
                        y,
                        out_variables,
                        transform=transform,
                        lat=lat,
                        log_steps=log_steps,
                        log_days=log_days,
                        log_day=log_day,
                        clim=clim,
                    )
                    for m in metric
                ],
                x,
            )
        else:
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

    def val_rollout(self, *args, **kwargs):
        return self.rollout(*args, **kwargs)

    def test_rollout(self, *args, **kwargs):
        return self.rollout(*args, **kwargs)

    def upsample(self, x, y, out_vars, transform, metric):
        with torch.no_grad():
            pred = self.predict(x)
        return ([m(pred, y, out_vars, transform=transform) for m in metric], x)

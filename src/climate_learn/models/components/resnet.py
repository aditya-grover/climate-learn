import torch
from torch import nn
from torch.distributions.normal import Normal
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
        prob_type: str = None,  # parametric, mcdropout, categorical
        n_samples: int = 50,  # only used for mcdropout
        num_output_bins: int = 50,  # only used for categorical
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

        assert not prob_type or prob_type in ["parametric", "mcdropout", "categorical"]
        self.prob_type = prob_type
        self.n_samples = n_samples

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
                    mc_dropout=(self.prob_type == "mcdropout"),
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

        if prob_type == "parametric":
            self.final_std = PeriodicConv2D(
                hidden_channels, out_channels, kernel_size=7, padding=3
            )

        # final layer for categorical, change to be number of bins
        if self.prob_type == "categorical":
            self.num_output_bins = num_output_bins
            self.final = PeriodicConv2D(
                self.hidden_channels, self.num_output_bins, kernel_size=7, padding=3
            )

    def predict(self, x):
        if len(x.shape) == 5:  # history
            x = x.flatten(1, 2)
        # x.shape [128, 1, 32, 64]
        x = self.image_proj(x)  # [128, 128, 32, 64]

        for m in self.blocks:
            x = m(x)

        pred = self.final(self.activation(self.norm(x)))  # pred.shape [128, 50, 32, 64]

        # following the implementation on https://github.com/sagar-garg/WeatherBench/blob/f41f497ac45377d363dc30bfa77daf50d7b28afd/src/networks.py#L208
        if self.prob_type == "categorical":
            bins = int(x.shape[-1] / self.n_vars)  # bins = 64, self.n_vars = 1
            outputs = []
            for i in range(self.n_vars):
                o = nn.Softmax(dim=1)(pred[..., i * bins : (i + 1) * bins])
                # o.shape [128, 50, 32, 64]
                outputs.append(o)
            pred = torch.stack(outputs, dim=2)  # [128, 50, 1, 32, 64]

        if self.prob_type == "parametric":
            std = self.final_std(self.activation(self.norm(x)))
            std = torch.exp(std)
            pred = Normal(pred, std)

        return pred

    def forward(self, x: torch.Tensor, y: torch.Tensor, out_variables, metric, lat):
        # B, C, H, W
        if self.prob_type == "categorical":
            self.n_vars = len(out_variables)

        pred = self.predict(x)
        return ([m(pred, y, out_variables, lat=lat) for m in metric], x)

    def val_rollout(
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

            if self.prob_type == "categorical":
                self.n_vars = len(out_variables)

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

            # no normalization on y for categorical
            if self.prob_type is not "categorical":
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

    def test_rollout(
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

            if self.prob_type == "categorical":
                self.n_vars = len(out_variables)

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

            # no normalization on y for categorical
            if self.prob_type is not "categorical":
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

    def upsample(self, x, y, out_vars, transform, metric):
        with torch.no_grad():
            pred = self.predict(x)
        return ([m(pred, y, out_vars, transform=transform) for m in metric], x)

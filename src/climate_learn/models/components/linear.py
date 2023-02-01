# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# timm: https://github.com/rwightman/pytorch-image-models/tree/master/timm
# DeiT: https://github.com/facebookresearch/deit
# --------------------------------------------------------
import torch
import torch.nn as nn


class Linear(nn.Module):
    """Base model for tokenized MAE and tokenized ViT Including patch embedding and encoder."""

    def __init__(
        self,
        img_size=[128, 256],
        in_channels=2,
        out_channels=2,
    ):
        super().__init__()

        self.h, self.w = img_size
        self.out_channels = out_channels
        in_dim = in_channels * self.h * self.w
        out_dim = out_channels * self.h * self.w

        self.W = nn.Linear(in_dim, out_dim)

    def predict(self, x):
        pred = self.W(x.flatten(1))
        return pred.unflatten(dim=1, sizes=(self.out_channels, self.h, self.w))

    def forward(self, x: torch.Tensor, y, out_variables, metric, lat):
        pred = self.predict(x)
        return [m(pred, y, out_variables, lat) for m in metric], pred

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
        # transform: get back to the original range
        if steps > 1:
            # can only rollout for more than 1 step if input variables and output variables are the same
            assert len(variables) == len(out_variables)

        preds = []
        for _ in range(steps):
            x = self.predict(x)
            preds.append(x)
        preds = torch.stack(preds, dim=1)

        preds = transform(preds)
        y = transform(y)

        return [
            m(preds, y, out_variables, lat, log_steps, log_days) for m in metric
        ], preds


# model = TokenizedMAE(depth=4, decoder_depth=2).cuda()
# x = torch.randn(2, 3, 128, 256).cuda()
# loss, pred, mask = model(x)
# print (loss)

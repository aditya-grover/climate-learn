# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# timm: https://github.com/rwightman/pytorch-image-models/tree/master/timm
# DeiT: https://github.com/facebookresearch/deit
# --------------------------------------------------------


import numpy as np
import torch
import torch.nn as nn
from src.utils.pos_embed import (get_1d_sincos_pos_embed_from_grid,
                                 get_2d_sincos_pos_embed)
from timm.models.vision_transformer import Block, PatchEmbed, trunc_normal_


class VisionTransformer(nn.Module):
    def __init__(
        self,
        time_history=1,
        img_size=[128, 256],
        patch_size=16,
        drop_path=0.1,
        drop_rate=0.1,
        learn_pos_emb=False,
        in_vars=[
            "2m_temperature",
            "10m_u_component_of_wind",
            "10m_v_component_of_wind",
        ],
        embed_dim=1024,
        depth=24,
        decoder_depth=8,
        num_heads=16,
        mlp_ratio=4.0,
        out_vars=None,
        init_mode="xavier",
    ):
        super().__init__()

        self.time_history = time_history
        self.img_size = img_size
        self.n_channels = len(in_vars)
        self.patch_size = patch_size
        self.init_mode = init_mode

        out_vars = out_vars if out_vars is not None else in_vars
        self.in_vars = in_vars
        self.out_vars = out_vars

        # --------------------------------------------------------------------------
        # ViT encoder specifics - exactly the same to MAE
        self.patch_embed = PatchEmbed(img_size, patch_size, len(self.in_vars), embed_dim)
        self.num_patches = self.patch_embed.num_patches  # 128

        self.pos_embed = nn.Parameter(
            torch.zeros(1, self.num_patches, embed_dim), requires_grad=learn_pos_emb
        )  # fixed sin-cos embedding
        self.time_pos_embed = nn.Parameter(torch.zeros(1, time_history, embed_dim), requires_grad=learn_pos_emb)
        self.pos_drop = nn.Dropout(p=drop_rate)

        dpr = [x.item() for x in torch.linspace(0, drop_path, depth)]  # stochastic depth decay rule
        self.blocks = nn.ModuleList(
            [
                Block(
                    embed_dim,
                    num_heads,
                    mlp_ratio,
                    qkv_bias=True,
                    drop_path=dpr[i],
                    norm_layer=nn.LayerNorm,
                    drop=drop_rate,
                )
                for i in range(depth)
            ]
        )
        self.norm = nn.LayerNorm(embed_dim)
        # --------------------------------------------------------------------------

        # --------------------------------------------------------------------------
        # ViT encoder specifics - exactly the same to MAE
        self.head = nn.ModuleList()
        for i in range(decoder_depth):
            self.head.append(nn.Linear(embed_dim, embed_dim))
            self.head.append(nn.GELU())
        self.head.append(nn.Linear(embed_dim, len(self.out_vars) * patch_size**2))
        self.head = nn.Sequential(*self.head)
        # --------------------------------------------------------------------------

        self.initialize_weights()

    def initialize_weights(self):
        # initialization
        # initialize (and freeze) pos_embed by sin-cos embedding
        pos_embed = get_2d_sincos_pos_embed(
            self.pos_embed.shape[-1],
            int(self.img_size[0] / self.patch_size),
            int(self.img_size[1] / self.patch_size),
            cls_token=False,
        )
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))
        time_pos_embed = get_1d_sincos_pos_embed_from_grid(self.time_pos_embed.shape[-1], np.arange(self.time_history))
        self.time_pos_embed.data.copy_(torch.from_numpy(time_pos_embed).float().unsqueeze(0))

        # initialize patch_embed like nn.Linear (instead of nn.Conv2d)
        w = self.patch_embed.proj.weight.data
        if self.init_mode == "xavier":
            torch.nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
        else:
            trunc_normal_(w.view([w.shape[0], -1]), std=0.02)

        # initialize nn.Linear and nn.LayerNorm
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            if self.init_mode == "xavier":
                torch.nn.init.xavier_uniform_(m.weight)
            else:
                trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def patchify(self, imgs):
        """
        imgs: (N, C, H, W)
        x: (N, L, patch_size**2 *3)
        """
        p = self.patch_size
        assert imgs.shape[2] % p == 0 and imgs.shape[3] % p == 0

        h = self.img_size[0] // p
        w = self.img_size[1] // p
        c = len(self.in_vars)
        x = imgs.reshape(shape=(imgs.shape[0], c, h, p, w, p))
        x = torch.einsum("nchpwq->nhwpqc", x)
        x = x.reshape(shape=(imgs.shape[0], h * w, p**2 * c))
        return x

    def unpatchify(self, x):
        """
        x: (N, L, patch_size**2 *3)
        imgs: (N, 3, H, W)
        """
        p = self.patch_size
        c = len(self.out_vars)
        h = self.img_size[0] // p
        w = self.img_size[1] // p
        assert h * w == x.shape[1]

        x = x.reshape(shape=(x.shape[0], h, w, p, p, c))
        x = torch.einsum("nhwpqc->nchpwq", x)
        imgs = x.reshape(shape=(x.shape[0], c, h * p, w * p))
        return imgs

    def forward_encoder(self, x: torch.Tensor):
        """
        x: B, T, C, H, W
        """
        b, t, _, _, _ = x.shape
        x = x.flatten(0, 1)  # BxT, C, H, W
        # embed patches
        x = self.patch_embed(x)
        x = x.unflatten(dim=0, sizes=(b, t))  # B, T, L, D

        # add time and pos embed
        # pos emb: 1, L, D
        x = x + self.pos_embed.unsqueeze(1)
        # time emb: 1, T, D
        x = x + self.time_pos_embed.unsqueeze(2)

        x = x.flatten(1, 2)  # B, TxL, D

        x = self.pos_drop(x)

        # apply Transformer blocks
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)

        return x

    def forward_loss(self, y, pred, variables, out_variables, metric, lat):  # metric is a list
        """
        y: [N, 3, H, W]
        pred: [N, L, p*p*3]
        """
        pred = self.unpatchify(pred)
        return [m(pred, y, out_variables, lat) for m in metric], pred

    def forward(self, x, y, variables, out_variables, metric, lat):
        embeddings = self.forward_encoder(x)  # B, TxL, D
        embeddings = embeddings
        preds = self.head(embeddings)[:, -self.num_patches :]
        loss, preds = self.forward_loss(y, preds, variables, out_variables, metric, lat)
        return loss, preds

    def predict(self, x, variables):
        with torch.no_grad():
            embeddings = self.forward_encoder(x)
            embeddings = embeddings
            pred = self.head(embeddings)[:, -self.num_patches :]
        return self.unpatchify(pred)

    def rollout(self, x, y, variables, out_variables, steps, metric, transform, lat, log_steps, log_days):
        preds = []
        for _ in range(steps):
            x = self.predict(x, variables)
            preds.append(x)
        preds = torch.stack(preds, dim=1)

        return [m(preds, y, transform, out_variables, lat, log_steps, log_days) for m in metric], preds


# model = VisionTransformer(depth=8).cuda()
# x, y = torch.randn(2, 3, 128, 256).cuda(), torch.randn(2, 3, 128, 256).cuda()
# loss, preds = model.forward(x, y)

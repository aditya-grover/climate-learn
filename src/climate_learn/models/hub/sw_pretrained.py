#Local application
from .components.cnn_blocks import PeriodicConv2D
from .components.pos_embed import get_2d_sincos_pos_embed
from .utils import register

#Third Party
import torch
import torch.nn as nn
import torchvision
import sys
import ipdb
import timm
from transformers import ViTModel, AutoConfig, AutoModel, CLIPModel
from timm.models.vision_transformer import Block, PatchEmbed, trunc_normal_
from transformers import Swinv2Config, Swinv2Model


@register('vit_pretrained')
class SwinV2Pretrained(nn.Module):

    def __init__(self,
        in_channels, 
        out_channels,
        history,
        pretrained_model=None,
        use_pretrained_weights=False,
        # use_pretrained_embeddings=False,
        # use_n_blocks=None,
        # freeze_backbone=False, 
        # freeze_embeddings=False,
        mlp_embed_depth=0,
        decoder_depth=2,
    ):
        super().__init__()

        if use_pretrained_weights:
            self.backbone = Swinv2Model.from_pretrained(pretrained_model)
            self.config = self.backbone.config
            print ('Using a pretrained backbone')
        else:
            self.config = Swinv2Config.from_pretrained(pretrained_model)
            self.backbone = Swinv2Model(self.config)
            print ('Initializing a random backbone')

        self.in_channels = in_channels * history
        self.out_channels = out_channels
        self.use_pretrained_weights = use_pretrained_weights
        self.pretrained_model_name = pretrained_model
        
        # Embedding layers
        self.embedding = nn.ModuleList()
        self.embedding.append(PatchEmbed(
            self.config.image_size,
            self.config.patch_size,
            self.in_channels,
            self.config.embed_dim,
            flatten=False
        ))
        for _ in range(mlp_embed_depth):
            self.embedding.append(nn.GELU())
            self.embedding.append(nn.Conv2d(
                in_channels=self.config.embed_dim, out_channels=self.config.embed_dim, kernel_size=1
            ))
        self.embedding = nn.Sequential(*self.embedding)

        
        # prediction head
        self.head = nn.ModuleList()
        for _ in range(decoder_depth):
            self.head.append(nn.Conv2d(
                in_channels=self.config.hidden_size, out_channels=self.config.hidden_size, kernel_size=1
            ))
            self.head.append(nn.GELU())
        self.head.append(nn.Conv2d(
            in_channels=self.config.hidden_size, out_channels=self.config.encoder_stride**2 * self.out_channels, kernel_size=1
        ))
        self.head.append(nn.PixelShuffle(self.config.encoder_stride))
        self.head = nn.Sequential(*self.head)

        if not self.use_pretrained_weights:
            self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward_encoder(self, x):
        # x.shape = [B,T*in_channels,H,W]
        x = torch.nn.functional.interpolate(x, size=(self.config.image_size, self.config.image_size))

        x = self.embedding(x)
        _, _, h, w = x.shape
        input_dimensions = (h, w)
        x = x.flatten(2).transpose(1, 2)
        x = self.backbone.encoder(
            x,
            input_dimensions,
            output_attentions=None,
            output_hidden_states=True,
            return_dict=True,
        )
        x = x.reshaped_hidden_states[-1]
        return x

    def forward(self, x):
        if len(x.shape) == 5:
            x = x.flatten(1, 2)
        x = self.forward_encoder(x)
        x = self.head(x)
        return x

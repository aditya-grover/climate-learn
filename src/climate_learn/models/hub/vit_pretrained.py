#Third Party
import torch
import torch.nn as nn
import sys
from transformers import ViTModel, AutoConfig, AutoModel
from timm.models.vision_transformer import Block, PatchEmbed, trunc_normal_


class ViTPretrained(nn.Module):

    def __init__(self, 
        img_size, 
        in_channels, 
        out_channels, 
        use_pretrained_backbone=False,
        use_pretrained_embeddings=False,
        freeze_backbone=False, 
        freeze_embeddings=False,
        patch_size=16, 
        embed_dim=1024, 
        decoder_depth=2
    ):
        super().__init__()

        if not use_pretrained_embeddings:
            self.patch_size = 4
        self.img_size = img_size
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_patches = (img_size[0] * img_size[1]) // (patch_size)**2
        self.embed_dim = embed_dim
        self.use_pretrained_embeddings = use_pretrained_embeddings

        self.in_conv = nn.Conv2d(in_channels=in_channels, out_channels=3, kernel_size=1)
        if use_pretrained_backbone:
            self.forward_encoder = AutoModel.from_pretrained('google/vit-large-patch16-224-in21k')
            if freeze_backbone:
                for name, param in self.forward_encoder.named_parameters():
                    if 'embeddings' in name and not freeze_embeddings:
                        continue
                    param.requires_grad = False
        else:
            ViTModelConfig = AutoConfig.from_pretrained('google/vit-large-patch16-224-in21k')
            self.forward_encoder = AutoModel.from_config(ViTModelConfig)
        
        if not use_pretrained_embeddings:
            self.forward_encoder.embeddings = nn.Identity()

            self.patch_embed = PatchEmbed(img_size, patch_size, in_channels, embed_dim)
            self.pos_embed = nn.Parameter(
                torch.zeros(1, self.num_patches, embed_dim), requires_grad=True
            )
            self.pos_drop = nn.Dropout(p=0.1)

            print('Using new embeddings')

        
        # prediction head
        self.head = nn.ModuleList()
        for _ in range(decoder_depth):
            self.head.append(nn.Linear(embed_dim, embed_dim))
            self.head.append(nn.GELU())
        self.head.append(nn.Linear(embed_dim, out_channels*patch_size**2))
        self.head = nn.Sequential(*self.head)

        # self.final = PeriodicConv2D(
            # (self.num_patches * 1024) // (img_size[-2] * img_size[-1]),
            # self.out_channels,
            # kernel_size=7,
            # padding=3,
        # )

    def unpatchify(self, x: torch.Tensor, h=None, w=None):
        """
        x: (B, L, V * patch_size**2)
        return imgs: (B, V, H, W)
        """
        p = 16
        v = self.in_channels
        h = self.img_size[0] // p if h is None else h // p
        w = self.img_size[1] // p if w is None else w // p
        assert h * w == x.shape[1]

        x = x.reshape(shape=(x.shape[0], h, w, p, p, v))
        x = torch.einsum("nhwpqv->nvhpwq", x)
        imgs = x.reshape(shape=(x.shape[0], v, h * p, w * p))
        return imgs

    def forward(self, x):
        # x.shape = [B,T,in_channels,H,W]
        x = x.flatten(1, 2)
        # x.shape = [B,T*in_channels,H,W]
        if self.use_pretrained_embeddings:
            x = self.in_conv(x)
            # x.shape = [B, 3, H, W]
        else:
            x = self.patch_embed(x)
            x = x + self.pos_embed
            x = self.pos_drop(x)
            # x.shape = [B,num_patches,embed_dim]
            
        x = self.forward_encoder(x, interpolate_pos_encoding=True)
        x = x.last_hidden_state

        if self.use_pretrained_embeddings:
            # x.shape = [B,num_patches+1,embed_dim]        
            x = x[:, 1:]
        # x.shape = [B, num_patches, embed_dim]
        x = self.head(x)
        # x.shape = [B, num_patches, out_channels*patch_size**2]
        x = self.unpatchify(x)
        # x.shape = [B, out_channels, H, W]
        return x


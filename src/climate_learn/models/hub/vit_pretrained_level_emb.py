#Local application
from .components.cnn_blocks import PeriodicConv2D
from .components.pos_embed import get_2d_sincos_pos_embed
from .utils import register

#Third Party
import numpy as np
import torch
import torch.nn as nn
import torchvision
import sys
import ipdb
import timm
from transformers import ViTModel, AutoConfig, AutoModel, CLIPModel
from timm.models.vision_transformer import Block, PatchEmbed, trunc_normal_
from climate_learn.data.climate_dataset.era5.constants import PRESSURE_LEVEL_VARS, DEFAULT_PRESSURE_LEVELS, SINGLE_LEVEL_VARS, CONSTANTS
from climate_learn.models.hub.climax import get_1d_sincos_pos_embed_from_grid, lru_cache


class LevelEmbedding(nn.Module):
    def __init__(
        self,
        default_vars,
        var_map,
        img_size=[32, 64],
        patch_size=2,
        embed_dim=1024,
        num_heads=16,
        drop_rate=0.1,
    ):
        super().__init__()

        # TODO: remove time_history parameter
        self.img_size = img_size
        self.patch_size = patch_size
        self.default_vars = default_vars
        self.var_map = var_map

        # variable tokenization: separate embedding layer for each group of input variables
        self.token_embeds = nn.ModuleList([])
        for k in var_map.keys():
            self.token_embeds.append(PatchEmbed(img_size, patch_size, len(var_map[k]), embed_dim))
        self.num_patches = self.token_embeds[0].num_patches

        # variable embedding to denote which variable each token belongs to
        # helps in aggregating variables
        self.channel_embed = nn.Parameter(torch.zeros(1, len(self.token_embeds), embed_dim), requires_grad=True)

        # variable aggregation: a learnable query and a single-layer cross attention
        self.channel_query = nn.Parameter(torch.zeros(1, 1, embed_dim), requires_grad=True)
        self.channel_agg = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)

        # positional embedding
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches, embed_dim), requires_grad=True)

        self.initialize_weights()

    def initialize_weights(self):
        # initialize pos_emb and var_emb
        pos_embed = get_2d_sincos_pos_embed(
            self.pos_embed.shape[-1],
            int(self.img_size[0] / self.patch_size),
            int(self.img_size[1] / self.patch_size),
            cls_token=False,
        )
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        channel_embed = get_1d_sincos_pos_embed_from_grid(self.channel_embed.shape[-1], np.arange(len(self.token_embeds)))
        self.channel_embed.data.copy_(torch.from_numpy(channel_embed).float().unsqueeze(0))

        # token embedding layer
        for i in range(len(self.token_embeds)):
            w = self.token_embeds[i].proj.weight.data
            trunc_normal_(w.view([w.shape[0], -1]), std=0.02)

        # initialize nn.Linear and nn.LayerNorm
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    # def create_var_embedding(self, dim):
    #     var_embed = nn.Parameter(torch.zeros(1, len(self.token_embeds), dim), requires_grad=True)
    #     # TODO: create a mapping from var --> idx
    #     var_map = {}
    #     idx = 0
    #     for var in self.default_vars:
    #         var_map[var] = idx
    #         idx += 1
    #     return var_embed, var_map

    # @lru_cache(maxsize=None)
    # def get_var_ids(self, vars, device):
    #     ids = np.array([self.channel_map[var] for var in vars])
    #     return torch.from_numpy(ids).to(device)

    # def get_var_emb(self, var_emb, vars):
    #     ids = self.get_var_ids(vars, var_emb.device)
    #     return var_emb[:, ids, :]

    def aggregate_variables(self, x: torch.Tensor):
        """
        x: B, V, L, D
        """
        b, _, l, _ = x.shape
        x = torch.einsum("bvld->blvd", x)
        x = x.flatten(0, 1)  # BxL, V, D

        var_query = self.channel_query.repeat_interleave(x.shape[0], dim=0)
        x, _ = self.channel_agg(var_query, x, x)  # BxL, D
        x = x.squeeze()

        x = x.unflatten(dim=0, sizes=(b, l))  # B, L, D
        return x

    def forward(self, x: torch.Tensor, variables: list):
        embeds = [] # each group is embedded separately
        # iterating over all variables in each group
        # may not be the most efficient way
        for i, group in enumerate(self.var_map.keys()):
            input_group = []
            for var in self.var_map[group]:
                var_id = variables.index(var)
                input_group.append(x[:, var_id])
            input_group = torch.stack(input_group, dim=1)
            embeds.append(self.token_embeds[i](input_group))

        x = torch.stack(embeds, dim=1)  # B, G, L, D

        # add variable embedding
        x = x + self.channel_embed.unsqueeze(2)  # B, G, L, D

        # variable aggregation
        x = self.aggregate_variables(x)  # B, L, D

        # add pos embedding
        x = x + self.pos_embed

        return x


@register('vit_pretrained_level_emb')
class ViTPretrainedLevelEmb(nn.Module):

    def __init__(self, 
        in_img_size,
        out_img_size, 
        in_channels, 
        out_channels,
        history,
        use_pretrained_weights=False,
        use_n_blocks=None,
        freeze_backbone=False, 
        freeze_embeddings=False,
        learn_pos_emb=False,
        resize_img=False,
        patch_size=16, 
        embed_dim=1024, 
        decoder_depth=2,
        pretrained_model=None,
        mlp_embed_depth=0,
    ):
        super().__init__()
        self.patch_size = patch_size
        self.in_img_size = in_img_size
        self.out_img_size = out_img_size
        self.in_channels = in_channels * history
        self.out_channels = out_channels
        self.num_patches = (in_img_size[0] * in_img_size[1]) // (patch_size)**2
        self.embed_dim = embed_dim
        self.use_pretrained_weights = use_pretrained_weights
        self.use_n_blocks = use_n_blocks
        self.freeze_embeddings= freeze_embeddings
        self.freeze_backbone = freeze_backbone
        self.pretrained_model_name = pretrained_model
        self.resize_img = resize_img
        self.eff_patch_size = [int((patch_size / in_img_size[0]) * out_img_size[0]), int((patch_size / in_img_size[1]) * out_img_size[1])]

        self.load_pretrained_model()

        self.input_variables = [
            "land_sea_mask",
            "orography",
            "lattitude",
            "2m_temperature",
            # "10m_u_component_of_wind",
            # "10m_v_component_of_wind",
            "geopotential_50",
            "geopotential_250",
            "geopotential_500",
            "geopotential_600",
            "geopotential_700",
            "geopotential_850",
            "geopotential_925",
            "u_component_of_wind_50",
            "u_component_of_wind_250",
            "u_component_of_wind_500",
            "u_component_of_wind_600",
            "u_component_of_wind_700",
            "u_component_of_wind_850",
            "u_component_of_wind_925",
            "v_component_of_wind_50",
            "v_component_of_wind_250",
            "v_component_of_wind_500",
            "v_component_of_wind_600",
            "v_component_of_wind_700",
            "v_component_of_wind_850",
            "v_component_of_wind_925",
            "temperature_50",
            "temperature_250",
            "temperature_500",
            "temperature_600",
            "temperature_700",
            "temperature_850",
            "temperature_925",
            # "relative_humidity_50",
            # "relative_humidity_250",
            # "relative_humidity_500",
            # "relative_humidity_600",
            # "relative_humidity_700",
            # "relative_humidity_850",
            # "relative_humidity_925",
            "specific_humidity_50",
            "specific_humidity_250",
            "specific_humidity_500",
            "specific_humidity_600",
            "specific_humidity_700",
            "specific_humidity_850",
            "specific_humidity_925",
        ]

        var_map = {}

        # pressure-level variables
        for level in DEFAULT_PRESSURE_LEVELS:
            k = f"pressure_{level}"
            var_map[k] = []
            for var in PRESSURE_LEVEL_VARS:
                var_name_with_pressure = f"{var}_{level}"
                if var_name_with_pressure in self.input_variables:
                    var_map[k].append(var_name_with_pressure)
            if len(var_map[k]) == 0:
                del var_map[k]
        
        # surface-level variables
        var_map["surface"] = []
        for var in SINGLE_LEVEL_VARS:
            if var not in CONSTANTS:
                if var in self.input_variables:
                    var_map["surface"].append(var)
        
        # constant variables
        var_map["constant"] = []
        for var in CONSTANTS:
            if var in self.input_variables:
                var_map["constant"].append(var)

        
        self.level_emb = LevelEmbedding(
            default_vars=self.input_variables,
            var_map=var_map,
            img_size=in_img_size,
            patch_size=patch_size,
            embed_dim=embed_dim,
            num_heads=self.pretrained_backbone.num_heads,
            drop_rate=0.1,
        )

        if self.freeze_embeddings:
            self.level_emb.requires_grad_(False)
        
        # prediction head
        self.head = nn.ModuleList()
        for _ in range(decoder_depth):
            self.head.append(nn.Linear(embed_dim, embed_dim))
            self.head.append(nn.GELU())
        self.head.append(nn.Linear(embed_dim, out_channels*self.eff_patch_size[0]*self.eff_patch_size[1]))
        self.head = nn.Sequential(*self.head)

        # self.initialize_weights()

    def load_pretrained_model(self):
        if 'google/vit' in self.pretrained_model_name:
            print('Loading google/vit')
            if self.use_pretrained_weights:
                self.pretrained_backbone = AutoModel.from_pretrained(self.pretrained_model_name)
                if self.freeze_backbone:
                    for name, param in self.pretrained_backbone.named_parameters():
                        if 'embeddings' in name and not self.freeze_embeddings:
                            continue
                        param.requires_grad = False
            else:
                print(f'Loading randomly initialized model like {self.pretrained_model_name}')
                ViTModelConfig = AutoConfig.from_pretrained(self.pretrained_model_name)
                self.pretrained_backbone = AutoModel.from_config(ViTModelConfig)
        elif 'dinov2' in self.pretrained_model_name:
            if self.use_pretrained_weights:
                print('Loading dinov2 weights')
                self.pretrained_backbone = torch.hub.load('facebookresearch/dinov2', self.pretrained_model_name)
            else:
                print('Loading randomly initialized model like DINOv2')
                self.pretrained_backbone = torch.hub.load('facebookresearch/dinov2', self.pretrained_model_name, pretrained=False)
            if self.freeze_backbone:
                print('Freezing Backbone')
                if self.freeze_embeddings:
                    print('Freezing Embeddings')
                for name, param in self.pretrained_backbone.named_parameters():
                    if 'norm' in name or '.ls' in name or 'bias' in name:
                        continue
                    if ('embed' in name or 'token' in name) and not self.freeze_embeddings:
                        continue
                    param.requires_grad = False

        elif 'clip' in self.pretrained_model_name:
            print('Loading clip')
            if self.use_pretrained_weights:
                self.pretrained_backbone = CLIPModel.from_pretrained(self.pretrained_model_name)
                if self.freeze_backbone:
                    print('Freezing Backbone')
                    for name, param in self.pretrained_backbone.named_parameters():
                        print(name)
                        if 'norm' in name or 'bias' in name:
                            continue
                        if 'embeddings' in name and not self.freeze_embeddings:
                            continue
                        param.requires_grad = False
            else:
                print(f'Loading randomly initialized model like {self.pretrained_model_name}')
                CLIPModelConfig = AutoConfig.from_pretrained(self.pretrained_model_name)
                self.pretrained_backbone = AutoModel.from_config(CLIPModelConfig)
        else:
            print('Not Implemented')
            exit()

    # def initialize_weights(self):
    #     if not self.use_pretrained_embeddings:
    #         pos_embed = get_2d_sincos_pos_embed(
    #             self.pos_embed.shape[-1],
    #             self.in_img_size[0] // self.patch_size,
    #             self.in_img_size[1] // self.patch_size,
    #             cls_token=False,
    #         )
    #         self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))
    #     if not self.use_pretrained_weights:
    #         self.apply(self._init_weights)

    # def _init_weights(self, m):
    #     if isinstance(m, nn.Linear):
    #         trunc_normal_(m.weight, std=0.02)
    #         if m.bias is not None:
    #             nn.init.constant_(m.bias, 0)
    #     elif isinstance(m, nn.LayerNorm):
    #         nn.init.constant_(m.bias, 0)
    #         nn.init.constant_(m.weight, 1.0)

    def unpatchify(self, x: torch.Tensor, h=None, w=None):
        """
        x: (B, L, V * patch_size**2)
        return imgs: (B, V, H, W)
        """
        p = self.patch_size
        v = self.out_channels
        h = self.out_img_size[0] // self.eff_patch_size[0] if h is None else h // p
        w = self.out_img_size[1] // self.eff_patch_size[1] if w is None else w // p
        assert h * w == x.shape[1]

        x = x.reshape(shape=(x.shape[0], h, w, self.eff_patch_size[0], self.eff_patch_size[1], v))
        x = torch.einsum("nhwpqv->nvhpwq", x)
        imgs = x.reshape(shape=(x.shape[0], v, h * self.eff_patch_size[0], w * self.eff_patch_size[1]))
        return imgs


    def forward_encoder(self, x, variables):
        # x.shape = [B,T*in_channels,H,W]
        if self.resize_img:
            x = torchvision.transforms.Resize((self.in_img_size[0] ,self.in_img_size[1]))(x)
        
        x = self.level_emb(x, variables)
        
        if 'google/vit' in self.pretrained_model_name:
            # x.shape = [B,num_patches,embed_dim]
            x = self.pretrained_backbone.encoder(x)
            x = x[0]
            x = self.pretrained_backbone.layernorm(x)
        elif 'dinov2' in self.pretrained_model_name:
            # print('Forward Encoder 2')
            # x.shape = [B,num_patches,embed_dim]
            # for blk in self.pretrained_backbone.blocks:
            #     x = blk(x)
            use_n_blocks = self.use_n_blocks if self.use_n_blocks is not None else len(self.pretrained_backbone.blocks)
            for i in range(use_n_blocks):
                blk = self.pretrained_backbone.blocks[i]
                x = blk(x)
            x = self.pretrained_backbone.norm(x)
        elif 'clip' in self.pretrained_model_name:
            # print('Forward Encoder 3')
            # x.shape = [B,num_patches,embed_dim]
            x = self.pretrained_backbone.vision_model.pre_layrnorm(x)
            x = self.pretrained_backbone.vision_model.encoder(x)
            x = x.last_hidden_state
        else:
            print('Not Implemented')
            exit()
        return x

    def forward(self, x, variables):
        # x.shape = [B,T,in_channels,H,W]
        x = x.flatten(1, 2)
        # x.shape = [B,T*in_channels,H,W]
        x = self.forward_encoder(x, variables)
        # x.shape = [B, num_patches, embed_dim]
        x = self.head(x)
        # x.shape = [B, num_patches, out_channels*patch_size**2]
        x = self.unpatchify(x)
        # x.shape = [B, out_channels, H, W]
        return x

# variables = [
#     "land_sea_mask",
#     "orography",
#     "lattitude",
#     "2m_temperature",
#     "10m_u_component_of_wind",
#     "10m_v_component_of_wind",
#     "geopotential_50",
#     "geopotential_250",
#     "geopotential_500",
#     "geopotential_600",
#     "geopotential_700",
#     "geopotential_850",
#     "geopotential_925",
#     "u_component_of_wind_50",
#     "u_component_of_wind_250",
#     "u_component_of_wind_500",
#     "u_component_of_wind_600",
#     "u_component_of_wind_700",
#     "u_component_of_wind_850",
#     "u_component_of_wind_925",
#     "v_component_of_wind_50",
#     "v_component_of_wind_250",
#     "v_component_of_wind_500",
#     "v_component_of_wind_600",
#     "v_component_of_wind_700",
#     "v_component_of_wind_850",
#     "v_component_of_wind_925",
#     "temperature_50",
#     "temperature_250",
#     "temperature_500",
#     "temperature_600",
#     "temperature_700",
#     "temperature_850",
#     "temperature_925",
#     "relative_humidity_50",
#     "relative_humidity_250",
#     "relative_humidity_500",
#     "relative_humidity_600",
#     "relative_humidity_700",
#     "relative_humidity_850",
#     "relative_humidity_925",
#     "specific_humidity_50",
#     "specific_humidity_250",
#     "specific_humidity_500",
#     "specific_humidity_600",
#     "specific_humidity_700",
#     "specific_humidity_850",
#     "specific_humidity_925",
# ]
# model = ViTPretrainedLevelEmb(
#     [32, 64],
#     [32, 64],
#     48,
#     3,
#     1,
#     False,
#     None,
#     False,
#     False,
#     True,
#     False,
#     2,
#     1024,
#     8,
#     "dinov2_vitl14",
#     1
# ).cuda()
# x = torch.randn(1, 1, 48, 32, 64).cuda()
# pred = model(x, variables)
# print (pred.shape)
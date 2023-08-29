#Local application
from .utils import register

#Third Party
import torch
import torch.nn as nn
from timm.models.vision_transformer import PatchEmbed, trunc_normal_
from transformers import Swinv2Config, Swinv2Model
from .climax import ClimaXEmbedding


@register('swinv2_classification')
class SwinV2Classification(nn.Module):

    def __init__(self,
        in_img_size,
        in_channels, 
        out_channels,
        patch_size=4,
        embed_type='normal', # normal or climax
        embed_norm=False,
        mlp_embed_depth=1,
        decoder_depth=1,
        freeze_backbone=False,
        freeze_embeddings=False,
        pretrained_weights=None,
        use_pretrained_weights=False,
        continuous_model=False,
    ):
        super().__init__()

        self.patch_size = patch_size
        self.in_img_size = in_img_size
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.embed_type = embed_type
        self.num_patches = (in_img_size[0] * in_img_size[1]) // (patch_size)**2
        self.use_pretrained_weights = use_pretrained_weights
        self.freeze_embeddings= freeze_embeddings
        self.freeze_backbone = freeze_backbone
        self.embed_norm = embed_norm
        self.continuous_model = continuous_model

        if use_pretrained_weights:
            self.pretrained_backbone = Swinv2Model.from_pretrained(pretrained_weights)
            self.config = self.pretrained_backbone.config
            print (f'Using a {pretrained_weights} pretrained backbone')
        else:
            self.config = Swinv2Config.from_pretrained(pretrained_weights)
            self.pretrained_backbone = Swinv2Model(self.config)
            print (f'Initializing a {pretrained_weights} random backbone')

        self.pretrained_img_size = self.config.image_size
        self.pretrained_patch_size = self.config.patch_size
        self.pretrained_embed_dim = self.config.embed_dim
        
        # Embedding layers
        if embed_type == 'normal':
            self.patch_embed = PatchEmbed(in_img_size, patch_size, in_channels, self.pretrained_embed_dim, flatten=False)
            self.pos_drop = nn.Dropout(p=0.1)

            self.mlp_embed = nn.ModuleList()
            for _ in range(mlp_embed_depth):
                self.mlp_embed.append(nn.GELU())
                self.mlp_embed.append(nn.Conv2d(in_channels=self.pretrained_embed_dim, out_channels=self.pretrained_embed_dim, kernel_size=1))
            self.mlp_embed = nn.Sequential(*self.mlp_embed)
        else:
            default_variables = [
                "land_sea_mask",
                "orography",
                "lattitude",
                "2m_temperature",
                "10m_u_component_of_wind",
                "10m_v_component_of_wind",
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
                "relative_humidity_50",
                "relative_humidity_250",
                "relative_humidity_500",
                "relative_humidity_600",
                "relative_humidity_700",
                "relative_humidity_850",
                "relative_humidity_925",
                "specific_humidity_50",
                "specific_humidity_250",
                "specific_humidity_500",
                "specific_humidity_600",
                "specific_humidity_700",
                "specific_humidity_850",
                "specific_humidity_925",
            ]

            self.embedding = ClimaXEmbedding(
                default_vars=default_variables,
                img_size=in_img_size,
                patch_size=patch_size,
                embed_dim=self.pretrained_embed_dim,
                num_heads=self.config.num_heads[0],
                drop_rate=0.1,
            )
        
        self.embed_norm_layer = nn.LayerNorm(self.pretrained_embed_dim) if embed_norm else nn.Identity()

        if self.continuous_model:
            self.lead_time_embed = nn.Linear(1, self.pretrained_embed_dim)
        
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

        self.initialize_weights()
        
        if freeze_backbone:
            # self.pretrained_backbone.requires_grad_(False)
            for name, param in self.pretrained_backbone.named_parameters():
                if 'norm' in name or 'bias' in name: # finetune norm layer
                    continue
                param.requires_grad = False

        if freeze_embeddings:
            if embed_type == 'normal':
                self.patch_embed.requires_grad_(False)
                self.mlp_embed.requires_grad_(False)
            elif embed_type =='climax':
                self.embedding.requires_grad_(False)

    def initialize_weights(self):
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

    def forward_encoder(self, x, variables, lead_times):
        # x.shape = [B,T*in_channels,H,W]
        x = torch.nn.functional.interpolate(x, size=self.in_img_size)

        if self.embed_type == 'normal':
            x = self.patch_embed(x)
        else:
            x = self.embedding(x, variables) # B, L, D
            x = x.transpose(1, 2).unflatten(
                2,
                sizes=(self.in_img_size[0] // self.patch_size, self.in_img_size[1] // self.patch_size)
            )
        
        # x.shape = [B,embed_dim,H,W]
        if self.embed_norm:
            x = x.permute(0,2,3,1).contiguous()
            x = self.embed_norm_layer(x)
            x = x.permute(0,3,1,2).contiguous()
        
        if self.embed_type == 'normal':
            x = self.mlp_embed(x)
            x = self.pos_drop(x)
        # x.shape = [B,embed_dim,H,W]

        if self.continuous_model:
            lead_time_emb = self.lead_time_embed(lead_times.unsqueeze(-1))
            # lead_time_emb = self.emb_lead_time(lead_times, x.shape[1], x.device)
            lead_time_emb = lead_time_emb[..., None, None]
            # lead_time_emb.shape = [B, embed_dim, 1, 1]
            x = x + lead_time_emb

        _, _, h, w = x.shape
        input_dimensions = (h, w)
        x = x.flatten(2).transpose(1, 2)
        x = self.pretrained_backbone.encoder(
            x,
            input_dimensions,
            output_attentions=None,
            output_hidden_states=True,
            return_dict=True,
        )
        x = x.reshaped_hidden_states[-1]
        return x

    def forward(self, x, variables, lead_times=None):
        if self.continuous_model:
            assert lead_times is not None, 'Lead times must be provided for continuous models'
        else:
            assert lead_times is None, 'Different lead times can only be provided to continuous models'
        
        if len(x.shape) == 5:
            x = x.flatten(1, 2)
        
        x = self.forward_encoder(x, variables, lead_times)
        x = self.head(x)
        return x

# model = SwinV2Classification(
#     in_img_size=(256, 256),
#     in_channels=3,
#     out_channels=3,
#     patch_size=4,
#     embed_type='climax',
#     embed_norm=True,
#     decoder_depth=2,
#     freeze_backbone=True,
#     freeze_embeddings=False,
#     pretrained_weights="microsoft/swinv2-large-patch4-window12to16-192to256-22kto1k-ft",
#     use_pretrained_weights=True,
#     continuous_model=True
# ).cuda()
# x = torch.randn(4, 1, 3, 128, 256).cuda()
# variables = ['2m_temperature', "10m_u_component_of_wind", "10m_v_component_of_wind"]
# lead_times = torch.rand(4).cuda()
# with torch.no_grad():
#     y = model(x, variables, lead_times)
# print (y.shape)
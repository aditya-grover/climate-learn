#Local application
from .utils import register

#Third Party
import torch
import torch.nn as nn
import torchvision
from transformers import CLIPVisionModel, CLIPVisionConfig
from timm.models.vision_transformer import PatchEmbed, trunc_normal_
from .climax import ClimaXEmbedding
from .prithvi import MaskedAutoencoderViT


@register('vit_pretrained')
class ViTPretrained(nn.Module):

    def __init__(self, 
        in_img_size,
        in_channels, 
        out_channels,
        patch_size=2,
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
        self.embed_norm = embed_norm
        self.num_patches = (in_img_size[0] * in_img_size[1]) // (patch_size)**2
        self.pretrained_weights = pretrained_weights
        self.use_pretrained_weights = use_pretrained_weights
        self.freeze_embeddings= freeze_embeddings
        self.freeze_backbone = freeze_backbone
        self.continuous_model = continuous_model

        self.load_pretrained_backbone(pretrained_weights) # set self.pretrained_embed_dim here

        # Embedding layers
        if embed_type == 'normal':
            self.patch_embed = PatchEmbed(in_img_size, patch_size, in_channels, self.pretrained_embed_dim, flatten=False)
            self.pos_embed = nn.Parameter(
                torch.zeros(1, self.num_patches, self.pretrained_embed_dim), requires_grad=True,
            )
            self.pos_drop = nn.Dropout(p=0.1)

            self.mlp_embed = nn.ModuleList()
            for _ in range(mlp_embed_depth):
                self.mlp_embed.append(nn.GELU())
                self.mlp_embed.append(nn.Linear(self.pretrained_embed_dim, self.pretrained_embed_dim))
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
                num_heads=self.num_heads,
                drop_rate=0.1,
            )
        
        self.embed_norm_layer = nn.LayerNorm(self.pretrained_embed_dim) if embed_norm else nn.Identity()

        if self.continuous_model:
            self.lead_time_embed = nn.Linear(1, self.pretrained_embed_dim)
        
        # prediction head
        self.head = nn.ModuleList()
        for _ in range(decoder_depth):
            self.head.append(nn.Linear(self.pretrained_embed_dim, self.pretrained_embed_dim))
            self.head.append(nn.GELU())
        self.head.append(nn.Linear(self.pretrained_embed_dim, out_channels*self.patch_size**2))
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

    def load_pretrained_backbone(self, pretrained_weights):
        if 'dinov2' in pretrained_weights:
            if self.use_pretrained_weights:
                print(f'Loading {self.pretrained_weights} weights')
                self.pretrained_backbone = torch.hub.load('facebookresearch/dinov2', pretrained_weights)
            else:
                print(f'Loading randomly initialized model like {self.pretrained_weights}')
                self.pretrained_backbone = torch.hub.load('facebookresearch/dinov2', pretrained_weights, pretrained=False)
            self.pretrained_embed_dim = self.pretrained_backbone.embed_dim
            self.num_heads = self.pretrained_backbone.num_heads

        elif 'clip' in pretrained_weights:
            if self.use_pretrained_weights:
                print(f'Loading {self.pretrained_weights} weights')
                self.pretrained_backbone = CLIPVisionModel.from_pretrained(self.pretrained_weights)
            else:
                print(f'Loading randomly initialized model like {self.pretrained_weights}')
                cfg = CLIPVisionConfig.from_pretrained(self.pretrained_weights)
                self.pretrained_backbone = CLIPVisionModel(cfg)
            self.pretrained_embed_dim = self.pretrained_backbone.config.hidden_size
            self.num_heads = self.pretrained_backbone.config.num_attention_heads
        elif 'nasa' in pretrained_weights:
            # hard-coded
            self.pretrained_backbone = MaskedAutoencoderViT(
                img_size=224,
                patch_size=16,
                num_frames=3,
                tubelet_size=1,
                in_chans=6,
                embed_dim=768,
                depth=12,
                num_heads=12,
                decoder_embed_dim=512,
                decoder_depth=8,
                decoder_num_heads=16,
            )
            if self.use_pretrained_weights:
                print ('Loading NASA weights')
                checkpoint = "https://huggingface.co/ibm-nasa-geospatial/Prithvi-100M/resolve/main/Prithvi_100M.pt"
                state_dict = torch.hub.load_state_dict_from_url(checkpoint, map_location='cpu')
                msg = self.pretrained_backbone.load_state_dict(state_dict)
                print (msg)
            else:
                print(f'Loading randomly initialized model like NASA')
            self.pretrained_embed_dim = 768
            self.num_heads = 12
        else:
            raise NotImplementedError('Only support Dinov2, CLIP, and NASA')

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

    def unpatchify(self, x: torch.Tensor, h=None, w=None):
        """
        x: (B, L, V * patch_size**2)
        return imgs: (B, V, H, W)
        """
        p = self.patch_size
        v = self.out_channels
        h = self.in_img_size[0] // self.patch_size if h is None else h // p
        w = self.in_img_size[1] // self.patch_size if w is None else w // p
        assert h * w == x.shape[1]

        x = x.reshape(shape=(x.shape[0], h, w, self.patch_size, self.patch_size, v))
        x = torch.einsum("nhwpqv->nvhpwq", x)
        imgs = x.reshape(shape=(x.shape[0], v, h * self.patch_size, w * self.patch_size))
        return imgs


    def forward_encoder(self, x, variables, lead_times):
        x = torch.nn.functional.interpolate(x, size=self.in_img_size)
        
        if self.embed_type == 'normal':
            x = self.patch_embed(x)
        else:
            x = self.embedding(x, variables) # B, L, D
        
        if self.embed_norm:
            x = self.embed_norm_layer(x)
        
        if self.embed_type == 'normal':
            x = self.mlp_embed(x)
            x = x + self.pos_embed
            x = self.pos_drop(x)

        if self.continuous_model:
            lead_time_emb = self.lead_time_embed(lead_times.unsqueeze(-1)).unsqueeze(1) # B, 1, D
            x = x + lead_time_emb
        
        if 'dinov2' in self.pretrained_weights or 'nasa' in self.pretrained_weights:
            for blk in self.pretrained_backbone.blocks:
                x = blk(x)
            x = self.pretrained_backbone.norm(x)
        elif 'clip' in self.pretrained_weights:
            x = self.pretrained_backbone.vision_model.pre_layrnorm(x)
            x = self.pretrained_backbone.vision_model.encoder(x)
            x = x.last_hidden_state
        else:
            raise NotImplementedError('Only support Dinov2 and CLIP')

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
        x = self.unpatchify(x)
        return x

# model = ViTPretrained(
#     in_img_size=(32, 64),
#     in_channels=3,
#     out_channels=3,
#     patch_size=2,
#     embed_type='climax',
#     embed_norm=True,
#     decoder_depth=2,
#     freeze_backbone=False,
#     freeze_embeddings=False,
#     # pretrained_weights="dinov2_vits14",
#     # pretrained_weights="openai/clip-vit-base-patch32",
#     pretrained_weights="nasa",
#     use_pretrained_weights=True,
#     continuous_model=True
# ).cuda()
# x = torch.randn(4, 1, 3, 32, 64).cuda()
# variables = ['2m_temperature', "10m_u_component_of_wind", "10m_v_component_of_wind"]
# lead_times = torch.rand(4).cuda()
# with torch.no_grad():
#     y = model(x, variables, lead_times)
# print (y.shape)
#Local application
from .utils import register

#Third Party
import torch
import torch.nn as nn
from timm.models.vision_transformer import trunc_normal_
from .segmentor import EncoderDecoder
from .swin_transformer import SwinTransformer, PatchEmbed
from .uper_head import UPerHead
from .fcn_head import FCNHead
from .climax import ClimaXEmbedding


@register('swin_pretrained_segmentation')
class SwinPretrainedSegmentation(nn.Module):

    def __init__(self,
        in_img_size,
        input_channels, 
        out_channels,
        embed_type='normal', # normal or climax
        mlp_embed_depth=0,
        decoder_depth=2,
        ckpt_path=None,
        freeze_backbone=False,
        freeze_embeddings=False,

        # Backbone
        patch_size=4,
        embed_dim=128,
        depths=[2, 2, 18, 2],
        num_heads=[4, 8, 16, 32],
        window_size=7,
        mlp_ratio=4.,
        qkv_bias=True,
        qk_scale=None,
        drop_rate=0.,
        attn_drop_rate=0.,
        drop_path_rate=0.3,
        ape=False,
        patch_norm=True,
        out_indices=(0, 1, 2, 3),
        use_checkpoint=False,

        # UPerHead
        in_channels=[128, 256, 512, 1024],
        in_index=[0, 1, 2, 3],
        pool_scales=[1, 2, 3, 6],
        channels=512,
        dropout_ratio=0.1,
        num_classes=150,
        align_corners=False,

        # FCN Head
        fcn_in_channels=512,
        fcn_in_index=2,
        fcn_channels=256,
        num_convs=1,
        concat_input=False,
        fcn_dropout_ratio=0.1,
        fcn_num_classes=150,
        fcn_align_corners=False,
    ):
        super().__init__()

        assert embed_type in ['normal', 'climax']

        self.input_channels = input_channels
        self.out_channels = out_channels
        self.embed_type = embed_type
        self.in_img_size = in_img_size
        self.patch_size = patch_size

        backbone = SwinTransformer(
            patch_size=patch_size,
            embed_dim=embed_dim,
            depths=depths,
            num_heads=num_heads,
            window_size=window_size,
            mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            drop_rate=drop_rate,
            attn_drop_rate=attn_drop_rate,
            drop_path_rate=drop_path_rate,
            ape=ape,
            patch_norm=patch_norm,
            out_indices=out_indices,
            use_checkpoint=use_checkpoint
        )

        uperhead = UPerHead(
            in_channels=in_channels,
            in_index=in_index,
            pool_scales=pool_scales,
            channels=channels,
            dropout_ratio=dropout_ratio,
            num_classes=num_classes,
            norm_cfg=dict(type='BN', requires_grad=True),
            align_corners=align_corners,
        )

        auxiliary_head = FCNHead(
            in_channels=fcn_in_channels,
            in_index=fcn_in_index,
            channels=fcn_channels,
            num_convs=num_convs,
            concat_input=concat_input,
            dropout_ratio=fcn_dropout_ratio,
            num_classes=fcn_num_classes,
            norm_cfg=dict(type='BN', requires_grad=True),
            align_corners=fcn_align_corners,
        )

        self.swin_pretrained_segmentor = EncoderDecoder(
            backbone=backbone,
            decode_head=uperhead,
            auxiliary_head=auxiliary_head
        )

        if ckpt_path is not None:
            print ('Loading pretrained segmentor from ' + ckpt_path)
            ckpt = torch.load(ckpt_path, map_location='cpu')
            state_dict = ckpt['state_dict']
            for k in list(state_dict.keys()):
                if k not in self.swin_pretrained_segmentor.state_dict().keys() or self.swin_pretrained_segmentor.state_dict()[k].shape != state_dict[k].shape:
                    print(f"Removing key {k} from pretrained checkpoint")
                    del state_dict[k]
            msg = self.swin_pretrained_segmentor.load_state_dict(state_dict, strict=False)
            print (msg)
            
        
        # Initialize embedding layers
        if self.embed_type == 'normal':
            self.embedding = nn.ModuleList()
            self.embedding.append(
                PatchEmbed(
                    patch_size=patch_size, in_chans=input_channels, embed_dim=embed_dim,
                    norm_layer=nn.LayerNorm if backbone.patch_norm else None
            ))
            # make patch embedding non-linear
            for _ in range(mlp_embed_depth):
                self.embedding.append(nn.ReLU())
                self.embedding.append(nn.Conv2d(
                    in_channels=embed_dim, out_channels=embed_dim, kernel_size=1
                ))
            self.embedding = nn.Sequential(*self.embedding)
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
                embed_dim=embed_dim,
                num_heads=num_heads[0],
                drop_rate=0.1,
            )
        
        # Initialize prediction head
        self.head = nn.ModuleList()
        for _ in range(decoder_depth):
            self.head.append(nn.Conv2d(
                in_channels=uperhead.channels, out_channels=uperhead.channels, kernel_size=1
            ))
            self.head.append(nn.GELU())
        self.head.append(nn.Conv2d(
            in_channels=uperhead.channels, out_channels=patch_size**2 * out_channels, kernel_size=1
        ))
        self.head.append(nn.PixelShuffle(patch_size))
        self.head = nn.Sequential(*self.head)

        if ckpt_path is None:
            self.apply(self._init_weights)

        if freeze_backbone:
            self.swin_pretrained_segmentor.requires_grad_(False)
        if freeze_embeddings:
            self.embedding.requires_grad_(False)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward_encoder(self, x, variables):
        # x.shape = [B,T*in_channels,H,W]
        if self.embed_type == 'normal':
            patches = self.embedding(x)
        else:
            patches = self.embedding(x, variables) # B, L, D
            patches = patches.transpose(1, 2).unflatten(
                2,
                sizes=(self.in_img_size[0] // self.patch_size, self.in_img_size[1] // self.patch_size)
            ) # B, D, h, w
        return self.swin_pretrained_segmentor.forward_features_given_patches(patches)

    def forward(self, x, variables):
        if len(x.shape) == 5:
            x = x.flatten(1, 2)
        x = self.forward_encoder(x, variables)
        x = self.head(x)
        return x

# model = SwinPretrainedSegmentation(
#     input_channels=3,
#     out_channels=3,
#     mlp_embed_depth=1,
#     decoder_depth=1,
#     patch_size=1,
#     ckpt_path='/home/tungnd/Swin-Transformer-Semantic-Segmentation/upernet_swin_base_patch4_window7_512x512.pth'
# ).cuda()
# x = torch.randn(2, 3, 32, 64).cuda()
# y = model(x)
# print (y.shape)
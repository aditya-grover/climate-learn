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


@register('swin_pretrained_segmentation')
class SwinPretrainedSegmentation(nn.Module):

    def __init__(self,
        input_channels, 
        out_channels,
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

        self.input_channels = input_channels
        self.out_channels = out_channels

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

    def forward_encoder(self, x):
        # x.shape = [B,T*in_channels,H,W]

        patches = self.embedding(x)
        return self.swin_pretrained_segmentor.forward_features_given_patches(patches)

    def forward(self, x):
        if len(x.shape) == 5:
            x = x.flatten(1, 2)
        x = self.forward_encoder(x)
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
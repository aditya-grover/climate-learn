#Local application
from .utils import register

#Third Party
import torch
import torch.nn as nn
import sys
import os
from argparse import Namespace
from timm.models.vision_transformer import PatchEmbed, trunc_normal_
from .climax import ClimaXEmbedding


sys.path.append(os.path.abspath('/home/tungnd/Mask2Former'))
from mask2former import add_maskformer2_config
from detectron2.engine import DefaultTrainer
from train_net_video import setup
from detectron2.checkpoint import DetectionCheckpointer


@register('mask2former')
class Mask2Former(nn.Module):

    def __init__(self, 
        in_img_size,
        in_channels, 
        out_channels, 
        embed_type='normal', # normal or climax
        mlp_embed_depth=1,
        decoder_depth=1,
        freeze_backbone=False,
        freeze_embeddings=False,
        use_pretrained_weights=False,
        patch_size=4, 
        embed_dim=192,
        out_embed_dim=256, 
        embed_norm=False,
    ):
        super().__init__()
        self.patch_size = patch_size
        self.in_img_size = in_img_size
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.embed_type = embed_type
        self.num_patches = (in_img_size[0] * in_img_size[1]) // (patch_size)**2
        self.embed_dim = embed_dim
        self.use_pretrained_weights = use_pretrained_weights
        self.freeze_embeddings= freeze_embeddings
        self.freeze_backbone = freeze_backbone
        self.embed_norm = embed_norm
        
        # load config of the segmentation model
        args = Namespace(
            config_file='/home/tungnd/Mask2Former/configs/youtubevis_2019/swin/video_maskformer2_swin_large_IN21k_384_bs16_8ep.yaml', 
            resume=False, 
            eval_only=True, 
            num_gpus=1, 
            num_machines=1, 
            machine_rank=0, 
            dist_url='tcp://127.0.0.1:56669', 
            opts=['MODEL.WEIGHTS', '/home/tungnd/Mask2Former/checkpoints/model_final_c5c739.pkl']
        )
        cfg = setup(args)
        self.cfg = cfg
        self.args = args

        # initialize embedding
        if embed_type == 'normal':
            self.patch_embed = PatchEmbed(in_img_size, patch_size, in_channels, embed_dim, flatten=False)
            self.pos_drop = nn.Dropout(p=0.1)

            self.mlp_embed = nn.ModuleList()
            for _ in range(mlp_embed_depth):
                self.mlp_embed.append(nn.GELU())
                self.mlp_embed.append(nn.Conv2d(in_channels=embed_dim, out_channels=embed_dim, kernel_size=1))
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
                embed_dim=embed_dim,
                num_heads=self.cfg['MODEL']['SWIN']['NUM_HEADS'][0],
                drop_rate=0.1,
            )
        
        self.embed_norm_layer = nn.LayerNorm(embed_dim) if embed_norm else nn.Identity()

        # initialize prediction head
        self.head = nn.ModuleList()
        for _ in range(decoder_depth):
            self.head.append(nn.Conv2d(
                in_channels=out_embed_dim, out_channels=out_embed_dim, kernel_size=1
            ))
            self.head.append(nn.GELU())
        self.head.append(nn.Conv2d(
            in_channels=out_embed_dim, out_channels=patch_size**2 * out_channels, kernel_size=1
        ))
        self.head.append(nn.PixelShuffle(patch_size))
        self.head = nn.Sequential(*self.head)

        self.initialize_weights()

        self.load_pretrained_model()

    def load_pretrained_model(self):
        print('Initializing Mask2Former')
        cfg = self.cfg
        model = DefaultTrainer.build_model(cfg)
        model.to('cpu')
        if self.use_pretrained_weights:
            print ('Loading pretrained Mask2Former')
            DetectionCheckpointer(model, save_dir=cfg.OUTPUT_DIR).resume_or_load(
                cfg.MODEL.WEIGHTS, resume=self.args.resume
            )
        model.backbone.patch_embed = nn.Identity()
        model.backbone.pos_drop = nn.Identity()
        model.sem_seg_head.predictor = None
        model.criterion = None
        self.pretrained_backbone = model

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


    def forward_encoder(self, x, variables):
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
        
        # x.shape = [B,192,H,W]
        x = self.pretrained_backbone.backbone(x)
        # x.shape = {Multi_Scale_Feautures}
        x, _, _ = self.pretrained_backbone.sem_seg_head.pixel_decoder.forward_features(x)
        # x.shape = [B,256,H,W]

        return x

    def forward(self, x, variables):
        # x.shape = [B,T*in_channels,H,W]
        if len(x.shape) == 5:
            x = x.flatten(1, 2)
        # x.shape = [B,T*in_channels,H,W]
        x = self.forward_encoder(x, variables)
        # x.shape = [B,out_embed_dim,H,W]
        x = self.head(x)
        # x.shape = [B,out_channels,H,W]
        return x
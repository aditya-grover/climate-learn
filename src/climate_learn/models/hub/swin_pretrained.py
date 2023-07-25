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
import os
from argparse import Namespace
from timm.models.vision_transformer import Block, PatchEmbed, trunc_normal_


sys.path.append(os.path.abspath('../../Mask2Former/'))
from mask2former import add_maskformer2_config
from detectron2.engine import DefaultTrainer
from train_net_video import setup
from detectron2.checkpoint import DetectionCheckpointer
from .climax import ClimaXEmbedding


@register('swin_pretrained')
class SwinPretrained(nn.Module):

    def __init__(self, 
        in_img_size,
        in_channels, 
        out_channels, 
        use_pretrained_weights=False,
        use_pretrained_embeddings=False,
        freeze_backbone=False, 
        freeze_embeddings=False,
        learn_pos_emb=False,
        resize_img=False,
        patch_size=16, 
        embed_dim=1024,
        out_embed_dim=1024, 
        decoder_depth=2,
        pretrained_model=None,
        mlp_embed_depth=0,
        embed_norm=False,
        embed_type='normal',
        default_variables=[],
    ):
        super().__init__()
        self.patch_size = patch_size
        self.in_img_size = in_img_size
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_patches = (in_img_size[0] * in_img_size[1]) // (patch_size)**2
        self.embed_dim = embed_dim
        self.use_pretrained_embeddings = use_pretrained_embeddings
        self.use_pretrained_weights = use_pretrained_weights
        self.freeze_embeddings= freeze_embeddings
        self.freeze_backbone = freeze_backbone
        self.pretrained_model_name = pretrained_model
        self.resize_img = resize_img
        self.embed_norm = embed_norm
        self.embed_type = embed_type


        self.args = Namespace(
                    config_file='/local/hbansal/Mask2Former/configs/youtubevis_2019/swin/video_maskformer2_swin_large_IN21k_384_bs16_8ep.yaml', 
                    resume=False, 
                    eval_only=True, 
                    num_gpus=1, 
                    num_machines=1, 
                    machine_rank=0, 
                    dist_url='tcp://127.0.0.1:56669', 
                    opts=['MODEL.WEIGHTS', '/local/hbansal/Mask2Former/checkpoints/model_final_c5c739.pkl']
        )
        self.cfg = setup(self.args)

        self.load_pretrained_model()
        
        if not use_pretrained_embeddings:
            if embed_norm:
                self.embed_norm_layer = nn.LayerNorm(embed_dim)
            if embed_type == 'normal':
                self.patch_embed = PatchEmbed(in_img_size, patch_size, in_channels, embed_dim, 
                    flatten=False)
                self.pos_drop = nn.Dropout(p=0.1)

                self.mlp_embed = nn.ModuleList()
                for _ in range(mlp_embed_depth):
                    self.mlp_embed.append(nn.GELU())
                    self.mlp_embed.append(nn.Conv2d(in_channels=embed_dim, out_channels=embed_dim, kernel_size=1))
                self.mlp_embed = nn.Sequential(*self.mlp_embed)
            else:
                self.embedding = ClimaXEmbedding(
                    default_vars=default_variables,
                    img_size=in_img_size,
                    patch_size=patch_size,
                    embed_dim=embed_dim,
                    num_heads=self.cfg['MODEL']['SWIN']['NUM_HEADS'][0],
                    drop_rate=0.1
                )
            print('Using new embeddings')

        # Initialize prediction head
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

    def load_pretrained_model(self):
        if 'mask2former' in self.pretrained_model_name:
            print('Loading Mask2Former')
            model = DefaultTrainer.build_model(self.cfg)
            model.to('cpu')
            if self.use_pretrained_weights:
                DetectionCheckpointer(model, save_dir=self.cfg.OUTPUT_DIR).resume_or_load(
                    self.cfg.MODEL.WEIGHTS, resume=self.args.resume
                )
            model.backbone.patch_embed = nn.Identity()
            model.backbone.pos_drop = nn.Identity()
            model.sem_seg_head.predictor = None
            model.criterion = None
            if self.freeze_backbone:
                for name, param in model.named_parameters():
                    if 'bias' in name or 'norm' in name:
                        continue
                    param.requires_grad = False
            self.pretrained_backbone = model
        else:
            print('Not Implemented')
            exit()

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


    def forward_encoder(self, x, variables, periodic=False):
        # x.shape = [B,T*in_channels,H,W]
        if self.resize_img:
            # x = torchvision.transforms.Resize((self.in_img_size[0] ,self.in_img_size[1]))(x)
            x = torch.nn.functional.interpolate(x, (self.in_img_size[0], self.in_img_size[1]))
        if not self.use_pretrained_embeddings:
            if self.embed_type == 'normal':  
                x = self.patch_embed(x)
                # x.shape = [B,embed_dim,H,W]
                if self.embed_norm:
                    x = x.permute(0,2,3,1).contiguous()
                    x = self.embed_norm_layer(x)
                    x = x.permute(0,3,1,2).contiguous()
                x = self.mlp_embed(x)
                # x.shape = [B,embed_dim,H,W]
                x = self.pos_drop(x)
                # x.shape = [B,num_patches+1,embed_dim]
            else:
                x = self.embedding(x, variables)
                x = x.transpose(1, 2).unflatten(
                    2,
                    sizes=(self.in_img_size[0] // self.patch_size, self.in_img_size[1] // self.patch_size)
                )
                if self.embed_norm:
                    x = x.permute(0,2,3,1).contiguous()
                    x = self.embed_norm_layer(x)
                    x = x.permute(0,3,1,2).contiguous()
        
        if 'mask2former' in self.pretrained_model_name:
            # print('Mask2Former')
            if self.use_pretrained_embeddings:
                print('Not Implemented')
                exit()
            else:
                # x.shape = [B,192,H,W]
                x = self.pretrained_backbone.backbone(x, periodic)
                # x.shape = {Multi_Scale_Feautures}
                x, _, _ = self.pretrained_backbone.sem_seg_head.pixel_decoder.forward_features(x)
                # x.shape = [B,256,H,W]
        else:
            print('Not Implemented')
            exit()
        return x

    def forward(self, x, variables, periodic=False):
        # x.shape = [B,T*in_channels,H,W]
        # x = x.flatten(1, 2)
        # x.shape = [B,T*in_channels,H,W]
        x = self.forward_encoder(x, variables, periodic)
        # x.shape = [B,out_embed_dim,H,W]
        x = self.head(x)
        # x.shape = [B,out_channels,H,W]
        return x


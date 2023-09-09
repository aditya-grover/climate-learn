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



#Local application
from .utils import register
from detectron2.engine import DefaultTrainer
from detectron2.checkpoint import DetectionCheckpointer


@register('mask2former_pred_all')
class Mask2FormerPredictAll(nn.Module):

    def __init__(self, 
        in_img_size,
        decoder_depth=1,
        freeze_backbone=False,
        freeze_embeddings=False,
        use_pretrained_weights=False,
        patch_size=4, 
        embed_dim=192,
        out_embed_dim=256, 
        embed_norm=False,
        continuous_model=False,
        mask2former_dir=None,
        pretrained_weights=None,
    ):
        super().__init__()
        self.patch_size = patch_size
        self.in_img_size = in_img_size
        self.num_patches = (in_img_size[0] * in_img_size[1]) // (patch_size)**2
        self.embed_dim = embed_dim
        self.use_pretrained_weights = use_pretrained_weights
        self.freeze_embeddings= freeze_embeddings
        self.freeze_backbone = freeze_backbone
        self.embed_norm = embed_norm
        self.continuous_model = continuous_model
        # load config of the segmentation model
        sys.path.append(mask2former_dir)
        if pretrained_weights == 'video':
            mask2former_config_file = f'{mask2former_dir}/configs/youtubevis_2019/swin/video_maskformer2_swin_large_IN21k_384_bs16_8ep.yaml'
            mask2former_opts = ['MODEL.WEIGHTS', f'{mask2former_dir}/checkpoints/model_final_c5c739.pkl']
            from train_net_video import setup
        elif pretrained_weights == 'image':
            # mask2former_config_file = f'{mask2former_dir}/configs/coco/panoptic-segmentation/swin/maskformer2_swin_large_IN21k_384_bs16_100ep.yaml'
            # mask2former_opts = ['MODEL.WEIGHTS', f'{mask2former_dir}/checkpoints/model_final_f07440.pkl']
            mask2former_config_file = f'{mask2former_dir}/configs/ade20k/semantic-segmentation/swin/maskformer2_swin_large_IN21k_384_bs16_160k_res640.yaml'
            mask2former_opts = ['MODEL.WEIGHTS', f'{mask2former_dir}/checkpoints/model_final_6b4a3a.pkl']
            from train_net import setup
        else:
            print('Pretrained weights must be either video or image')
            exit()
        args = Namespace(
            config_file=mask2former_config_file,
            resume=False,
            eval_only=True,
            num_gpus=1,
            num_machines=1,
            machine_rank=0,
            dist_url='tcp://127.0.0.1:56669',
            opts=mask2former_opts,
        )
        cfg = setup(args)
        self.cfg = cfg
        self.args = args

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

        if self.continuous_model:
            self.lead_time_embed = nn.Linear(1, embed_dim)

        # initialize prediction head
        self.head = nn.ModuleList()
        for _ in range(decoder_depth):
            self.head.append(nn.Conv2d(
                in_channels=out_embed_dim, out_channels=out_embed_dim, kernel_size=1
            ))
            self.head.append(nn.GELU())
        self.head.append(nn.Conv2d(
            in_channels=out_embed_dim, out_channels=patch_size**2 * len(default_variables), kernel_size=1
        ))
        self.head.append(nn.PixelShuffle(patch_size))
        self.head = nn.Sequential(*self.head)

        self.initialize_weights()

        self.load_pretrained_model()

        if freeze_backbone:
            # self.pretrained_backbone.requires_grad_(False)
            for name, param in self.pretrained_backbone.named_parameters():
                if 'norm' in name or 'bias' in name: # finetune norm layer
                    continue
                param.requires_grad = False

        if freeze_embeddings:
            self.embedding.requires_grad_(False)

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


    def forward_encoder(self, x, variables, lead_times):
        # x.shape = [B,T*in_channels,H,W]
        x = torch.nn.functional.interpolate(x, size=self.in_img_size)

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
        
        if self.continuous_model:
            lead_time_emb = self.lead_time_embed(lead_times.unsqueeze(-1))
            lead_time_emb = lead_time_emb[..., None, None]
            # lead_time_emb.shape = [B, embed_dim, 1, 1]
            x = x + lead_time_emb

        
        # x.shape = [B,192,H,W]
        x = self.pretrained_backbone.backbone(x)
        # x.shape = {Multi_Scale_Feautures}
        x, _, _ = self.pretrained_backbone.sem_seg_head.pixel_decoder.forward_features(x)
        # x.shape = [B,256,H,W]

        return x

    def forward(self, x, variables, out_variables, lead_times=None):
        if self.continuous_model:
            assert lead_times is not None, 'Lead times must be provided for continuous models'
        else:
            assert lead_times is None, 'Different lead times can only be provided to continuous models'

        # x.shape = [B,T*in_channels,H,W]
        if len(x.shape) == 5:
            x = x.flatten(1, 2)
        # x.shape = [B,T*in_channels,H,W]
        x = self.forward_encoder(x, variables, lead_times)
        # x.shape = [B,out_embed_dim,H,W]
        x = self.head(x)
        out_var_ids = self.embedding.get_var_ids(tuple(out_variables), x.device)
        x = x[:, out_var_ids]
        # x.shape = [B,out_channels,H,W]
        return x
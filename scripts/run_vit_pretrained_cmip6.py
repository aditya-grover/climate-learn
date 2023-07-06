import climate_learn as cl
from climate_learn.data import IterDataModule
from climate_learn.utils.datetime import Hours

import torch
import yaml
import os
from pytorch_lightning.loggers.tensorboard import TensorBoardLogger
from datetime import datetime

now = datetime.now()
now = now.strftime("%H-%M-%S_%d-%m-%Y")

os.environ["NCCL_P2P_DISABLE"] = "1"

def main():
    with open('scripts/configs/config_cmip6.yaml') as f:
        cfg = yaml.safe_load(f)
    
    default_root_dir=f"results_pretrained_cmip6/dinov2_vitb14_scratch_5e-4"
    
    dm = IterDataModule(
        task='forecasting',
        inp_root_dir=cfg['data_dir'],
        out_root_dir=cfg['data_dir'],
        in_vars=cfg['in_variables'] + cfg['constants'],
        out_vars=cfg['out_variables'],
        history=cfg['history'],
        window=cfg['window'],
        pred_range=Hours(cfg['pred_range']),
        subsample=Hours(cfg['subsample']),
        batch_size=cfg['batch_size'],
        num_workers=cfg['num_workers'],
    )

    #VIT Pretrained
    vit_pretrained = cl.load_forecasting_module(
        data_module=dm, 
        preset=cfg['model'], 
        cfg=cfg,
    )

    # state_dict = torch.load('/home/tungnd/climate-learn/results_pretrained/dinov2_vitl14_1_mlp_1_decoder_freeze_embed_5e-4_finetune_backbone_and_head_5e-5_interpolate_224_patchsize_14/checkpoints/epoch_005.ckpt')['state_dict']
    # msg = vit_pretrained.load_state_dict(state_dict)
    # print (msg)

    logger = TensorBoardLogger(
        save_dir=f"{default_root_dir}/logs"
    )

    trainer = cl.Trainer(
        early_stopping="val/lat_mse:aggregate",
        patience=cfg["patience"],
        accelerator="gpu",
        devices=cfg["gpu"],
        precision=16,
        max_epochs=cfg["num_epochs"],
        default_root_dir=default_root_dir,
        logger=logger
    )

    trainer.fit(vit_pretrained, datamodule=dm)
    trainer.test(vit_pretrained, datamodule=dm, ckpt_path="best")

if __name__ == "__main__":
    main()
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
    with open('scripts/configs/config_cmip6_mask2former.yaml') as f:
        cfg = yaml.safe_load(f)

    default_root_dir=f"{cfg['default_root_dir']}/mask2former_{cfg['embed_type']}_emb_pretrained_{cfg['use_pretrained_weights']}/"
    
    dm = IterDataModule(
        task='forecasting',
        inp_root_dir=cfg['data_dir'],
        out_root_dir=cfg['data_dir'],
        in_vars=cfg['in_variables'],
        out_vars=cfg['out_variables'],
        history=cfg['history'],
        window=cfg['window'],
        pred_range=Hours(cfg['pred_range']),
        subsample=Hours(cfg['subsample']),
        batch_size=cfg['batch_size'],
        num_workers=cfg['num_workers'],
    )

    # load module
    module = cl.load_forecasting_module(
        data_module=dm, 
        preset=cfg['model'], 
        cfg=cfg,
    )

    if cfg['ckpt'] is not None:
        state_dict = torch.load(f'results_pretrained_cmip6_stage1/mask2former_{cfg["embed_type"]}_emb_pretrained_{cfg["use_pretrained_weights"]}/checkpoints/', map_location='cpu')['state_dict']
        msg = module.load_state_dict(state_dict)
        print (msg)
    
    # logger = TensorBoardLogger(
    #     save_dir=f"{default_root_dir}/logs"
    # )

    wandb.init(
        project='Climate', 
        name=f"{cfg['model'].upper()}, Pretrained Backbone = {cfg['use_pretrained_weights']}", 
        config=cfg
    )
    wandb_logger = WandbLogger()

    trainer = cl.Trainer(
        early_stopping="val/lat_mse:aggregate",
        patience=cfg["patience"],
        accelerator="gpu",
        devices=cfg["gpu"],
        precision=16,
        max_epochs=cfg["num_epochs"],
        default_root_dir=default_root_dir,
        logger=wandb_logger,
    )

    trainer.fit(module, datamodule=dm)
    trainer.test(module, datamodule=dm, ckpt_path="best")


if __name__ == "__main__":
    main()
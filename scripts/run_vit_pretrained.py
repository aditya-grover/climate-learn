import climate_learn as cl
from climate_learn.data import IterDataModule
from climate_learn.utils.datetime import Hours

import wandb
import yaml
import os
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.loggers.tensorboard import TensorBoardLogger
from pytorch_lightning.callbacks import LearningRateMonitor, TQDMProgressBar
from datetime import datetime

now = datetime.now()
now = now.strftime("%H-%M-%S_%d-%m-%Y")

os.environ["NCCL_P2P_DISABLE"] = "1"

def main():
    with open('scripts/configs/config.yaml') as f:
        cfg = yaml.safe_load(f)
    
    default_root_dir=f"results_pretrained/dinov2_vits14_scratch_emb_finetune_backbone_all_blocks"
    
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

    # wandb.init(
    #     project='Climate', 
    #     name=f"{cfg['model'].upper()}, Pretrained Backbone = {cfg['use_pretrained_weights']}, \
    #         New Embeddings = {not cfg['use_pretrained_embeddings']}, Frozen Backbone = {cfg['freeze_backbone']}, \
    #             Frozen Embeddings = {cfg['freeze_embeddings']}, {now}", 
    #     config=cfg
    # )
    # logger = WandbLogger()
    logger = TensorBoardLogger(
        save_dir=f"{default_root_dir}/logs"
    )

    lr_monitor = LearningRateMonitor(logging_interval='step')
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

    # print('Testing Climatology')
    # trainer.test(climatology, dm)
# 
    # print('Testing Persistence')
    # trainer.test(persistence, dm)
# 
    # print('Testing VIT Pretrained')
    # trainer.test(vit_pretrained, dm)


if __name__ == "__main__":
    main()
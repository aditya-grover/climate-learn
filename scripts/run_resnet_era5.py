import climate_learn as cl
from climate_learn.data import IterDataModule, ContinuousIterDataModule
from climate_learn.utils.datetime import Hours

import torch
import yaml
import os
import wandb
from pytorch_lightning.loggers.tensorboard import TensorBoardLogger
from pytorch_lightning.loggers import WandbLogger
from datetime import datetime


def get_best_checkpoint(dir):
    ckpt_paths = os.listdir(f'{dir}/checkpoints/')
    assert len(ckpt_paths) == 2
    for ckpt_path in ckpt_paths:
        if 'last' not in ckpt_paths:
            return os.path.join(dir, 'checkpoints/', ckpt_path)

# os.environ["NCCL_P2P_DISABLE"] = "1"

def main():
    with open('scripts/configs/config_era5_resnet.yaml') as f:
        cfg = yaml.safe_load(f)
    
    default_root_dir=f"{cfg['default_root_dir']}/resnet_lead_time_{cfg['pred_range']}/"
    os.makedirs(default_root_dir, exist_ok=True)
    
    dm = ContinuousIterDataModule(
        task='forecasting',
        inp_root_dir=cfg['data_dir'],
        out_root_dir=cfg['data_dir'],
        in_vars=cfg['in_variables'],
        out_vars=cfg['out_variables'],
        history=cfg['history'],
        window=cfg['window'],
        random_lead_time=False,
        min_pred_range=Hours(cfg['pred_range']),
        max_pred_range=Hours(cfg['pred_range']),
        hrs_each_step=Hours(cfg['hrs_each_step']),
        subsample=Hours(cfg['subsample']),
        batch_size=cfg['batch_size'],
        num_workers=cfg['num_workers'],
        pin_memory=True,
    )

    # load module
    module = cl.load_forecasting_module(
        data_module=dm, 
        preset=cfg['model'], 
        cfg=cfg,
    )

    
    if cfg['ckpt_dir'] is not None:
        ckpt_path = get_best_checkpoint(f"{cfg['ckpt_dir']}/resnet")
        state_dict = torch.load(f'{ckpt_path}', map_location='cpu')['state_dict']
        msg = module.load_state_dict(state_dict)
        print(msg)

    # tb_logger = TensorBoardLogger(
        # save_dir=f"{default_root_dir}/logs"
    # )
    wandb.init(
        dir=default_root_dir,
        project='climate-vision23',
        name=f"ERA5, Resnet, Lead Time = {cfg['pred_range']}", 
        config=cfg,
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
        logger=[wandb_logger],
        accumulate_grad_batches=cfg['grad_acc'],
        val_check_interval=cfg['val_every_n_steps'],
        min_delta=cfg['min_delta'],
    )

    trainer.fit(module, datamodule=dm)
    trainer.test(module, datamodule=dm, ckpt_path="best")


if __name__ == "__main__":
    main()
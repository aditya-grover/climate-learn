import climate_learn as cl
from climate_learn.data import ContinuousIterDataModule, IterDataModule
from climate_learn.utils.datetime import Hours

import os
import torch
import yaml
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

os.environ["NCCL_P2P_DISABLE"] = "1"

def main():
    with open('scripts/configs/config_cmip6_mask2former_stage1.yaml') as f:
        cfg = yaml.safe_load(f)
    
    default_root_dir=f"{cfg['default_root_dir']}/mask2former_{cfg['pretrained_weights']}_{cfg['embed_type']}_emb_pretrained_{cfg['use_pretrained_weights']}/"
    os.makedirs(default_root_dir, exist_ok=True)


    dm = ContinuousIterDataModule(
        task='forecasting',
        inp_root_dir=cfg['data_dir'],
        out_root_dir=cfg['data_dir'],
        in_vars=cfg['in_variables'],
        out_vars=cfg['out_variables'],
        history=cfg['history'],
        window=cfg['window'],
        random_lead_time=True,
        min_pred_range=Hours(cfg['min_pred_range']),
        max_pred_range=Hours(cfg['max_pred_range']),
        hrs_each_step=Hours(cfg['hrs_each_step']),
        subsample=Hours(cfg['subsample']),
        batch_size=cfg['batch_size'],
        num_workers=cfg['num_workers'],
        fixed_lead_time_eval=cfg['fixed_lead_time_eval'],
        pin_memory=True,
    )

    # load module
    module = cl.load_forecasting_module(
        data_module=dm, 
        preset=cfg['model'], 
        cfg=cfg,
    )

    if cfg['ckpt_dir'] is not None:
        ckpt_path = get_best_checkpoint(f"{cfg['ckpt_dir']}/mask2former_{cfg['pretrained_weights']}_{cfg['embed_type']}_emb_pretrained_{cfg['use_pretrained_weights']}/")
        state_dict = torch.load(f'{ckpt_path}', map_location='cpu')['state_dict']
        msg = module.load_state_dict(state_dict)
        print(msg)
    
    # tb_logger = TensorBoardLogger(
        # save_dir=f"{default_root_dir}/logs"
    # )
    wandb.init(
        project='climate-vision23',
        dir=default_root_dir,
        name=f"{cfg['model'].upper()}, Pretrained Backbone = {cfg['use_pretrained_weights']} Stage = {cfg['stage']}, Model = {cfg['pretrained_weights']}", 
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
        logger=[wandb_logger],
        accumulate_grad_batches=cfg['grad_acc'],
        val_check_interval=cfg['val_every_n_steps'],
    )

    trainer.fit(module, datamodule=dm)
    trainer.test(module, datamodule=dm, ckpt_path="best")


if __name__ == "__main__":
    main()
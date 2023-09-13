from argparse import ArgumentParser

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


def get_last_checkpoint(dir):
    ckpt_paths = os.listdir(os.path.join(dir, 'checkpoints'))
    assert len(ckpt_paths) == 2
    for ckpt_path in ckpt_paths:
        if 'last' in ckpt_path:
            return os.path.join(dir, 'checkpoints/', ckpt_path)

### comment this line if not running on mint clusters
# os.environ["NCCL_P2P_DISABLE"] = "1"

def main():
    parser = ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--nodes", type=int, required=True)
    parser.add_argument("--gpus", type=int, required=True)
    args = parser.parse_args()

    config_path = args.config
    nodes = args.nodes
    gpus = args.gpus

    with open(config_path) as f:
        cfg = yaml.safe_load(f)
    
    model_name = cfg['model']

    if '/' in cfg['pretrained_weights']:
        pretrained_name = cfg['pretrained_weights'].split('/')[1]
    else:
        pretrained_name = cfg['pretrained_weights']
    
    default_root_dir = os.path.join(cfg['default_root_dir'], f"{model_name}_{pretrained_name}_{cfg['embed_type']}_emb_pretrained_{cfg['use_pretrained_weights']}")
    ### can uncomment the following lines if not running on mint clusters
    # if os.path.exists(default_root_dir):
    #     print('Root directory exists')
    #     exit()
    # else:
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
    module = cl.load_forecasting_module_train_only(
        data_module=dm, 
        preset=cfg['model'], 
        cfg=cfg,
    )

    if cfg['ckpt_dir'] is not None:
        ckpt_path = get_last_checkpoint(os.path.join(cfg['ckpt_dir'], f"{model_name}_{pretrained_name}_{cfg['embed_type']}_emb_pretrained_{cfg['use_pretrained_weights']}"))
        state_dict = torch.load(f'{ckpt_path}', map_location='cpu')['state_dict']
        msg = module.load_state_dict(state_dict)
        print(msg)
    
    # tb_logger = TensorBoardLogger(
        # save_dir=f"{default_root_dir}/logs"
    # )
    # wandb.init(
    #     project='climate-vision23',
    #     dir=default_root_dir,
    #     name=f"{model_name.upper()}, Pretrained Backbone = {cfg['use_pretrained_weights']} Stage = {cfg['stage']}, Model = {pretrained_name}", 
    #     config=cfg
    # )
    wandb_logger = WandbLogger(
        project='climate-vision23',
        save_dir=default_root_dir,
        name=f"{model_name.upper()}, Pretrained Backbone = {cfg['use_pretrained_weights']} Stage = {cfg['stage']}, Model = {pretrained_name}", 
    )

    trainer = cl.Trainer(
        # early_stopping="val/lat_mse:aggregate",
        early_stopping=None,
        patience=cfg["patience"],
        accelerator="gpu",
        devices=gpus,
        num_nodes=nodes,
        precision=16,
        max_epochs=cfg["num_epochs"],
        default_root_dir=default_root_dir,
        logger=[wandb_logger],
        accumulate_grad_batches=cfg['grad_acc'],
        val_check_interval=cfg['val_every_n_steps'],
        num_sanity_val_steps=0,
    )
    
    if os.path.exists(os.path.join(default_root_dir, 'checkpoints', 'last.ckpt')):
        ckpt_resume_path = os.path.join(default_root_dir, 'checkpoints', 'last.ckpt')
    else:
        ckpt_resume_path = None

    trainer.fit(module, datamodule=dm, ckpt_path=ckpt_resume_path)
    # trainer.test(module, datamodule=dm, ckpt_path="best")


if __name__ == "__main__":
    main()
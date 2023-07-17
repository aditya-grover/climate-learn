import climate_learn as cl
from climate_learn.data.climate_dataset.args import ERA5Args
from climate_learn.data.task.args import ForecastingArgs
from climate_learn.data.dataset.args import MapDatasetArgs
from climate_learn.data import IterDataModule, CMIP6IterDataModule
from climate_learn.utils.datetime import Hours

import wandb
import yaml
import os
import torch
from pytorch_lightning import seed_everything
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import LearningRateMonitor, RichProgressBar, ModelCheckpoint
from datetime import datetime
from transformers import AutoConfig, ViTImageProcessor


def load_config(cfg_file):
    with open(cfg_file) as f:
        cfg = yaml.safe_load(f)
    return cfg

def load_data_xarray(cfg):
    seed_everything(cfg["seed"], workers=True)

    in_vars = [f"era5:{v}" for v in cfg["in_variables"]]
    out_vars = [f"era5:{v}" for v in cfg["out_variables"]]
    consts = [f"era5:{v}" for v in cfg["constants"]]
    train_years = range(*cfg["train_years"])
    val_years = range(*cfg["val_years"])
    test_years = range(*cfg["test_years"])

    forecasting_args = ForecastingArgs(
        in_vars,
        out_vars,
        consts,
        pred_range=cfg["pred_range"],
        subsample=cfg["subsample"],
    )

    train_dataset_args = MapDatasetArgs(
        ERA5Args(cfg["data_dir"], cfg["in_variables"], train_years, cfg["constants"]),
        forecasting_args
    )

    val_dataset_args = MapDatasetArgs(
        ERA5Args(cfg["data_dir"], cfg["in_variables"], val_years, cfg["constants"]),
        forecasting_args
    )

    test_dataset_args = MapDatasetArgs(
        ERA5Args(cfg["data_dir"], cfg["in_variables"], test_years, cfg["constants"]),
        forecasting_args
    )

    dm = cl.data.DataModule(
        train_dataset_args,
        val_dataset_args,
        test_dataset_args,
        batch_size=cfg["batch_size"],
        num_workers=cfg["num_workers"],
    )

    return dm


def load_data_numpy(cfg):
    seed_everything(cfg["seed"], workers=True)

    if cfg['dataset'] == 'era5':
        data_module_cls = IterDataModule
    elif cfg['dataset'] == 'cmip6':
        data_module_cls = CMIP6IterDataModule
    else:
        print('Invalid Dataset')
        exit()

    dm = data_module_cls(
        task='forecasting',
        inp_root_dir=cfg['data_dir'],
        out_root_dir=cfg['data_dir'],
        in_vars=cfg['in_variables'] + cfg['constants'],
        out_vars=cfg['out_variables'],
        constants=cfg['constants'],
        history=cfg['history'],
        window=cfg['window'],
        pred_range=Hours(cfg['pred_range']),
        subsample=Hours(cfg['subsample']),
        batch_size=cfg['batch_size'],
        num_workers=cfg['num_workers'],
    )
    dm.setup()
    return dm


def load_trainer(cfg):

    wandb.init(
        project='Climate', 
        name=f"{cfg['model'].upper()}, Pretrained Backbone = {cfg['use_pretrained_weights']}", 
        config=cfg
    )

    lr_monitor = LearningRateMonitor(logging_interval='step')
    progress_bar = RichProgressBar()
    checkpoint_callback = ModelCheckpoint(save_top_k=1, monitor='lat_rmse:aggregate [val]', mode='min', save_last=True)

    logger = WandbLogger()
    trainer = cl.Trainer(
        # stop when latitude-weighted RMSE, a validation metric, stops improving
        early_stopping="lat_rmse:aggregate [val]",
        # wait for 10 epochs of no improvement
        patience=cfg["patience"],
        # uncomment to use gpu acceleration
        accelerator="gpu",
        devices=[cfg["gpu"]],
        # max epochs
        max_epochs=cfg["num_epochs"],
        # log to wandb
        logger=logger,
        callbacks=[lr_monitor, progress_bar, checkpoint_callback],
        val_check_interval=cfg['val_every_n_steps'],
    )
    return trainer


def load_checkpoint(ckpt, module, dm, trainer):
    ckpt_dir = f'lightning_logs/{ckpt}/checkpoints/'
    ckpt_files = os.listdir(ckpt_dir)
    if len(ckpt_files) == 1:
        ckpt_path = os.path.join(ckpt_dir, ckpt_files[0])
        ckpt = torch.load(ckpt_path)
    elif len(ckpt_files) == 2:
        if 'last' in ckpt_files[0]:
            ckpt_path = os.path.join(ckpt_dir, ckpt_files[1])
        else:
            ckpt_path = os.path.join(ckpt_dir, ckpt_files[0])
        ckpt = torch.load(ckpt_path)
    else:
        print('Too many checkpoints in the directory')
        exit()
    module.load_state_dict(ckpt['state_dict'])
    print(f'Loaded checkpoint from {ckpt_path}')

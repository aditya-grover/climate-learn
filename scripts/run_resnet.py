# Standard library
from argparse import ArgumentParser
import yaml
from datetime import datetime

# Third party
import climate_learn as cl
import wandb
from climate_learn.data import IterDataModule, CMIP6IterDataModule
from climate_learn.utils.datetime import Hours
from climate_learn.data.climate_dataset.era5.constants import *
import torch.multiprocessing
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning import seed_everything
from pytorch_lightning.callbacks import LearningRateMonitor, RichProgressBar

now = datetime.now()
now = now.strftime("%H-%M-%S_%d-%m-%Y")

def load_config(cfg_file):
    with open(cfg_file) as f:
        cfg = yaml.safe_load(f)
    return cfg

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

def main():
    torch.multiprocessing.set_sharing_strategy("file_system")

    parser = ArgumentParser()
    parser.add_argument('--config', default='resnet')
    args = parser.parse_args()
    cfg = load_config(f'configs/{args.config}.yaml')
    
    dm = load_data_numpy(cfg)

    model = cl.models.hub.ResNet(
        in_channels=len(cfg['in_variables'])*cfg['history'] + len(cfg['constants']),
        out_channels=len(cfg['out_variables']),
        history=1,
        hidden_channels=cfg['hidden_channels'],
        activation=cfg['activation'],
        norm=cfg['norm'],
        dropout=cfg['dropout'],
        n_blocks=cfg['n_blocks'],
    )

    optimizer = cl.load_optimizer(
        model, "AdamW", {"lr": cfg['lr'], "weight_decay": cfg['weight_decay'], "betas": cfg['betas']}
    )
    lr_scheduler = cl.load_lr_scheduler(
        "linear-warmup-cosine-annealing",
        optimizer,
        {"warmup_epochs": cfg['warmup_epochs'], "max_epochs": cfg['max_epochs'], "warmup_start_lr": cfg['warmup_start_lr'], "eta_min": cfg['eta_min']}
    )

    resnet = cl.load_forecasting_module(
        data_module=dm,
        model=model,
        optim=optimizer,
        sched=lr_scheduler
    )

    wandb.init(
        project='Climate',
        name=f'Climate Learn Resnet, {now}',
        config=cfg,
    )
    logger = WandbLogger()

    trainer = cl.Trainer(
        early_stopping="val/lat_mse:aggregate",
        patience=cfg['patience'],
        accelerator="gpu",
        devices=[cfg['gpu']],
        precision=16,
        max_epochs=cfg['max_epochs'],
        logger=logger,
        callbacks=[LearningRateMonitor(logging_interval='step'), RichProgressBar()],
        val_check_interval=cfg['val_every_n_steps'],
    )
    
    trainer.fit(resnet, datamodule=dm)
    trainer.test(resnet, datamodule=dm, ckpt_path="best")

    
if __name__ == "__main__":
    main()
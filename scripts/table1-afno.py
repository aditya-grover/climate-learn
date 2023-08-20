# Standard library
from argparse import ArgumentParser

# Third party
import climate_learn as cl
from climate_learn.data import IterDataModule
from climate_learn.utils.datetime import Hours
from climate_learn.data.climate_dataset.era5.constants import *
import torch.multiprocessing
from pytorch_lightning.loggers.tensorboard import TensorBoardLogger

import os
os.environ["NCCL_P2P_DISABLE"] = "1"

def main():
    torch.multiprocessing.set_sharing_strategy("file_system")

    parser = ArgumentParser()
    parser.add_argument("--pred_range", type=int)
    parser.add_argument("--root_dir")
    # parser.add_argument("--gpu", type=int)
    args = parser.parse_args()
    
    variables = [
        "land_sea_mask",
        "orography",
        "lattitude",
        # "toa_incident_solar_radiation",
        "2m_temperature",
        "10m_u_component_of_wind",
        "10m_v_component_of_wind",
        "geopotential",
        "u_component_of_wind",
        "v_component_of_wind",
        "temperature",
        "relative_humidity",
        "specific_humidity",
    ]
    in_vars = []
    for var in variables:
        if var in PRESSURE_LEVEL_VARS:
            for level in DEFAULT_PRESSURE_LEVELS:
                in_vars.append(var + "_" + str(level))
        else:
            in_vars.append(var)

    out_variables = [
        "2m_temperature",
        "geopotential_500",
        "temperature_850"
    ]
    out_vars = []
    for var in out_variables:
        if var in PRESSURE_LEVEL_VARS:
            for level in DEFAULT_PRESSURE_LEVELS:
                out_vars.append(var + "_" + str(level))
        else:
            out_vars.append(var)
    
    history = 1
    window = 0
    subsample = Hours(1)
    pred_range = Hours(args.pred_range)
    batch_size = 64
    default_root_dir=f"results_rebuttal/fourcastnet_{args.pred_range}"
    
    dm = IterDataModule(
        "forecasting",
        args.root_dir,
        args.root_dir,
        in_vars,
        out_vars,
        history,
        window,
        pred_range,
        subsample,
        buffer_size=2000,
        batch_size=batch_size,
        num_workers=1
    )
    # dm.setup()
    
    model = cl.models.hub.AFNONet(
        img_size=(32, 64),
        patch_size=(2, 2),
        in_chans=48,
        out_chans=3,
        embed_dim=768,
        depth=12,
        mlp_ratio=4.,
        drop_rate=0.,
        drop_path_rate=0.,
        num_blocks=16,
    )
    optimizer = cl.load_optimizer(
        model, "AdamW", {"lr": 5e-4, "weight_decay": 1e-5, "betas": (0.9, 0.99)}
    )
    lr_scheduler = cl.load_lr_scheduler(
        "linear-warmup-cosine-annealing",
        optimizer,
        {"warmup_epochs": 5, "max_epochs": 50, "warmup_start_lr": 1e-8, "eta_min": 1e-8}
    )
    afno = cl.load_forecasting_module(
        data_module=dm,
        model=model,
        optim=optimizer,
        sched=lr_scheduler
    )
    
    logger = TensorBoardLogger(
        save_dir=f"{default_root_dir}/logs"
    )
    trainer = cl.Trainer(
        early_stopping="val/lat_mse:aggregate",
        patience=5,
        accelerator="gpu",
        devices=[6, 7],
        # devices=[args.gpu],
        precision=16,
        max_epochs=50,
        default_root_dir=default_root_dir,
        logger=logger
    )
    
    trainer.fit(afno, datamodule=dm)
    trainer.test(afno, datamodule=dm, ckpt_path="best")

    
if __name__ == "__main__":
    main()
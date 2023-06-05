# Standard library
from argparse import ArgumentParser

# Third party
import climate_learn as cl
from climate_learn.data import IterDataModule
from climate_learn.utils.datetime import Hours
from climate_learn.data.climate_dataset.era5.constants import *
import torch
import torch.multiprocessing
from pytorch_lightning.loggers.tensorboard import TensorBoardLogger


def main():
    torch.multiprocessing.set_sharing_strategy("file_system")

    parser = ArgumentParser()
    parser.add_argument("pred_range", type=int)
    parser.add_argument("root_dir")
    parser.add_argument("gpu", type=int)
    args = parser.parse_args()
    
    variables = [
        "land_sea_mask",
        "orography",
        "lattitude",
        "toa_incident_solar_radiation",
        "2m_temperature",
        "10m_u_component_of_wind",
        "10m_v_component_of_wind",
        "geopotential",
        "temperature",
        "relative_humidity",
        "specific_humidity",
        "u_component_of_wind",
        "v_component_of_wind"
    ]
    in_vars = []
    for var in variables:
        if var in PRESSURE_LEVEL_VARS:
            for level in DEFAULT_PRESSURE_LEVELS:
                in_vars.append(var + "_" + str(level))
        else:
            in_vars.append(var)

    out_variables = [
        "land_sea_mask",
        "orography",
        "lattitude",
        "toa_incident_solar_radiation",
        "2m_temperature",
        "10m_u_component_of_wind",
        "10m_v_component_of_wind",
        "geopotential",
        "temperature",
        "relative_humidity",
        "specific_humidity",
        "u_component_of_wind",
        "v_component_of_wind"
    ]
    out_vars = []
    for var in out_variables:
        if var in PRESSURE_LEVEL_VARS:
            for level in DEFAULT_PRESSURE_LEVELS:
                out_vars.append(var + "_" + str(level))
        else:
            out_vars.append(var)
    
    history = 3
    subsample = Hours(1)
    window = 6
    pred_range = Hours(args.pred_range)
    batch_size = 128
    default_root_dir=f"results/unet_forecasting_all_vars_{args.pred_range}"
    
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

    model = cl.models.hub.Unet(
        in_channels=49,
        out_channels=49,
        history=history,
        hidden_channels=64,
        dropout=0.1,
        ch_mults=(1, 2, 2),
        is_attn=(False, False, False),
        n_blocks=2,
    )
    optimizer = cl.load_optimizer(
        model, "AdamW", {"lr": 5e-4, "weight_decay": 1e-5, "betas": (0.9, 0.99)}
    )
    lr_scheduler = cl.load_lr_scheduler(
        "linear-warmup-cosine-annealing",
        optimizer,
        {"warmup_epochs": 5, "max_epochs": 50, "warmup_start_lr": 1e-8, "eta_min": 1e-8}
    )
    unet = cl.load_forecasting_module(
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
        devices=[args.gpu],
        precision=16,
        max_epochs=50,
        default_root_dir=default_root_dir,
        logger=logger
    )
    
    trainer.fit(unet, datamodule=dm)

    # ckpt_path = '/home/tungnd/climate-learn/results/unet_forecasting_all_vars_6/checkpoints/epoch_048.ckpt'
    # ckpt = torch.load(ckpt_path)
    # msg = unet.load_state_dict(ckpt['state_dict'])
    # print (msg)

    # for lead_time in [6, 24, 72, 120, 240]:
    #     n_iters = lead_time // pred_range.hours()
    #     unet.set_mode('iter')
    #     unet.set_n_iters(n_iters)

    #     test_logger = TensorBoardLogger(
    #         save_dir=f"{default_root_dir}/logs"
    #     )

    #     test_trainer = cl.Trainer(
    #         early_stopping="val/lat_mse:aggregate",
    #         patience=5,
    #         accelerator="gpu",
    #         devices=[args.gpu],
    #         precision=16,
    #         max_epochs=50,
    #         default_root_dir=default_root_dir,
    #         logger=test_logger
    #     )

    #     test_dm = IterDataModule(
    #         "forecasting",
    #         args.root_dir,
    #         args.root_dir,
    #         in_vars,
    #         out_vars,
    #         history,
    #         window,
    #         Hours(lead_time),
    #         subsample=subsample,
    #         buffer_size=2000,
    #         batch_size=batch_size,
    #         num_workers=1
    #     )

    #     test_trainer.test(unet, datamodule=test_dm)

    
if __name__ == "__main__":
    main()
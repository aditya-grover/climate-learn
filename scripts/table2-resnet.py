# Standard library
from argparse import ArgumentParser

# Third party
import climate_learn as cl
from climate_learn.data import ContinuousIterDataModule
from climate_learn.utils.datetime import Hours
from climate_learn.data.climate_dataset.era5.constants import *
import torch.multiprocessing
from pytorch_lightning.loggers.tensorboard import TensorBoardLogger


def main():
    torch.multiprocessing.set_sharing_strategy("file_system")

    parser = ArgumentParser()
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
    
    history = 3
    window = 6
    min_pred_range = Hours(6)
    max_pred_range = Hours(120)
    hrs_each_step = Hours(1)
    subsample = Hours(1)
    batch_size = 128
    default_root_dir=f"results/resnet_new_forecasting_1_worker_continuous"
    
    dm = ContinuousIterDataModule(
        "forecasting",
        args.root_dir,
        args.root_dir,
        in_vars,
        out_vars,
        history,
        window,
        random_lead_time=True,
        min_pred_range=min_pred_range,
        max_pred_range=max_pred_range,
        hrs_each_step=hrs_each_step,
        subsample=subsample,
        buffer_size=2000,
        batch_size=batch_size,
        num_workers=1
    )
    # dm.setup()
    
    model = cl.models.hub.ResNet(
        in_channels=50,
        out_channels=3,
        history=history,
        hidden_channels=128,
        activation="leaky",
        norm=True,
        dropout=0.1,
        n_blocks=28,
    )
    optimizer = cl.load_optimizer(
        model, "AdamW", {"lr": 5e-4, "weight_decay": 1e-5, "betas": (0.9, 0.99)}
    )
    lr_scheduler = cl.load_lr_scheduler(
        "linear-warmup-cosine-annealing",
        optimizer,
        {"warmup_epochs": 5, "max_epochs": 50, "warmup_start_lr": 1e-8, "eta_min": 1e-8}
    )
    resnet = cl.load_forecasting_module(
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
    
    trainer.fit(resnet, datamodule=dm)

    for lead_time in [6, 24, 72, 120, 240]:
        test_logger = TensorBoardLogger(
            save_dir=f"{default_root_dir}/logs"
        )

        test_trainer = cl.Trainer(
            early_stopping="val/lat_mse:aggregate",
            patience=5,
            accelerator="gpu",
            devices=[args.gpu],
            precision=16,
            max_epochs=50,
            default_root_dir=default_root_dir,
            logger=test_logger
        )

        test_dm = ContinuousIterDataModule(
            "forecasting",
            args.root_dir,
            args.root_dir,
            in_vars,
            out_vars,
            history,
            window,
            random_lead_time=False,
            min_pred_range=Hours(lead_time),
            max_pred_range=Hours(lead_time),
            hrs_each_step=hrs_each_step,
            subsample=subsample,
            buffer_size=2000,
            batch_size=batch_size,
            num_workers=1
        )

        test_trainer.test(resnet, datamodule=test_dm, ckpt_path="best")

    
if __name__ == "__main__":
    main()
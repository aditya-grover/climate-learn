# Standard library
from argparse import ArgumentParser

# Third party
import climate_learn as cl
from climate_learn.data.cmip6_itermodule import CMIP6IterDataModule
from climate_learn.utils.datetime import Hours
from climate_learn.data.climate_dataset.cmip6.constants import *
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
        "air_temperature",
        "geopotential",
        "temperature",
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
        "air_temperature",
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
    pred_range = Hours(args.pred_range)
    subsample = Hours(1)
    batch_size = 128
    default_root_dir=f"results/resnet_forecasting_{args.pred_range}"
    
    dm = CMIP6IterDataModule(
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
    
    model = cl.models.hub.ResNet(
        in_channels=36,
        out_channels=3,
        history=history,
        hidden_channels=128,
        activation="leaky",
        norm=True,
        dropout=0.1,
        n_blocks=28,
    )
    optimizer = cl.load_optimizer(
        model, "AdamW", {"lr": 5e-4, "weight_decay": 1e-5}
    )
    lr_scheduler = cl.load_lr_scheduler(
        "reduce-lr-on-plateau",
        optimizer,
        {"mode": "min", "factor": 0.5, "patience": 0, "threshold": 0.0, "min_lr": 5e-7}
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
        precision="bf16",
        max_epochs=40,
        default_root_dir=default_root_dir,
        logger=logger
    )
    
    trainer.fit(resnet, datamodule=dm)
    trainer.test(resnet, datamodule=dm, ckpt_path="best")

    
if __name__ == "__main__":
    main()
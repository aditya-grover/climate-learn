# Standard library
from argparse import ArgumentParser
import os

# Third party
import climate_learn as cl
from climate_learn.data import IterDataModule
from climate_learn.utils.datetime import Hours
from climate_learn.data.climate_dataset.era5.constants import *
from pytorch_lightning.loggers.tensorboard import TensorBoardLogger


def main():
    out_var_dict = {
        "t2m": "2m_temperature",
        "z500": "geopotential_500",
        "t850": "temperature_850"
    }

    parser = ArgumentParser()
    parser.add_argument("preset")
    parser.add_argument("variable", choices=list(out_var_dict.keys()))
    parser.add_argument("gpu", type=int)    
    parser.add_argument("--checkpoint", default=None)
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
    out_vars = [out_var_dict[args.variable]]
    
    subsample = Hours(1)
    batch_size = 32
    default_root_dir = f"{args.preset}_downscaling_{args.variable}"
    
    dm = IterDataModule(
        "downscaling",
        os.environ["ERA5_5DEG"],
        os.environ["ERA5_2DEG"],
        in_vars,
        out_vars,
        subsample=subsample,
        batch_size=batch_size,
        buffer_size=2000,
        num_workers=4
    )
    dm.setup()
    
    model = cl.load_downscaling_module(
        data_module=dm,
        preset=args.preset
    )
    logger = TensorBoardLogger(save_dir=f"{default_root_dir}/logs")
    if args.checkpoint:
        trainer = cl.Trainer(
            accelerator="gpu",
            devices=[args.gpu],
            logger=logger,
            precision="16",
            summary_depth=1
        )
        model = cl.LitModule.load_from_checkpoint(
            args.checkpoint,
            net=model.net,
            optimizer=model.optimizer,
            lr_scheduler=None,
            train_loss=None,
            val_loss=None,
            test_loss=model.test_loss,
            test_target_tranfsorms=model.test_target_transforms
        )
        trainer.test(model, datamodule=dm)
    else:
        trainer = cl.Trainer(
            early_stopping="val/mse:aggregate",
            patience=5,
            accelerator="gpu",
            devices=[args.gpu],
            max_epochs=50,
            default_root_dir=default_root_dir,
            logger=logger,
            precision="16",
            summary_depth=1
        )
        trainer.fit(model, datamodule=dm)
        trainer.test(model, datamodule=dm, ckpt_path="best")

    
if __name__ == "__main__":
    main()
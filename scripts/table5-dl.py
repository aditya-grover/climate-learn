# Standard library
from argparse import ArgumentParser
import os

# Third party
import climate_learn as cl
from climate_learn.data import IterDataModule
from climate_learn.utils.datetime import Hours
from climate_learn.data.climate_dataset.era5.constants import *


def main():
    parser = ArgumentParser()
    parser.add_argument("preset")
    parser.add_argument("gpu", type=int)
    args = parser.parse_args()
    
    variables = [
        "2m_temperature",
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
    out_vars = [
        "2m_temperature",
        "geopotential_500",
        "temperature_850"
    ]
    
    subsample = Hours(6)
    batch_size = 32
    
    dm = IterDataModule(
        "downscaling",
        os.environ["ERA5_5DEG"],
        os.environ["ERA5_2DEG"],
        in_vars,
        out_vars,
        subsample=subsample,
        batch_size=batch_size,
        num_workers=8
    )
    dm.setup()
    
    model = cl.load_downscaling_module(
        data_module=dm,
        preset=args.preset
    )
    trainer = cl.Trainer(
        early_stopping="val/mse:aggregate",
        patience=5,
        accelerator="gpu",
        devices=[args.gpu],
        max_epochs=64,
        default_root_dir=f"{args.preset}_downscaling",
        precision="bf16",
        summary_depth=1
    )
    trainer.fit(model, dm)
    trainer.test(model, dm)

    
if __name__ == "__main__":
    main()
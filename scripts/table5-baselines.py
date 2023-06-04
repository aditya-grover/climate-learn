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
    parser.add_argument("gpu", type=int)
    args = parser.parse_args()
    
    in_vars = [
        "2m_temperature",
        "geopotential_500",
        "temperature_850",
    ]
    out_vars = [
        "2m_temperature",
        "geopotential_500",
        "temperature_850"
    ]
    
    subsample = Hours(1)
    batch_size = 32
    
    dm = IterDataModule(
        "downscaling",
        os.environ["ERA5_5DEG"],
        os.environ["ERA5_2DEG"],
        in_vars,
        out_vars,
        subsample=subsample,
        batch_size=batch_size,
        num_workers=4
    )
    dm.setup()
    
    nearest = cl.load_downscaling_module(
        data_module=dm,
        preset="nearest-interpolation"
    )
    bilinear = cl.load_downscaling_module(
        data_module=dm,
        preset="bilinear-interpolation"
    )
    trainer = cl.Trainer(accelerator="gpu", devices=[args.gpu])
    trainer.test(nearest, dm)
    trainer.test(bilinear, dm)

    
if __name__ == "__main__":
    main()
# Standard library
from argparse import ArgumentParser

# Third party
import climate_learn as cl
import pytorch_lightning as pl


parser = ArgumentParser()
parser.add_argument("era5_low_res_dir")
parser.add_argument("era5_high_res_dir")
args = parser.parse_args()

# Set up data
in_vars = out_vars = [
    "2m_temperature",
    "geopotential_500",
    "temperature_850",
]
dm = cl.data.IterDataModule(
    "downscaling",
    args.era5_low_res_dir,
    args.era5_high_res_dir,
    in_vars,
    out_vars,
    subsample=1,
    batch_size=32,
    num_workers=4,
)
dm.setup()

# Set up baseline models
nearest = cl.load_downscaling_module(data_module=dm, preset="nearest-interpolation")
bilinear = cl.load_downscaling_module(data_module=dm, preset="bilinear-interpolation")

# Evaluate baselines (no training needed)
trainer = pl.Trainer()
trainer.test(nearest, dm)
trainer.test(bilinear, dm)

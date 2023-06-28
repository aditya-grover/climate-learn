# Standard library
from argparse import ArgumentParser

# Third party
import climate_learn as cl
import pytorch_lightning as pl


parser = ArgumentParser()
parser.add_argument("cmip6_dir")
parser.add_argument("pred_range", type=int, choices=[6, 24, 72, 120, 240])
args = parser.parse_args()

# Set up data
in_vars = out_vars = ["air_temperature", "geopotential_500", "temperature_850"]
dm = cl.data.IterDataModule(
    "direct-forecasting",
    args.cmip6_dir,
    args.cmip6_dir,
    in_vars,
    out_vars,
    src="mpi-esm1-2-hr",
    history=3,
    window=6,
    pred_range=args.pred_range,
    subsample=6,
    batch_size=128,
    num_workers=8,
)
dm.setup()

# Set up baseline models
climatology = cl.load_forecasting_module(data_module=dm, preset="climatology")
persistence = cl.load_forecasting_module(data_module=dm, preset="persistence")

# Evaluate baslines (no training needed)
trainer = pl.Trainer()
trainer.test(climatology, dm)
trainer.test(persistence, dm)

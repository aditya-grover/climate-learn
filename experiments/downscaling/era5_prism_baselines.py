# Standard library
from argparse import ArgumentParser

# Third party
import climate_learn as cl
from climate_learn.transforms import Mask, Denormalize
import pytorch_lightning as pl


parser = ArgumentParser()
parser.add_argument("era5_cropped_dir")
parser.add_argument("prism_processed_dir")
args = parser.parse_args()

# Set up data
dm = cl.data.ERA5toPRISMDataModule(
    args.era5_cropped_dir,
    args.prism_processed_dir,
    batch_size=32,
    num_workers=4,
)
dm.setup()

# Set up baseline models
mask = Mask(dm.get_out_mask())
denorm = Denormalize(dm)
denorm_mask = lambda x: denorm(mask(x))
nearest = cl.load_downscaling_module(
    data_module=dm,
    preset="nearest-interpolation",
    train_target_transform=mask,
    val_target_transform=[denorm_mask, denorm_mask, denorm_mask, mask],
    test_target_transform=[denorm_mask, denorm_mask, denorm_mask],
)
bilinear = cl.load_downscaling_module(
    data_module=dm,
    preset="bilinear-interpolation",
    train_target_transform=mask,
    val_target_transform=[denorm_mask, denorm_mask, denorm_mask, mask],
    test_target_transform=[denorm_mask, denorm_mask, denorm_mask],
)

# Evaluate baselines (no training needed)
trainer = pl.Trainer()
trainer.test(nearest, dm)
trainer.test(bilinear, dm)

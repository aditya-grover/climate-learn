# Standard library
from argparse import ArgumentParser

# Third party
import climate_learn as cl
from climate_learn.data.processing.era5_constants import (
    PRESSURE_LEVEL_VARS,
    DEFAULT_PRESSURE_LEVELS,
)
import pytorch_lightning as pl
from pytorch_lightning.callbacks import (
    EarlyStopping,
    ModelCheckpoint,
    RichModelSummary,
    RichProgressBar,
)
from pytorch_lightning.loggers.tensorboard import TensorBoardLogger


parser = ArgumentParser()
parser.add_argument("era5_low_res_dir")
parser.add_argument("era5_high_res_dir")
parser.add_argument("preset", choices=["resnet", "unet", "vit"])
parser.add_argument(
    "variable", choices=["t2m", "z500", "t850"], help="The variable to predict."
)
parser.add_argument("--summary_depth", type=int, default=1)
parser.add_argument("--max_epochs", type=int, default=50)
parser.add_argument("--patience", type=int, default=5)
parser.add_argument("--gpu", type=int, default=-1)
parser.add_argument("--checkpoint", default=None)
args = parser.parse_args()

# Set up data
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
    "v_component_of_wind",
]
out_var_dict = {
    "t2m": "2m_temperature",
    "z500": "geopotential_500",
    "t850": "temperature_850",
}
in_vars = []
for var in variables:
    if var in PRESSURE_LEVEL_VARS:
        for level in DEFAULT_PRESSURE_LEVELS:
            in_vars.append(var + "_" + str(level))
    else:
        in_vars.append(var)
dm = cl.data.IterDataModule(
    "downscaling",
    args.era5_low_res_dir,
    args.era5_high_res_dir,
    in_vars,
    out_vars=[out_var_dict[args.variable]],
    subsample=1,
    batch_size=32,
    buffer_size=2000,
    num_workers=4,
)
dm.setup()

# Set up deep learning model
model = cl.load_downscaling_module(data_module=dm, architecture=args.preset)

# Setup trainer
pl.seed_everything(0)
default_root_dir = f"{args.preset}_downscaling_{args.variable}"
logger = TensorBoardLogger(save_dir=f"{default_root_dir}/logs")
early_stopping = "val/mse:aggregate"
callbacks = [
    RichProgressBar(),
    RichModelSummary(max_depth=args.summary_depth),
    EarlyStopping(monitor=early_stopping, patience=args.patience),
    ModelCheckpoint(
        dirpath=f"{default_root_dir}/checkpoints",
        monitor=early_stopping,
        filename="epoch_{epoch:03d}",
        auto_insert_metric_name=False,
    ),
]
trainer = pl.Trainer(
    logger=logger,
    callbacks=callbacks,
    default_root_dir=default_root_dir,
    accelerator="gpu" if args.gpu != -1 else None,
    devices=[args.gpu] if args.gpu != -1 else None,
    max_epochs=args.max_epochs,
    strategy="ddp",
    precision="16",
)

# Train and evaluate model from scratch
if args.checkpoint is None:
    trainer.fit(model, datamodule=dm)
    trainer.test(model, datamodule=dm, ckpt_path="best")
# Evaluate saved model checkpoint
else:
    model = cl.LitModule.load_from_checkpoint(
        args.checkpoint,
        net=model.net,
        optimizer=model.optimizer,
        lr_scheduler=None,
        train_loss=None,
        val_loss=None,
        test_loss=model.test_loss,
        test_target_tranfsorms=model.test_target_transforms,
    )
    trainer.test(model, datamodule=dm)

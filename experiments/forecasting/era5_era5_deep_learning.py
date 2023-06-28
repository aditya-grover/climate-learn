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

parser.add_argument("--summary_depth", type=int, default=1)
parser.add_argument("--max_epochs", type=int, default=50)
parser.add_argument("--patience", type=int, default=5)
parser.add_argument("--gpu", type=int, default=-1)
parser.add_argument("--checkpoint", default=None)

subparsers = parser.add_subparsers(
    help="Whether to perform direct, iterative, or continuous forecasting.",
    dest="forecast_type",
)
direct = subparsers.add_parser("direct")
iterative = subparsers.add_parser("iterative")
continuous = subparsers.add_parser("continuous")

direct.add_argument("era5_dir")
direct.add_argument("model", choices=["resnet", "unet", "vit"])
direct.add_argument("pred_range", type=int, choices=[6, 24, 72, 120, 240])

iterative.add_argument("era5_dir")
iterative.add_argument("model", choices=["resnet", "unet", "vit"])
iterative.add_argument("pred_range", type=int, choices=[6, 24, 72, 120, 240])

continuous.add_argument("era5_dir")
continuous.add_argument("model", choices=["resnet", "unet", "vit"])

args = parser.parse_args()

# Set up data
variables = [
    "geopotential",
    "temperature",
    "u_component_of_wind",
    "v_component_of_wind",
    "relative_humidity",
    "specific_humidity",
    "2m_temperature",
    "10m_u_component_of_wind",
    "10m_v_component_of_wind",
    "toa_incident_solar_radiation",
    "land_sea_mask",
    "orography",
    "lattitude",
]
in_vars = []
for var in variables:
    if var in PRESSURE_LEVEL_VARS:
        for level in DEFAULT_PRESSURE_LEVELS:
            in_vars.append(var + "_" + str(level))
    else:
        in_vars.append(var)
if args.forecast_type in ("direct", "continuous"):
    out_variables = ["2m_temperature", "geopotential_500", "temperature_850"]
elif args.forecast_type == "iterative":
    out_variables = variables
out_vars = []
for var in out_variables:
    if var in PRESSURE_LEVEL_VARS:
        for level in DEFAULT_PRESSURE_LEVELS:
            out_vars.append(var + "_" + str(level))
    else:
        out_vars.append(var)
if args.forecast_type in ("direct", "iterative"):
    dm = cl.data.IterDataModule(
        f"{args.forecast_type}-forecasting",
        args.era5_dir,
        args.era5_dir,
        in_vars,
        out_vars,
        src="era5",
        history=3,
        window=6,
        pred_range=args.pred_range,
        subsample=6,
        batch_size=128,
        num_workers=8,
    )
elif args.forecast_type == "continuous":
    dm = cl.data.IterDataModule(
        "continuous-forecasting",
        args.era5_dir,
        args.era5_dir,
        in_vars,
        out_vars,
        src="era5",
        history=3,
        window=6,
        pred_range=1,
        max_pred_range=120,
        random_lead_time=True,
        hrs_each_step=1,
        subsample=6,
        batch_size=128,
        buffer_size=2000,
        num_workers=8,
    )
dm.setup()

# Set up deep learning model
in_channels = 49
if args.forecast_type == "continuous":
    in_channels += 1  # time dimension
if args.forecast_type == "iterative":  # iterative predicts every var
    out_channels = in_channels
else:
    out_channels = 3
if args.model == "resnet":
    model_kwargs = {  # override some of the defaults
        "in_channels": in_channels,
        "out_channels": out_channels,
        "history": 3,
        "n_blocks": 28,
    }
elif args.model == "unet":
    model_kwargs = {  # override some of the defaults
        "in_channels": in_channels,
        "out_channels": out_channels,
        "history": 3,
        "ch_mults": (1, 2, 2),
        "is_attn": (False, False, False),
    }
elif args.model == "vit":
    model_kwargs = {  # override some of the defaults
        "img_size": (32, 64),
        "in_channels": in_channels,
        "out_channels": out_channels,
        "history": 3,
        "patch_size": 2,
        "embed_dim": 128,
        "depth": 8,
        "decoder_depth": 2,
        "learn_pos_emb": True,
        "num_heads": 4,
    }
optim_kwargs = {"lr": 5e-4, "weight_decay": 1e-5, "betas": (0.9, 0.99)}
sched_kwargs = {
    "warmup_epochs": 5,
    "max_epochs": 50,
    "warmup_start_lr": 1e-8,
    "eta_min": 1e-8,
}
model = cl.load_forecasting_module(
    data_module=dm,
    model=args.model,
    model_kwargs=model_kwargs,
    optim="adamw",
    optim_kwargs=optim_kwargs,
    sched="linear-warmup-cosine-annealing",
    sched_kwargs=sched_kwargs,
)

# Setup trainer
pl.seed_everything(0)
default_root_dir = f"{args.model}_{args.forecast_type}_forecasting_{args.pred_range}"
logger = TensorBoardLogger(save_dir=f"{default_root_dir}/logs")
early_stopping = "val/lat_mse:aggregate"
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


# Define testing regime for iterative forecasting
def iterative_testing(model, trainer, args, from_checkpoint=False):
    for lead_time in [6, 24, 72, 120, 240]:
        n_iters = lead_time // args.pred_range
        model.set_mode("iter")
        model.set_n_iters(n_iters)
        test_dm = cl.data.IterDataModule(
            "iterative-forecasting",
            args.era5_dir,
            args.era5_dir,
            in_vars,
            out_vars,
            src="era5",
            history=3,
            window=6,
            pred_range=lead_time,
            subsample=1,
        )
        if from_checkpoint:
            trainer.test(model, datamodule=test_dm)
        else:
            trainer.test(model, datamodule=test_dm, ckpt_path="best")


# Define testing regime for continuous forecasting
def continuous_testing(model, trainer, args, from_checkpoint=False):
    for lead_time in [6, 24, 72, 120, 240]:
        test_dm = cl.data.IterDataModule(
            "continuous-forecasting",
            args.era5_dir,
            args.era5_dir,
            in_vars,
            out_vars,
            src="era5",
            history=3,
            window=6,
            pred_range=lead_time,
            max_pred_range=lead_time,
            random_lead_time=False,
            hrs_each_step=1,
            subsample=1,
            batch_size=128,
            buffer_size=2000,
            num_workers=8,
        )
        if from_checkpoint:
            trainer.test(model, datamodule=test_dm)
        else:
            trainer.test(model, datamodule=test_dm, ckpt_path="best")


# Train and evaluate model from scratch
if args.checkpoint is None:
    trainer.fit(model, datamodule=dm)
    if args.forecast_type == "direct":
        trainer.test(model, datamodule=dm, ckpt_path="best")
    elif args.forecast_type == "iterative":
        iterative_testing(model, trainer, args)
    elif args.forecast_type == "continuous":
        continuous_testing(model, trainer, args)
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
    if args.forecast_type == "direct":
        trainer.test(model, datamodule=dm)
    elif args.forecast_type == "iterative":
        iterative_testing(model, trainer, args, from_checkpoint=True)
    elif args.forecast_type == "continuous":
        continuous_testing(model, trainer, args, from_checkpoint=True)

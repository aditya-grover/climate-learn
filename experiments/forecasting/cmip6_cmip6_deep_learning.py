# Standard library
from argparse import ArgumentParser

# Third party
import climate_learn as cl
from climate_learn.data.processing.cmip6_constants import (
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
parser.add_argument("cmip6_dir")
parser.add_argument("model", choices=["resnet", "unet", "vit"])
parser.add_argument("pred_range", type=int, choices=[6, 24, 72, 120, 240])
parser.add_argument("--summary_depth", type=int, default=1)
parser.add_argument("--max_epochs", type=int, default=50)
parser.add_argument("--patience", type=int, default=5)
parser.add_argument("--gpu", type=int, default=-1)
parser.add_argument("--checkpoint", default=None)
args = parser.parse_args()

# Set up data
variables = [
    "air_temperature",
    "geopotential",
    "temperature",
    "specific_humidity",
    "u_component_of_wind",
    "v_component_of_wind",
]
in_vars = []
for var in variables:
    if var in PRESSURE_LEVEL_VARS:
        for level in DEFAULT_PRESSURE_LEVELS:
            in_vars.append(var + "_" + str(level))
    else:
        in_vars.append(var)
out_variables = ["air_temperature", "geopotential_500", "temperature_850"]
out_vars = []
for var in out_variables:
    if var in PRESSURE_LEVEL_VARS:
        for level in DEFAULT_PRESSURE_LEVELS:
            out_vars.append(var + "_" + str(level))
    else:
        out_vars.append(var)
dm = cl.data.IterDataModule(
    "direct-forecasting",
    args.cmip6_dir,
    args.cmip6_dir,
    in_vars,
    out_vars,
    history=3,
    window=6,
    pred_range=args.pred_range,
    subsample=6,
    buffer_size=2000,
    batch_size=128,
    num_workers=4,
)
dm.setup()

# Set up deep learning model
if args.model == "resnet":
    model_kwargs = {  # override some of the defaults
        "in_channels": 36,
        "out_channels": 3,
        "history": 3,
        "n_blocks": 28,
    }
elif args.model == "unet":
    model_kwargs = {  # override some of the defaults
        "in_channels": 36,
        "out_channels": 3,
        "history": 3,
        "ch_mults": (1, 2, 2),
        "is_attn": (False, False, False),
    }
elif args.model == "vit":
    model_kwargs = {  # override some of the defaults
        "img_size": (32, 64),
        "in_channels": 36,
        "out_channels": 3,
        "history": 3,
        "patch_size": 2,
        "embed_dim": 128,
        "depth": 8,
        "decoder_depth": 2,
        "learn_pos_emb": True,
        "num_heads": 4,
    }
model = cl.load_forecasting_module(
    data_module=dm,
    model=args.model,
    model_kwargs=model_kwargs,
    optim="adamw",
    optim_kwargs={"lr": 5e-4, "weight_decay": 1e-5},
    sched="linear-warmup-cosine-annealing",
    sched_kwargs={"warmup_epochs": 5, "max_epoch": 50},
)

# Setup trainer
pl.seed_everything(0)
default_root_dir = f"{args.model}_forecasting_{args.pred_range}"
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

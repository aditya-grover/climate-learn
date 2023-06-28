# Standard library
from argparse import ArgumentParser

# Third party
import climate_learn as cl
import pytorch_lightning as pl
from pytorch_lightning.callbacks import (
    EarlyStopping,
    ModelCheckpoint,
    RichModelSummary,
    RichProgressBar,
)
from pytorch_lightning.loggers.tensorboard import TensorBoardLogger


parser = ArgumentParser()
parser.add_argument("climatebench_dir")
parser.add_argument("model", choices=["resnet", "unet", "vit"])
parser.add_argument(
    "variable",
    choices=["tas", "diurnal_temperature_range", "pr", "pr90"],
    help="The variable to predict.",
)
parser.add_argument("--summary_depth", type=int, default=1)
parser.add_argument("--max_epochs", type=int, default=50)
parser.add_argument("--patience", type=int, default=10)
parser.add_argument("--gpu", type=int, default=-1)
parser.add_argument("--checkpoint", default=None)
args = parser.parse_args()

# Set up data
variables = ["CO2", "SO2", "CH4", "BC"]
out_variables = args.variable
dm = cl.data.ClimateBenchDataModule(
    args.climatebench_dir,
    variables=variables,
    out_variables=out_variables,
    train_ratio=0.9,
    history=10,
    batch_size=16,
    num_workers=1,
)

# Set up deep learning model
if args.model == "resnet":
    model_kwargs = {  # override some of the defaults
        "in_channels": 4,
        "out_channels": 1,
        "history": 10,
        "n_blocks": 28,
    }
elif args.model == "unet":
    model_kwargs = {  # override some of the defaults
        "in_channels": 4,
        "out_channels": 1,
        "history": 10,
        "ch_mults": (1, 2, 2),
        "is_attn": (False, False, False),
    }
elif args.model == "vit":
    model_kwargs = {  # override some of the defaults
        "img_size": (32, 64),
        "in_channels": 4,
        "out_channels": 1,
        "history": 10,
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
model = cl.load_climatebench_module(
    data_module=dm,
    model=args.model,
    model_kwargs=model_kwargs,
    optim="adamw",
    optim_kwargs=optim_kwargs,
    sched="linear-warmup-cosine-annealing",
    sched_kwargs=sched_kwargs,
)

# Set up trainer
pl.seed_everything(0)
default_root_dir = f"{args.model}_climatebench_{args.variable}"
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
    log_every_n_steps=1,
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
        test_target_transforms=model.test_target_transforms,
    )
    trainer.test(model, datamodule=dm)

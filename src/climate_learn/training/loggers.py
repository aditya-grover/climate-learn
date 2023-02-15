import os
import wandb
from lightning.loggers import WandbLogger as wblogger


def WandbLogger(project="climate_tutorial", name="default", notebook="CCAI Tutorial"):
    os.environ["WANDB_NOTEBOOK_NAME"] = notebook
    wandb.login()
    return wblogger(project=project, name=name)

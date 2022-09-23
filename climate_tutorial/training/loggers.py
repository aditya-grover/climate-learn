import wandb
from pytorch_lightning.loggers import WandbLogger as wblogger

def WandbLogger(project = "climate_tutorial", name = "default"):
    wandb.login()
    return wblogger(project = project, name = name)
import wandb
from pytorch_lightning.loggers import WandbLogger as wblogger

def WandbLogger(project = "climate_tutorial", name = "default"):
    def __init__(self, ):
        wandb.login()
        return wblogger(project = project, name = name)
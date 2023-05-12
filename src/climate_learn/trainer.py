# Standard library
import logging
import sys
from warnings import warn

# Third party
import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import (
    ModelCheckpoint,
    RichModelSummary,
    RichProgressBar,
    EarlyStopping,
)

logging.getLogger("pytorch_lightning").setLevel(logging.ERROR)


class Trainer(pl.Trainer):
    """Wrapper for Lightning's trainer."""
    def __init__(self, early_stopping=None, patience=0, summary_depth=-1, seed=0, **kwargs):
        pl.seed_everything(seed)
        if "accelerator" not in kwargs and torch.cuda.is_available():
            print(
                "GPU detected but 'accelerator' not specified..."
                " setting 'accelerator' = \"gpu\""
            )
            kwargs["accelerator"] = "gpu"
        if "logger" not in kwargs:
            kwargs["logger"] = False
        if "callbacks" not in kwargs:
            checkpoint_callback = ModelCheckpoint(
                save_last=True,
                verbose=False,
                filename="epoch_{epoch:03d}",
                auto_insert_metric_name=False,
            )
            summary_callback = RichModelSummary(max_depth=summary_depth)
            progress_callback = RichProgressBar()
            callbacks = [
                checkpoint_callback,
                summary_callback,
                progress_callback,
            ]
            if early_stopping:
                early_stop_callback = EarlyStopping(
                    monitor=early_stopping,
                    patience=patience,
                    verbose=False
                )
                callbacks.append(early_stop_callback)
        if "strategy" not in kwargs:
            if sys.stdout.isatty():
                warn("In interactive environment: cannot use DDP spawn strategy")
                kwargs["strategy"] = None
            else:
                kwargs["strategy"] = "ddp_spawn"
        self.trainer = pl.Trainer(**kwargs)

    def fit(self, model_module, *args, **kwargs):
        if model_module.optimizer is None:
            raise RuntimeError(
                "Model module has no optimizer - maybe it has no parameters?"
            )
        self.trainer.fit(model_module, *args, **kwargs)

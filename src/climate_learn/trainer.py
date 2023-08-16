# Standard library
import logging
from warnings import warn

# Third party
import pytorch_lightning as pl
from pytorch_lightning.callbacks import (
    EarlyStopping,
    ModelCheckpoint,
    RichModelSummary,
    RichProgressBar,
    LearningRateMonitor
)

logging.getLogger("pytorch_lightning").setLevel(logging.ERROR)


class Trainer(pl.Trainer):
    """Wrapper for Lightning's trainer."""

    def __init__(
        self, early_stopping=None, patience=0, summary_depth=-1, seed=0, **kwargs
    ):
        pl.seed_everything(seed)
        default_root_dir = kwargs["default_root_dir"]
        if "logger" not in kwargs:
            kwargs["logger"] = False
        if "callbacks" not in kwargs:
            checkpoint_callback = ModelCheckpoint(
                dirpath=f"{default_root_dir}/checkpoints",
                monitor=early_stopping,
                mode="min",
                save_top_k=1,
                save_last=True,
                verbose=False,
                filename="epoch_{epoch:03d}",
                auto_insert_metric_name=False,
            )
            summary_callback = RichModelSummary(max_depth=summary_depth)
            progress_callback = RichProgressBar()
            lr_monitor = LearningRateMonitor(logging_interval="step")
            callbacks = [
                checkpoint_callback,
                summary_callback,
                progress_callback,
                lr_monitor
            ]
            if early_stopping:
                if 'min_delta' not in kwargs:
                    min_delta = 0.0
                else:
                    min_delta = kwargs['min_delta']
                    kwargs.pop('min_delta')
                early_stop_callback = EarlyStopping(
                    early_stopping, min_delta, patience
                )
                callbacks.append(early_stop_callback)
            kwargs["callbacks"] = callbacks
        if "strategy" not in kwargs:
            if in_notebook():
                warn("In interactive environment: cannot use DDP spawn strategy")
                kwargs["strategy"] = None
            else:
                kwargs["strategy"] = "ddp"
        self.trainer = pl.Trainer(**kwargs)
        self.trainer.favorite_metric = early_stopping

    def fit(self, model_module, *args, **kwargs):
        if model_module.optimizer is None:
            raise RuntimeError(
                "Model module has no optimizer - maybe it has no parameters?"
            )
        self.trainer.fit(model_module, *args, **kwargs)

    def test(self, model_module, *args, **kwargs):
        self.trainer.test(model_module, *args, **kwargs)


# https://stackoverflow.com/a/22424821
def in_notebook():
    try:
        from IPython import get_ipython

        if "IPKernelApp" not in get_ipython().config:  # pragma: no cover
            return False
    except ImportError:
        return False
    except AttributeError:
        return False
    return True

from pytorch_lightning import Trainer as LitTrainer
from pytorch_lightning import seed_everything
from pytorch_lightning.callbacks import (
    ModelCheckpoint,
    RichModelSummary,
    RichProgressBar,
)

import logging

logging.getLogger("pytorch_lightning").setLevel(logging.ERROR)
logging.getLogger("pytorch_lightning").setLevel(logging.ERROR)


class Trainer:
    def __init__(
        self,
        seed=0,
        accelerator="gpu",
        devices=1,
        precision=16,
        max_epochs=4,
        logger=False,
    ):
        seed_everything(seed)

        checkpoint_callback = ModelCheckpoint(
            save_last=True,
            verbose=False,
            filename="epoch_{epoch:03d}",
            auto_insert_metric_name=False,
        )
        summary_callback = RichModelSummary(max_depth=-1)
        progress_callback = RichProgressBar()

        self.trainer = LitTrainer(
            logger=logger,
            accelerator=accelerator,
            devices=devices,
            precision=precision,
            max_epochs=max_epochs,
            callbacks=[checkpoint_callback, summary_callback, progress_callback],
        )

    def fit(self, model_module, data_module):
        self.trainer.fit(model_module, data_module)

    def test(self, model_module, data_module):
        self.trainer.test(model_module, data_module)

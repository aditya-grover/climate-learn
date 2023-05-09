# Standard library
from typing import Any, Callable, Dict, Iterable, List, Union

# Third party
import torch
import pytorch_lightning as pl


class ForecastLitModule(pl.LightningModule):
    def __init__(
        self,
        net: torch.nn.Module,
        optimizer: Union[torch.optim.Optimizer, Dict[str, torch.optim.Optimizer]],
        train_loss: Union[Callable, List[Callable]],
        val_loss: Union[Callable, List[Callable]],
        test_loss: Union[Callable, List[Callable]]
    ):
        super().__init__()
        self.net = net
        self.optimizer = optimizer
        self.train_loss = train_loss
        self.val_loss = val_loss
        self.test_loss = test_loss
        if not isinstance(self.train_loss, Iterable):
            self.train_loss = [self.train_loss]
        if not isinstance(self.val_loss, Iterable):
            self.val_loss = [self.val_loss]
        if not isinstance(self.test_loss, Iterable):
            self.test_loss = [self.test_loss]

    def forward(self, x):
        return self.net(x)

    def training_step(self, batch: Any, batch_idx: int):
        x, y, _, _ = batch
        yhat = self(x)
        loss_fns = self.train_loss
        loss_dict = {}
        for i, lf in enumerate(loss_fns):
            loss = lf(yhat, y)
            name = getattr(lf, "name", f"loss_{i}")
            loss_dict[f"train/{name}"] = loss
        self.log_dict(
            loss_dict,
            on_step=True,
            on_epoch=False,
            prog_bar=True,
            batch_size=len(batch[0])
        )
        return loss_dict

    def validation_step(self, batch: Any, batch_idx: int):
        self.evaluate(batch, "val")

    def test_step(self, batch: Any, batch_idx: int):
        self.evaluate(batch, "test")
    
    def evaluate(self, batch, stage):
        x, y, _, _ = batch
        yhat = self(x)
        loss_fns = self.val_loss
        loss_dict = {}
        for i, lf in enumerate(loss_fns):
            loss = lf(yhat, y)
            name = getattr(lf, "name", f"loss_{i}")
            loss_dict[f"{stage}/{name}"] = loss
        self.log_dict(
            loss_dict,
            on_step=False,
            on_epoch=True,
            sync_dist=True,
            batch_size=len(batch[0])
        )
        return loss_dict

    def configure_optimizers(self):
        return self.optimizer
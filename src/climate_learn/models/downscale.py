from typing import Any, Callable, Dict, Iterable, List, Union
import torch
import torch.nn.functional as F
import pytorch_lightning as pl


class DownscaleLitModule(pl.LightningModule):
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
        x = self.interpolate_input(x)
        yhat = self(x)
        loss_fns = self.train_loss
        loss_dict = {}
        for i, lf in enumerate(loss_fns):
            loss = lf(yhat, y)
            if hasattr(lf, "name"):
                name = lf.name
            else:
                name = f"loss_{i}"
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
        x = self.interpolate_input(x, y)
        yhat = self(x)
        loss_fns = self.val_loss
        loss_dict = {}
        for i, lf in enumerate(loss_fns):
            loss = lf(yhat, y)
            if hasattr(lf, "name"):
                name = lf.name
            else:
                name = f"loss_{i}"
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
    
    def interpolate_input(x: torch.Tensor, y: torch.Tensor):
        # interpolate input to match output size
        out_h, out_w = y.shape[-2], y.shape[-1]
        x = F.interpolate(x, (out_h, out_w), mode="bilinear")
        return x

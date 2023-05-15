# Standard library
from typing import Any, Callable, Dict, List, Optional

# Third party
import torch
from torch.optim.lr_scheduler import _LRScheduler as LRScheduler
import pytorch_lightning as pl


class LitModule(pl.LightningModule):
    def __init__(
        self,
        net: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        lr_scheduler: LRScheduler,
        train_loss: Callable,
        val_loss: List[Callable],
        test_loss: List[Callable],
        train_target_transform: Optional[Callable] = None,
        val_target_transforms: Optional[List[Callable]] = None,
        test_target_transforms: Optional[List[Callable]] = None,
    ):
        super().__init__()
        self.net = net
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.train_loss = train_loss
        self.val_loss = val_loss
        self.test_loss = test_loss
        self.train_target_transform = train_target_transform
        self.val_target_transforms = val_target_transforms
        self.test_target_transforms = test_target_transforms

    def forward(self, x):
        return self.net(x)

    def training_step(self, batch: Any, batch_idx: int):
        x, y, in_variables, out_variables = batch
        yhat = self(x).to(device=y.device)
        if self.train_target_transform:
            yhat = self.train_target_transform(yhat)
            y = self.train_target_transform(y)
        losses = self.train_loss(yhat, y)
        loss_name = getattr(self.train_loss, "name", "loss")
        loss_dict = {}
        if losses.dim() == 0:  # aggregate loss only
            loss = losses
            loss_dict[loss_name] = loss
        else:  # per channel + aggregate            
            for var_name, loss in zip(out_variables, losses):
                loss_dict[f"{loss_name}:{var_name}"] = loss
            loss = losses[-1]
        self.log_dict(
            loss_dict,
            prog_bar=True,
            on_step=True,
            on_epoch=False,
            batch_size=x.shape[0],
        )
        return loss

    def validation_step(self, batch: Any, batch_idx: int):
        self.evaluate(batch, "val")

    def test_step(self, batch: Any, batch_idx: int):
        self.evaluate(batch, "test")

    def evaluate(self, batch, stage):
        x, y, in_variables, out_variables = batch
        yhat = self(x).to(device=y.device)
        if stage == "val":
            loss_fns = self.val_loss
        elif stage == "test":
            loss_fns = self.test_loss
        else:
            raise RuntimeError("Invalid evaluation stage")
        loss_dict = {}
        for i, lf in enumerate(loss_fns):
            if stage == "val" and self.val_target_transforms is not None:
                yhat_T = self.val_target_transforms[i](yhat)
                y_T = self.val_target_transforms[i](y)
            elif stage == "test" and self.test_target_transforms is not None:
                yhat_T = self.test_target_transforms[i](yhat)
                y_T = self.val_target_transforms[i](y)
            else:
                yhat_T = yhat
                y_T = y
            losses = lf(yhat_T, y_T)
            loss_name = getattr(lf, "name", f"loss_{i}")
            if losses.dim() == 0:  # aggregate loss
                loss_dict[loss_name] = losses
            else:  # per channel + aggregate
                for var_name, loss in zip(out_variables, losses):
                    name = f"{loss_name}:{var_name}"
                    loss_dict[name] = loss
                loss_dict[loss_name] = losses[-1]
        self.log_dict(
            loss_dict,
            on_step=False,
            on_epoch=True,
            sync_dist=True,
            batch_size=len(batch[0]),
        )
        return loss_dict

    def configure_optimizers(self):
        if self.lr_scheduler is None:
            return self.optimizer
        return {"optimizer": self.optimizer, "lr_scheduler": self.lr_scheduler}

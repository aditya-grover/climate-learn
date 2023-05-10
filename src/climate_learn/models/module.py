# Standard library
from typing import Any, Callable, Dict, List, Union

# Third party
import torch
import pytorch_lightning as pl


class LitModule(pl.LightningModule):
    def __init__(
        self,
        net: torch.nn.Module,
        optimizer: Union[torch.optim.Optimizer, Dict[str, torch.optim.Optimizer]],
        train_loss: List[Callable],
        val_loss: List[Callable],
        test_loss: List[Callable]
    ):
        super().__init__()
        self.net = net
        self.optimizer = optimizer
        self.train_loss = train_loss
        self.val_loss = val_loss
        self.test_loss = test_loss

    def forward(self, x):
        return self.net(x)

    def training_step(self, batch: Any, batch_idx: int):
        x, y, in_variables, out_variables = batch
        yhat = self(x)
        loss_fns = self.train_loss
        loss_dict = {}
        for i, lf in enumerate(loss_fns):
            losses = lf(yhat, y)
            loss_name = getattr(lf, "name", f"loss_{i}")
            if losses.dim() == 0:  # aggregate loss
                loss_dict[f"train/{loss_name}"] = losses
            else:  # per channel + aggregate
                for var_name, loss in zip(out_variables, losses):
                    name = f"{loss_name}:{var_name}"
                    loss_dict[f"train/{name}"] = loss
                loss_dict[f"train/{loss_name}"] = losses[-1]
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
        x, y, in_variables, out_variables = batch
        yhat = self(x)
        loss_fns = self.val_loss
        loss_dict = {}
        for i, lf in enumerate(loss_fns):
            losses = lf(yhat, y)
            loss_name = getattr(lf, "name", f"loss_{i}")
            if losses.dim() == 0:  # aggregate loss
                loss_dict[f"{stage}/{loss_name}"] = losses
            else:  # per channel + aggregate
                for var_name, loss in zip(out_variables, losses):                
                    name = f"{loss_name}:{var_name}"
                    loss_dict[f"{stage}/{name}"] = loss
                loss_dict[f"{stage}/{loss_name}"] = losses[-1]
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
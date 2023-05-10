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
        train_loss: Callable,
        val_loss: List[Callable],
        test_loss: List[Callable]
    ):
        super().__init__()
        self.net = net
        self.optimizer = optimizer
        self.train_loss = train_loss
        self.val_loss = val_loss
        self.test_loss = test_loss
        # Index of the train loss which is the optimization objective
        self.optim_objective_idx = 0

    def forward(self, x):
        return self.net(x)

    def training_step(self, batch: Any, batch_idx: int):
        x, y, in_variables, out_variables = batch
        yhat = self(x).to(device=y.device)
        losses = self.train_loss(yhat, y)
        loss_name = getattr(self.train_loss, "name", "loss")
        if losses.dim() == 0:  # aggregate loss only
            loss = losses
        else:  # per channel + aggregate
            loss_dict = {}
            for var_name, loss in zip(out_variables, losses):
                loss_dict[f"{loss_name}:{var_name}"] = loss
            self.log_dict(
                loss_dict,
                prog_bar=True,
                on_step=True,
                on_epoch=False,
                batch_size=x.shape[0]
            )
            loss = losses[-1]
        return loss

    def validation_step(self, batch: Any, batch_idx: int):
        self.evaluate(batch)

    def test_step(self, batch: Any, batch_idx: int):
        self.evaluate(batch,)
    
    def evaluate(self, batch):
        x, y, in_variables, out_variables = batch
        yhat = self(x).to(device=y.device)
        loss_fns = self.val_loss
        loss_dict = {}
        for i, lf in enumerate(loss_fns):
            losses = lf(yhat, y)
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
            batch_size=len(batch[0])
        )
        return loss_dict

    def configure_optimizers(self):
        return self.optimizer
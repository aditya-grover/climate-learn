from typing import Any, Callable, Dict, Iterable, List, Union
import torch
import pytorch_lightning as pl
from torchvision.transforms import transforms

from .utils.lr_scheduler import LinearWarmupCosineAnnealingLR
from ...metrics.metrics import (
    lat_weighted_mse,
    lat_weighted_mse_val,
    lat_weighted_acc,
    lat_weighted_rmse,
)

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
        self.train_loss = train_loss or [lat_weighted_mse]
        self.val_loss = val_loss or [lat_weighted_mse_val, lat_weighted_rmse, lat_weighted_acc]
        self.test_loss = test_loss or [lat_weighted_rmse, lat_weighted_acc]
        if not isinstance(self.train_loss, Iterable):
            self.train_loss = [self.train_loss]
        if not isinstance(self.val_loss, Iterable):
            self.val_loss = [self.val_loss]
        if not isinstance(self.test_loss, Iterable):
            self.test_loss = [self.test_loss]

    def forward(self, x):
        return self.net(x)

    def training_step(self, batch: Any, batch_idx: int):
        x, y, x_var_names, y_var_names = batch
        yhat = self(x)
        loss_fns = self.train_loss
        loss_dict = {}
        for lf in loss_fns:
            loss = lf(
                yhat,
                y,
                x_var_names,
                y_var_names,
                self.lat,
                self.i
            )
            loss_dict[f"train/{lf.name}"] = loss
        self.log_dict(
            loss_dict,
            on_step=True,
            on_epoch=False,
            prog_bar=True,
            batch_size=len(batch[0])
        )
        return loss_dict

    def validation_step(self, batch: Any, batch_idx: int):
        self.evaluate(batch)

    def test_step(self, batch: Any, batch_idx: int):
        self.evaluate(batch)
    
    def evaluate(self, batch):
        x, y, x_var_names, y_var_names = batch
        yhat = self(x)
        loss_fns = self.val_loss
        loss_dict = {}
        for lf in loss_fns:
            loss = lf(
                yhat,
                y,
                x_var_names,
                y_var_names,
                self.lat,
                self.i
            )
            loss_dict[f"val/{lf.name}"] = loss
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

    def set_denormalization(self, mean, std):
        self.denormalization = transforms.Normalize(mean, std)

    def set_lat_lon(self, lat, lon):
        self.lat = lat
        self.lon = lon

    def set_pred_range(self, r):
        self.pred_range = r

    def set_train_climatology(self, clim):
        self.train_clim = clim

    def set_val_climatology(self, clim):
        self.val_clim = clim

    def set_test_climatology(self, clim):
        self.test_clim = clim

    def get_log_postfix(self):
        pred_range = self.pred_range.hours()
        if pred_range < 24:
            log_postfix = f"{pred_range}_hours"
        else:
            days = pred_range / 24
            log_postfix = f"{days:.1f}_days"
        return log_postfix
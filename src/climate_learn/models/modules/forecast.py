from typing import Any, Callable, Iterable
import sys
import numpy as np
import pandas as pd
import torch
from pytorch_lightning import LightningModule
from torchvision.transforms import transforms
from sklearn.linear_model import Ridge

from .utils.lr_scheduler import LinearWarmupCosineAnnealingLR
from .utils.metrics import lat_weighted_acc, lat_weighted_mse, lat_weighted_rmse

OptimizerCallable = Callable[[Iterable], torch.optim.Optimizer]


class ForecastLitModule(LightningModule):
    def __init__(
        self,
        net: torch.nn.Module,
        optimizer: OptimizerCallable = torch.optim.Adam,
        lr: float = 0.001,
        weight_decay: float = 0.005,
        warmup_epochs: int = 5,
        max_epochs: int = 30,
        warmup_start_lr: float = 1e-8,
        eta_min: float = 1e-8,
    ):
        super().__init__()
        self.save_hyperparameters(logger=False, ignore=["net"])
        self.net = net
        self.test_loss = [lat_weighted_rmse, lat_weighted_acc]
        self.lr_baseline = None
        self.train_loss = [lat_weighted_mse]
        self.val_loss = [lat_weighted_rmse]
        self.optim_cls = optimizer

    def forward(self, x):
        with torch.no_grad():
            return self.net.predict(x)

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

    def training_step(self, batch: Any, batch_idx: int):
        x, y, _, out_variables = batch

        loss_dict, _ = self.net.forward(
            x, y, out_variables, metric=self.train_loss, lat=self.lat
        )
        loss_dict = loss_dict[0]
        for var in loss_dict.keys():
            self.log(
                "train/" + var,
                loss_dict[var],
                on_step=True,
                on_epoch=False,
                prog_bar=True,
                batch_size=len(x),
            )
        return loss_dict

    def validation_step(self, batch: Any, batch_idx: int):
        x, y, variables, out_variables = batch
        pred_steps = 1
        pred_range = self.pred_range.hours()

        default_days = [1, 3, 5]
        days_each_step = pred_range / 24
        default_steps = [
            d / days_each_step for d in default_days if d % days_each_step == 0
        ]
        steps = [int(s) for s in default_steps if s <= pred_steps and s > 0]
        days = [int(s * pred_range / 24) for s in steps]

        all_loss_dicts, _ = self.net.rollout(
            x,
            y,
            self.val_clim,
            variables,
            out_variables,
            steps=pred_steps,
            metric=self.val_loss,
            transform=self.denormalization,
            lat=self.lat,
            log_steps=steps,
            log_days=days,
        )
        loss_dict = {}
        for d in all_loss_dicts:
            for k in d.keys():
                loss_dict[k] = d[k]

        for var in loss_dict.keys():
            self.log(
                "val/" + var,
                loss_dict[var],
                on_step=False,
                on_epoch=True,
                prog_bar=False,
                sync_dist=True,
                batch_size=len(x),
            )
        return loss_dict

    def test_step(self, batch: Any, batch_idx: int):
        x, y, variables, out_variables = batch
        pred_steps = 1
        pred_range = self.pred_range.hours()

        default_days = [1, 3, 5]
        days_each_step = pred_range / 24
        default_steps = [
            d / days_each_step for d in default_days if d % days_each_step == 0
        ]
        steps = [int(s) for s in default_steps if s <= pred_steps and s > 0]
        days = [int(s * pred_range / 24) for s in steps]

        all_loss_dicts, _ = self.net.rollout(
            x,
            y,
            self.test_clim,
            variables,
            out_variables,
            steps=pred_steps,
            metric=self.test_loss,
            transform=self.denormalization,
            lat=self.lat,
            log_steps=steps,
            log_days=days,
        )

        loss_dict = {}
        for d in all_loss_dicts:
            for k in d.keys():
                loss_dict[k] = d[k]

        for var in loss_dict.keys():
            self.log(
                "test/" + var,
                loss_dict[var],
                on_step=False,
                on_epoch=True,
                sync_dist=True,
                batch_size=len(x),
            )

        # rmse for climatology baseline
        clim_pred = self.train_clim  # C, H, W
        clim_pred = (
            clim_pred.unsqueeze(0)
            .unsqueeze(0)
            .repeat(y.shape[0], y.shape[1], 1, 1, 1)
            .to(y.device)
        )
        baseline_rmse = lat_weighted_rmse(
            clim_pred,
            y.unsqueeze(1),
            out_variables,
            transform_pred=False,
            transform=self.denormalization,
            lat=self.lat,
            log_steps=steps,
            log_days=days,
        )
        for var in baseline_rmse.keys():
            self.log(
                "test_climatology_baseline/" + var,
                baseline_rmse[var],
                on_step=False,
                on_epoch=True,
                sync_dist=True,
                batch_size=len(x),
            )

        # rmse for persistence baseline
        pers_pred = x  # B, 1, C, H, W
        baseline_rmse = lat_weighted_rmse(
            pers_pred,
            y.unsqueeze(1),
            out_variables,
            transform_pred=True,
            transform=self.denormalization,
            lat=self.lat,
            log_steps=steps,
            log_days=days,
        )
        for var in baseline_rmse.keys():
            self.log(
                "test_persistence_baseline/" + var,
                baseline_rmse[var],
                on_step=False,
                on_epoch=True,
                sync_dist=True,
                batch_size=len(x),
            )

        # rmse for linear regression baseline
        # check if fit_lin_reg_baseline is called by checking whether self.lr_baseline is initialized
        if self.lr_baseline:
            lr_pred = self.lr_baseline.predict(
                x.cpu().reshape((x.shape[0], -1))
            ).reshape(y.shape)
            lr_pred = lr_pred[:, np.newaxis, :, :, :]  # B, 1, C, H, W
            lr_pred = torch.from_numpy(lr_pred).float().to(y.device)
            baseline_rmse = lat_weighted_rmse(
                lr_pred,
                y,
                out_variables,
                transform_pred=True,
                transform=self.denormalization,
                lat=self.lat,
                log_steps=steps,
                log_days=days,
            )
            for var in baseline_rmse.keys():
                self.log(
                    "test_ridge_regression_baseline/" + var,
                    baseline_rmse[var],
                    on_step=False,
                    on_epoch=True,
                    sync_dist=True,
                    batch_size=len(x),
                )

        return loss_dict

    def configure_optimizers(self):
        decay = []
        no_decay = []
        for name, m in self.named_parameters():
            if "pos_embed" in name:
                no_decay.append(m)
            else:
                decay.append(m)

        optimizer = self.optim_cls(
            [
                {
                    "params": decay,
                    "lr": self.hparams.lr,
                    "weight_decay": self.hparams.weight_decay,
                },
                {"params": no_decay, "lr": self.hparams.lr, "weight_decay": 0},
            ]
        )

        lr_scheduler = LinearWarmupCosineAnnealingLR(
            optimizer,
            self.hparams.warmup_epochs,
            self.hparams.max_epochs,
            self.hparams.warmup_start_lr,
            self.hparams.eta_min,
        )

        return {"optimizer": optimizer, "lr_scheduler": lr_scheduler}

    def fit_lin_reg_baseline(self, train_dataset, reg_hparam=0.0):
        X_train = train_dataset.inp_data.reshape(train_dataset.inp_data.shape[0], -1)
        y_train = train_dataset.out_data.reshape(train_dataset.out_data.shape[0], -1)
        self.lr_baseline = Ridge(alpha=reg_hparam)
        self.lr_baseline.fit(X_train, y_train)

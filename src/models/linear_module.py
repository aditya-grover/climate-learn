from typing import Any

import torch
from pytorch_lightning import LightningModule
from torchvision.transforms import transforms

from src.utils.metrics import lat_weighted_acc, lat_weighted_mse, lat_weighted_rmse


class LinearLitModule(LightningModule):
    def __init__(
        self,
        net: torch.nn.Module,
        lr: float = 0.001,
        weight_decay: float = 0.005,
    ) -> None:
        super().__init__()
        super().__init__()
        self.save_hyperparameters(logger=False, ignore=["net"])
        self.net = net

    def forward(self, x):
        return self.net.predict(x)

    def set_denormalization(self, mean, std):
        self.denormalization = transforms.Normalize(mean, std)

    def set_lat_lon(self, lat, lon):
        self.lat = lat
        self.lon = lon

    def set_pred_range(self, r):
        self.pred_range = r

    def training_step(self, batch: Any, batch_idx: int):
        x, y, _, out_variables = batch
        loss_dict, _ = self.net.forward(x, y, out_variables, [lat_weighted_mse], lat=self.lat)
        loss_dict = loss_dict[0]
        for var in loss_dict.keys():
            self.log(
                "train/" + var,
                loss_dict[var],
                on_step=True,
                on_epoch=False,
                prog_bar=True,
            )
        return loss_dict

    def validation_step(self, batch: Any, batch_idx: int):
        x, y, variables, out_variables = batch
        pred_steps = y.shape[1]
        pred_range = self.pred_range

        default_days = [1, 3, 5]
        days_each_step = pred_range / 24
        default_steps = [d / days_each_step for d in default_days if d % days_each_step == 0]
        steps = [int(s) for s in default_steps if s <= pred_steps and s > 0]
        days = [int(s * pred_range / 24) for s in steps]

        all_loss_dicts, _ = self.net.rollout(
            x,
            y,
            variables,
            out_variables,
            pred_steps,
            [lat_weighted_rmse, lat_weighted_acc],
            self.denormalization,
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
            )
        return loss_dict

    def test_step(self, batch: Any, batch_idx: int):
        x, y, variables, out_variables = batch
        pred_steps = y.shape[1]
        pred_range = self.pred_range

        default_days = [1, 3, 5]
        days_each_step = pred_range / 24
        steps = [int(d / days_each_step) for d in default_days]
        steps = [s for s in steps if s <= pred_steps]
        days = [int(s * pred_range / 24) for s in steps]

        all_loss_dicts, _ = self.net.rollout(
            x,
            y,
            variables,
            out_variables,
            pred_steps,
            [lat_weighted_rmse, lat_weighted_acc],
            self.denormalization,
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
            )
        return loss_dict

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.hparams.lr, weight_decay=self.hparams.weight_decay)

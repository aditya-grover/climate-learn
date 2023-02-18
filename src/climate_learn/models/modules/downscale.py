from typing import Any, Callable, Iterable

import torch
from pytorch_lightning import LightningModule
from torchvision.transforms import transforms

from .utils.lr_scheduler import LinearWarmupCosineAnnealingLR
from .utils.metrics import mse, rmse, pearson, mean_bias


def interpolate_input(x: torch.Tensor, y: torch.Tensor):
    # interpolate input to match output size
    out_h, out_w = y.shape[-2], y.shape[-1]
    x = torch.nn.functional.interpolate(x, (out_h, out_w), mode="bilinear")
    return x


OptimizerCallable = Callable[[Iterable], torch.optim.Optimizer]


class DownscaleLitModule(LightningModule):
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
        self.optim_cls = optimizer

    def forward(self, x):
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
        x = interpolate_input(x, y)

        loss_dict, _ = self.net.forward(
            x, y, out_variables, [mse], lat=self.lat, log_postfix=""
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
        x = interpolate_input(x, y)

        all_loss_dicts, _ = self.net.evaluate(
            x,
            y,
            variables,
            out_variables,
            self.denormalization,
            [rmse, pearson, mean_bias],
            self.lat,
            self.val_clim,
            log_postfix="",
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
        x = interpolate_input(x, y)

        all_loss_dicts, _ = self.net.evaluate(
            x,
            y,
            variables,
            out_variables,
            self.denormalization,
            [rmse, pearson, mean_bias],
            self.lat,
            self.test_clim,
            log_postfix="",
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

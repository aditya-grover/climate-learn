# credits: https://github.com/ashleve/lightning-hydra-template/blob/main/src/models/mnist_module.py
from typing import Any, Dict

import torch
from pytorch_lightning import LightningModule
from torchvision.transforms import transforms

from src.utils.lr_scheduler import LinearWarmupCosineAnnealingLR
from src.utils.metrics import (
    lat_weighted_acc,
    lat_weighted_mse,
    lat_weighted_mse_val,
    lat_weighted_rmse,
)


class ViTLitModule(LightningModule):
    def __init__(
        self,
        net: torch.nn.Module,
        pretrained_path: str,
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
        if len(pretrained_path) > 0:
            self.load_mae_weights(pretrained_path)

    def load_mae_weights(self, pretrained_path):
        checkpoint = torch.load(pretrained_path)

        print("Loading pre-trained checkpoint from: %s" % pretrained_path)
        checkpoint_model = checkpoint["state_dict"]
        state_dict = self.state_dict()
        checkpoint_keys = list(checkpoint_model.keys())
        for k in checkpoint_keys:
            if k not in state_dict.keys() or checkpoint_model[k].shape != state_dict[k].shape:
                print(f"Removing key {k} from pretrained checkpoint")
                del checkpoint_model[k]

        # load pre-trained model
        msg = self.load_state_dict(checkpoint_model, strict=False)
        print(msg)

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
        if isinstance(batch, dict):
            loss = 0
            for source_id in batch.keys():
                x, y, variables, out_variables = batch[source_id]
                loss_dict, _ = self.net.forward(x, y, variables, out_variables, [lat_weighted_mse], lat=self.lat)
                loss_dict = loss_dict[0]
                for var in loss_dict.keys():
                    self.log(
                        f"train/{source_id}/" + var,
                        loss_dict[var],
                        on_step=True,
                        on_epoch=False,
                        prog_bar=True,
                    )
                # return loss_dict
                loss += loss_dict["loss"]
            return loss / len(batch.keys())
        else:
            x, y, variables, out_variables = batch
            loss_dict, _ = self.net.forward(x, y, variables, out_variables, [lat_weighted_mse], lat=self.lat)
            loss_dict = loss_dict[0]
            for var in loss_dict.keys():
                self.log(
                    "train/" + var,
                    loss_dict[var],
                    on_step=True,
                    on_epoch=False,
                    prog_bar=True,
                )
            return loss_dict['loss']

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
            [lat_weighted_mse_val, lat_weighted_rmse, lat_weighted_acc],
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

    # def validation_epoch_end(self, outputs: List[Any]):
    #     acc = self.val_acc.compute()  # get val accuracy from current epoch
    #     self.val_acc_best.update(acc)
    #     self.log(
    #         "val/acc_best", self.val_acc_best.compute(), on_epoch=True, prog_bar=True
    #     )

    #     self.val_acc.reset()  # reset val accuracy for next epoch

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
            [lat_weighted_mse_val, lat_weighted_rmse, lat_weighted_acc],
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
        decay = []
        no_decay = []
        for name, m in self.named_parameters():
            if "pos_embed" in name:
                no_decay.append(m)
            else:
                decay.append(m)

        optimizer = torch.optim.AdamW(
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

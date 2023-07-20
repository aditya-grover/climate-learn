# Standard library
from typing import Callable, List, Optional, Tuple, Union

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
        val_target_transforms: Optional[List[Union[Callable, None]]] = None,
        test_target_transforms: Optional[List[Union[Callable, None]]] = None,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.net = net
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.train_loss = train_loss
        self.val_loss = val_loss
        self.test_loss = test_loss
        self.train_target_transform = train_target_transform
        if val_target_transforms is not None:
            if len(val_loss) != len(val_target_transforms):
                raise RuntimeError(
                    "If 'val_target_transforms' is not None, its length must"
                    " match that of 'val_loss'. 'None' can be passed for"
                    " losses which do not require transformation."
                )
        self.val_target_transforms = val_target_transforms
        if test_target_transforms is not None:
            if len(test_loss) != len(test_target_transforms):
                raise RuntimeError(
                    "If 'test_target_transforms' is not None, its length must"
                    " match that of 'test_loss'. 'None' can be passed for "
                    " losses which do not rqeuire transformation."
                )
        self.test_target_transforms = test_target_transforms

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

    def training_step(
        self,
        batch: Tuple[torch.Tensor, torch.Tensor, List[str], List[str]],
        batch_idx: int,
    ) -> torch.Tensor:
        x, y, in_variables, out_variables = batch
        yhat = self(x).to(device=y.device)
        if self.train_target_transform:
            yhat = self.train_target_transform(yhat)
            y = self.train_target_transform(y)
        
        # needed for swin transformers
        if yhat.shape[2] != y.shape[2] or yhat.shape[3] != y.shape[3]:
            yhat = torch.nn.functional.interpolate(yhat, size=(y.shape[2], y.shape[3]))

        losses = self.train_loss(yhat, y)
        loss_name = getattr(self.train_loss, "name", "loss")
        loss_dict = {}
        if losses.dim() == 0:  # aggregate loss only
            loss = losses
            loss_dict[f"{loss_name}:aggregate [train]"] = loss
        else:  # per channel + aggregate
            for var_name, loss in zip(out_variables, losses):
                loss_dict[f"{loss_name}:{var_name}"] = loss
            loss = losses[-1]
            loss_dict[f"{loss_name}:aggregate [train]"] = loss
        optimization_loss = loss
        loss_fns = self.val_loss                #Compute RMSE and other metrics on train set
        transforms = self.val_target_transforms
        for i, lf in enumerate(loss_fns):
            if transforms is not None and transforms[i] is not None:
                yhat_T = transforms[i](yhat)                    #Fixes bug when transforms[i] is None
                y_T = transforms[i](y)

                # needed for swin transformers
                if yhat_T.shape[2] != y_T.shape[2] or yhat_T.shape[3] != y_T.shape[3]:
                    yhat_T = torch.nn.functional.interpolate(yhat_T, size=(y_T.shape[2], y_T.shape[3]))
                losses = lf(yhat_T, y_T)
            else:
                # needed for swin transformers
                if yhat.shape[2] != y.shape[2] or yhat.shape[3] != y.shape[3]:
                    yhat = torch.nn.functional.interpolate(yhat, size=(y.shape[2], y.shape[3]))
                losses = lf(yhat, y)

            loss_name = getattr(lf, "name", f"loss_{i}")
            if losses.dim() == 0:
                loss_dict[f"{loss_name}:agggregate [train]"] = losses
            else:
                for var_name, loss in zip(out_variables, losses):
                    name = f"{loss_name}:{var_name} [train]"
                    loss_dict[name] = loss
                loss_dict[f"{loss_name}:aggregate [train]"] = losses[-1]

        self.log_dict(
            loss_dict,
            prog_bar=True,
            on_step=True,
            on_epoch=True,
            batch_size=x.shape[0],
        )
        return optimization_loss

    def validation_step(
        self,
        batch: Tuple[torch.Tensor, torch.Tensor, List[str], List[str]],
        batch_idx: int,
    ) -> torch.Tensor:
        self.evaluate(batch, "val")

    def test_step(
        self,
        batch: Tuple[torch.Tensor, torch.Tensor, List[str], List[str]],
        batch_idx: int,
    ) -> torch.Tensor:
        self.evaluate(batch, "test")

    def evaluate(
        self, batch: Tuple[torch.Tensor, torch.Tensor, List[str], List[str]], stage: str
    ):
        x, y, in_variables, out_variables = batch
        yhat = self(x).to(device=y.device)
        if stage == "val":
            loss_fns = self.val_loss
            transforms = self.val_target_transforms
        elif stage == "test":
            loss_fns = self.test_loss
            transforms = self.test_target_transforms
        else:
            raise RuntimeError("Invalid evaluation stage")
        loss_dict = {}
        for i, lf in enumerate(loss_fns):
            if transforms is not None and transforms[i] is not None:
                yhat_T = transforms[i](yhat)                            #Fixes bug when #transforms[i] is None
                y_T = transforms[i](y)

                # needed for swin transformers
                if yhat_T.shape[2] != y_T.shape[2] or yhat_T.shape[3] != y_T.shape[3]:
                    yhat_T = torch.nn.functional.interpolate(yhat_T, size=(y_T.shape[2], y_T.shape[3]))
                losses = lf(yhat_T, y_T)
            else:
                # needed for swin transformers
                if yhat.shape[2] != y.shape[2] or yhat.shape[3] != y.shape[3]:
                    yhat = torch.nn.functional.interpolate(yhat, size=(y.shape[2], y.shape[3]))
                losses = lf(yhat, y)

            loss_name = getattr(lf, "name", f"loss_{i}")
            if losses.dim() == 0:  # aggregate loss
                loss_dict[f"{loss_name}:agggregate [{stage}]"] = losses
            else:  # per channel + aggregate
                for var_name, loss in zip(out_variables, losses):
                    name = f"{loss_name}:{var_name} [{stage}]"
                    loss_dict[name] = loss
                loss_dict[f"{loss_name}:aggregate [{stage}]"] = losses[-1]
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

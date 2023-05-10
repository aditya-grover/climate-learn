# Standard library
from typing import Any, Dict
import warnings

# Local application
from .metrics import METRICS_REGISTRY, MetricsMetaInfo
from .models.hub import (
    Climatology,
    Persistence,
    LinearRegression,
    ResNet,
    Interpolation
)
from .models.lr_scheduler import LinearWarmupCosineAnnealingLR

# Third party
import torch
import torch.optim.lr_scheduler as LRSchedulers


def load_preset(task, data_module, preset):
    in_vars, out_vars = _get_data_variables(data_module)
    in_shape, out_shape = _get_data_dims(data_module)
    def raise_not_impl():
        raise NotImplementedError(
            f"{preset} is not an implemented preset for the {task} task. If"
            " you think it should be, please raise an issue at"
            " https://github.com/aditya-grover/climate-learn/issues."
        )

    if task == "forecasting":
        history, in_channels, in_height, in_width = in_shape[1:]
        out_channels, out_height, out_width = out_shape[1:]
        if preset.lower() == "climatology":
            train_climatology = data_module.get_climatology(split="train")
            train_climatology = torch.stack(tuple(train_climatology.values()))
            model = Climatology(train_climatology)
            optimizer = None
        elif preset == "persistence":
            if not set(out_vars).issubset(in_vars):
                raise RuntimeError(
                    "Persistence requires the output variables to be a subset of"
                    " the input variables."
                )
            model = Persistence()
            optimizer = lr_scheduler = None
        elif preset.lower() == "linear-regression":
            in_features = history * in_channels * in_height * in_width
            out_features = out_channels * out_height * out_width
            model = LinearRegression(in_features, out_features)
            optimizer = load_optimizer(model, "SGD", {"lr": 1e-5})
            lr_scheduler = None
        elif preset.lower() == "rasp-theurey-2020":
            model = ResNet(
                in_channels=in_channels,
                out_channels=out_channels,
                history=history,
                hidden_channels=128,
                activation="leaky",
                norm=True,
                dropout=0.1,
                n_blocks=19
            )
            optimizer = load_optimizer(
                model,
                "Adam",
                {"lr": 1e-5, "weight_decay": 1e-5}
            )
            lr_scheduler = None
        else:
            raise_not_impl()
    elif task == "downscaling":
        in_channels, in_height, in_width = in_shape[1:]
        out_channels, out_height, out_width = out_shape[1:]
        if preset.lower() in ("linear_interpolation", "bilinear_interpolation", "nearest_interpolation"):
            interpolation_mode = preset.split("_")[0]
            model = Interpolation(out_height*out_width, interpolation_mode)
            optimizer = lr_scheduler = None
        else:
            raise_not_impl()
    return model, optimizer, lr_scheduler

def load_optimizer(
    net: torch.nn.Module,
    optim: str,
    optim_kwargs: Dict[str, Any] = {}
):
    if len(list(net.parameters())) == 0:
        warnings.warn(
            "Net has no trainable parameters, setting optimizer to `None`"
        )
        optimizer = None
    if optim.lower() == "sgd":
        optimizer = torch.optim.SGD(net.parameters(), **optim_kwargs)
    elif optim.lower() == "adam":
        optimizer = torch.optim.Adam(net.parameters(), **optim_kwargs)
    elif optim.lower() == "adamw":
        optimizer = torch.optim.AdamW(net.parameters(), **optim_kwargs)
    else:
        raise NotImplementedError(
            f"{optim} is not an implemented optimizer. If you think it should"
            " be, please raise an issue at"
            " https://github.com/aditya-grover/climate-learn/issues"
        )
    return optimizer

def load_lr_scheduler(
    sched: str,
    optimizer: torch.optim.Optimizer,
    sched_kwargs: Dict[str, Any] = {}
):
    if optimizer is None:
        warnings.warn(
            "Optimizer is `None`, setting LR scheduler to `None` too"
        )
        lr_scheduler = None
    if sched == "constant":
        lr_scheduler = LRSchedulers.ConstantLR(optimizer, **sched_kwargs)
    elif sched == "linear":
        lr_scheduler = LRSchedulers.LinearLR(optimizer, **sched_kwargs)
    elif sched == "exponential":
        lr_scheduler = LRSchedulers.ExponentialLR(optimizer, **sched_kwargs)
    elif sched == "linear-warmup-cosine-annealing":
        lr_scheduler = LinearWarmupCosineAnnealingLR(optimizer, **sched_kwargs)    
    else:
        raise NotImplementedError(
            f"{sched} is not an implemented learning rate scheduler. If you"
            " think it should be, please raise an issue at"
            " https://github.com/aditya-grover/climate-learn/issues"
        )
    return lr_scheduler

def load_loss(loss_name, metainfo):
    loss_cls = METRICS_REGISTRY.get(loss_name, None)
    if loss_cls is None:
        raise NotImplementedError(
            f"{loss_name} is not an implemented loss. If you think it should be,"
            " please raise an issue at"
            " https://gtihub.com/aditya-grover/climate-learn/issues"
        )
    loss = loss_cls(
        aggregate_only=True,
        metainfo=metainfo
    )
    return loss
        
def get_data_dims(data_module):
    for batch in data_module.train_dataloader():
        x, y, _, _ = batch
        break
    return x.shape, y.shape

def get_data_variables(data_module):
    for batch in data_module.train_dataloader():
        _, _, in_vars, out_vars = batch
        break
    return in_vars, out_vars

def get_climatology(data_module, split):
    clim = data_module.get_climatology(split=split)
    # Hotfix to work with dict style data
    clim = torch.stack(tuple(clim.values()))
    return clim
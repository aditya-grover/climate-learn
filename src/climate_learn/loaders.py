# Standard library
from typing import Any, Callable, Dict, Iterable, Optional, Union
import warnings

# Local application
from .data import IterDataModule, DataModule
from .metrics import (
    Denormalized,
    MetricsMetaInfo,
    LatWeightedMSE,
    LatWeightedRMSE,    
    LatWeightedACC,
    MSE,
    RMSE,
    Pearson,
    MeanBias
)
from .models import LitModule, MODEL_REGISTRY
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
import torch.nn as nn
from torchvision import transforms


def load_forecasting_module(
    data_module: Union[DataModule, IterDataModule],
    preset: Optional[str] = None,
    model: Optional[str] = None,
    model_kwargs: Optional[Dict[str, Any]] = None,
    optim: Optional[str] = None,
    optim_kwargs: Optional[Dict[str, Any]] = None,
    lr_kwargs: Optional[Dict[str, Any]] = None,
    net: Optional[nn.Module] = None,
    optimizer: Optional[Dict[str, torch.optim.Optimizer]] = None,
    train_loss: Optional[Callable] = None,
    val_loss: Optional[Iterable[Callable]] = None,
    test_loss: Optional[Iterable[Callable]] = None
):
    in_vars, out_vars = _get_data_variables(data_module)
    in_shape, out_shape = _get_data_dims(data_module)
    history, in_channels, in_height, in_width = in_shape[1:]
    out_channels, out_height, out_width = out_shape[1:]
    if (preset is None) and (model is None) and (net is None):
        raise RuntimeError("Please specify one of 'preset', 'model', or 'net'")
    if preset and (model or net):
        warnings.warn("Ignoring 'model'/'net' since 'preset' was specified")
    elif model and net:
        warnings.warn("Ignoring 'net' since 'model' was specified")
    elif net and model_kwargs:
        warnings.warn("Ignoring 'model_kwargs' since 'net' was specified")
    if preset:
        if preset == "climatology":
            train_climatology = data_module.get_climatology(split="train")
            train_climatology = torch.stack(tuple(train_climatology.values()))
            net = Climatology(train_climatology)
            optimizer = None
        elif preset == "persistence":
            if not set(out_vars).issubset(in_vars):
                raise RuntimeError(
                    "Persistence requires the output variables to be a subset of"
                    " the input variables."
                )
            net = Persistence()
            optimizer = None
        elif preset == "linear-regression":
            in_features = history * in_channels * in_height * in_width
            out_features = out_channels * out_height * out_width
            net = LinearRegression(in_features, out_features)
            optimizer = _load_optimizer(net, "SGD", {"lr": 1e-5})
        elif preset == "rasp-theurey-2020":
            net = ResNet(
                in_channels=in_channels,
                out_channels=out_channels,
                history=history,
                hidden_channels=128,
                activation="leaky",
                norm=True,
                dropout=0.1,
                n_blocks=19
            )
            optimizer = _load_optimizer(
                net,
                "Adam_CosineAnnealingLR",
                {"lr": 1e-5, "weight_decay": 1e-5},
                {"warmup_epochs": 5, "max_epochs": 100, "warmup_start_lr": 1e-8, "eta_min": 1e-8}
            )
        else:
            raise NotImplementedError(
                f"{preset} is not an implemented preset. If you think it should be,"
                " please raise an issue at"
                " https://github.com/aditya-grover/climate-learn/issues."
            )
    else:
        if net is None:
            net = MODEL_REGISTRY.get(model, None)
            if net is None:
                raise NotImplementedError(
                    f"{model} is not an implemented model. If you think it should be,"
                    " please raise an issue at"
                    " https://github.com/aditya-grover/climate-learn/issues."
                )
            net = net(**model_kwargs)
        optimizer = _load_optimizer(net, optim, optim_kwargs, lr_kwargs, optimizer)
    if train_loss is None:
        train_metainfo = MetricsMetaInfo(
            in_vars,
            out_vars,
            *data_module.get_lat_lon(), 
            _get_climatology(data_module, "train")
        )
        train_loss = LatWeightedMSE(aggregate_only=True, metainfo=train_metainfo)
    if val_loss is None:
        val_metainfo = MetricsMetaInfo(
            in_vars,
            out_vars,
            *data_module.get_lat_lon(),
            _get_climatology(data_module, "val")
        )
        val_loss = nn.ModuleList([
            LatWeightedMSE(metainfo=val_metainfo),
            LatWeightedRMSE(metainfo=val_metainfo),
            Denormalized(
                _get_denorm(data_module),
                LatWeightedACC(metainfo=val_metainfo)
            )
        ])
    if test_loss is None:
        test_metainfo = MetricsMetaInfo(
            in_vars,
            out_vars,
            *data_module.get_lat_lon(),
            _get_climatology(data_module, "test")
        )
        test_loss = nn.ModuleList([
            LatWeightedRMSE(metainfo=test_metainfo),
            Denormalized(
                _get_denorm(data_module),
                LatWeightedACC(metainfo=test_metainfo)
            )
        ])
    model_module = LitModule(net, optimizer, train_loss, val_loss, test_loss)
    return model_module

def load_downscaling_module(
    data_module: Union[DataModule, IterDataModule],
    preset: Optional[str] = None,
    model: Optional[str] = None,
    model_kwargs: Optional[Dict[str, Any]] = None,
    optim: Optional[str] = None,
    optim_kwargs: Optional[Dict[str, Any]] = None,
    lr_kwargs: Optional[Dict[str, Any]] = None,
    net: Optional[nn.Module] = None,
    optimizer: Optional[Dict[str, torch.optim.Optimizer]] = None,
    train_loss: Optional[Callable] = None,
    val_loss: Optional[Iterable[Callable]] = None,
    test_loss: Optional[Iterable[Callable]] = None
):
    in_vars, out_vars = _get_data_variables(data_module)
    if (preset is None) and (model is None) and (net is None):
        raise RuntimeError("Please specify one of 'preset', 'model', or 'net'")
    if preset and (model or net):
        warnings.warn("Ignoring 'model'/'net' since 'preset' was specified")
    elif model and net:
        warnings.warn("Ignoring 'net' since 'model' was specified")
    elif net and model_kwargs:
        warnings.warn("Ignoring 'model_kwargs' since 'net' was specified")
    if preset:
        if preset in ("linear_interpolation", "bilinear_interpolation", "nearest_interpolation"):
            interpolation_mode = model.split("_")[0]            
            train_climatology = data_module.get_climatology(split="train")
            train_climatology = torch.stack(tuple(train_climatology.values()))
            size = train_climatology.shape
            net = Interpolation(size, interpolation_mode)
            optimizer = None
        else:
            raise NotImplementedError(
                f"{preset} is not an implemented preset. If you think it should be,"
                " please raise an issue at"
                " https://github.com/aditya-grover/climate-learn/issues."
            )
    else:
        if net is None:
            net = MODEL_REGISTRY.get(model, None)
            if net is None:
                raise NotImplementedError(
                    f"{model} is not an implemented model. If you think it should be,"
                    " please raise an issue at"
                    " https://github.com/aditya-grover/climate-learn/issues."
                )
            net = net(**model_kwargs)
        optimizer = _load_optimizer(net, optim, optim_kwargs, lr_kwargs, optimizer)
    if train_loss is None:
        train_metainfo = MetricsMetaInfo(
            in_vars,
            out_vars,
            *data_module.get_lat_lon(), 
            _get_climatology(data_module, "train")
        )        
        train_loss = MSE(aggregate_only=True, metainfo=train_metainfo)
    if val_loss is None:
        val_metainfo = MetricsMetaInfo(
            in_vars,
            out_vars,
            *data_module.get_lat_lon(),
            _get_climatology(data_module, "val")
        )
        val_loss = nn.ModuleList([
            RMSE(metainfo=val_metainfo),
            Denormalized(
                _get_denorm(data_module),
                Pearson(metainfo=val_metainfo)
            ),
            Denormalized(
                _get_denorm(data_module),
                MeanBias(metainfo=val_metainfo)
            )
        ])
    if test_loss is None:
        test_metainfo = MetricsMetaInfo(
            in_vars,
            out_vars,
            *data_module.get_lat_lon(),
            _get_climatology(data_module, "test")
        )
        test_loss = nn.ModuleList([
            RMSE(metainfo=test_metainfo),
            Denormalized(
                _get_denorm(data_module),
                Pearson(metainfo=test_metainfo)
            ),
            Denormalized(
                _get_denorm(data_module),
                MeanBias(metainfo=test_metainfo)
            )
        ])
    model_module = LitModule(net, optimizer, train_loss, val_loss, test_loss)
    return model_module

def _load_optimizer(
    net: torch.nn.Module,
    optim: Optional[str] = None,
    optim_kwargs: Optional[Dict[str, Any]] = None,
    lr_kwargs: Optional[Dict[str, Any]] = None,
    optimizer: Optional[Union[torch.optim.Optimizer, Dict[str, torch.optim.Optimizer]]] = None,
):
    if len(list(net.parameters())) == 0:
        warnings.warn("Net has no trainable parameters")
        return None
    if (optim is not None) and (optimizer is not None):
        raise RuntimeError("Please specify one of 'optim' or 'optimizer'")
    if optim and optimizer:
        warnings.warn("Ignoring 'optimizer' since 'optim' was specified")
    if optimizer and optim_kwargs:
        warnings.warn("Ignoring 'optim_kwargs' since 'optimizer' was specified")
    if lr_kwargs and optimizer:
        warnings.warn("Ignoring 'lr_kwargs' since 'optimizer was specified")
    if optim_kwargs is None:
        optim_kwargs = {}
    if optim.startswith("SGD"):
        optimizer = torch.optim.SGD(net.parameters(), **optim_kwargs)
    elif optim.startswith("Adam"):
        optimizer = torch.optim.Adam(net.parameters(), **optim_kwargs)
    elif optim.startswith("AdamW"):
        optimizer = torch.optim.AdamW(net.parameters(), **optim_kwargs)
    elif optim is not None:
        raise NotImplementedError(
            f"{optim} is not an implemented optimizer. If you think it should"
            " be, please raise an issue at"
            " https://github.com/aditya-grover/climate-learn/issues"
        )
    if optim.endswith("CosineAnnealingLR"):
        if lr_kwargs is None:
            lr_kwargs = {}
        lr_scheduler = LinearWarmupCosineAnnealingLR(optimizer, **lr_kwargs)
        optimizer = {
            "optimizer": optimizer,
            "lr_scheduler": lr_scheduler
        }
    return optimizer

def _get_climatology(data_module, split):
    clim = data_module.get_climatology(split=split)
    # Hotfix to work with dict style data
    clim = torch.stack(tuple(clim.values()))
    return clim

def _get_denorm(data_module):
    norm = data_module.get_out_transforms()
    # Hotfix to work with dict style data
    mean_norm = torch.tensor([norm[k].mean for k in norm.keys()])
    std_norm = torch.tensor([norm[k].std for k in norm.keys()])
    std_denorm = 1 / std_norm
    mean_denorm = -mean_norm * std_denorm
    return transforms.Normalize(mean_denorm, std_denorm)

def _get_data_dims(data_module):
    for batch in data_module.train_dataloader():
        x, y, _, _ = batch
        break
    return x.shape, y.shape

def _get_data_variables(data_module):
    for batch in data_module.train_dataloader():
        _, _, in_vars, out_vars = batch
        break
    return in_vars, out_vars
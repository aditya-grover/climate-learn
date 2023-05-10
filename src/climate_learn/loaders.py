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

def _get_denorm(data_module):
    norm = data_module.get_out_transforms()
    # Hotfix to work with dict style data
    mean_norm = torch.tensor([norm[k].mean for k in norm.keys()])
    std_norm = torch.tensor([norm[k].std for k in norm.keys()])
    std_denorm = 1 / std_norm
    mean_denorm = -mean_norm * std_denorm
    return transforms.Normalize(mean_denorm, std_denorm)
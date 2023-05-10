# Standard library
from typing import Any, Callable, Dict, Iterable, Optional, Tuple, Union
import warnings

# Local application
from .data import DataModule, IterDataModule
from .models import MODEL_REGISTRY, LitModule
from .metrics import METRICS_REGISTRY, MetricsMetaInfo
from .loader_utils import *

# Third party
import torch
import torch.nn as nn
import torch.optim.lr_scheduler as LRScheduler


def load_forecasting_module(
    data_module: Union[DataModule, IterDataModule],
    preset: Optional[str] = None,
    model: Optional[Union[str, nn.Module]] = None,
    model_kwargs: Optional[Dict[str, Any]] = None,
    optim: Optional[Union[str, torch.optim.Optimizer]] = None,
    optim_kwargs: Optional[Dict[str, Any]] = None,
    sched: Optional[Union[str, LRScheduler._LRScheduler]] = None,
    sched_kwargs: Optional[Dict[str, Any]] = None,
    train_loss: Union[str, Callable] = "lat_mse",
    val_loss: Union[Iterable[str], Iterable[Callable]] = [
        "denorm_lat_rmse",
        "denorm_lat_acc"
    ],
    test_loss: Union[Iterable[str], Iterable[Callable]] = [
        "denorm_lat_rmse",
        "denorm_lat_acc"
    ]
):
    # Load the model
    if preset is None and model is None:
        raise RuntimeError("Please specify 'preset' or 'model'")
    elif preset:
        print(f"Loading preset: {preset}")
        model, optimizer, lr_scheduler = load_preset(preset)
    elif isinstance(model, str):
        print(f"Loading model: {model}")
        model_cls = MODEL_REGISTRY.get(model, None)
        if model_cls is None:
            raise NotImplementedError(
                f"{model} is not an implemented model. If you think it should be,"
                " please raise an issue at"
                " https://github.com/aditya-grover/climate-learn/issues."
            )
        model = model_cls(**model_kwargs)
    elif isinstance(model, nn.Module):
        print("Using custom network")
    else:
        raise TypeError("'model' must be str or nn.Module")
    # Load the optimizer
    if preset is None and optim is None:
        raise RuntimeError("Please specify 'preset' or 'optim'")
    elif preset:
        print("Using preset optimizer")
    elif isinstance(optim, str):
        print(f"Loading optimizer {optim}")
        optimizer = load_optimizer(model, optim, optim_kwargs)
    elif isinstance(optim, torch.optim.Optimizer):
        print("Using custom optimizer")
    else:
        raise TypeError("'optim' must be str or torch.optim.Optimizer")
    # Load the LR scheduler
    if preset is None and sched is None:
        raise RuntimeError("Please specify 'preset' or 'sched'")
    elif preset:
            print("Using preset learning rate scheduler")            
    elif isinstance(sched, str):
        print(f"Loading learning rate scheduler: {sched}")
        lr_scheduler = load_lr_scheduler(sched, optimizer, sched_kwargs)
    elif isinstance(sched, LRScheduler._LRScheduler):
        print("Using custom learning rate scheduler")
    else:
        raise TypeError("'sched' must be str or torch.optim.lr_scheduler._LRScheduler")
    # Load training loss
    in_vars, out_vars = get_data_variables(data_module)
    lat, lon = data_module.get_lat_lon()    
    if isinstance(train_loss, str):
        print(f"Loading training loss: {train_loss}")
        clim = get_climatology(data_module, "train")
        metainfo = MetricsMetaInfo(in_vars, out_vars, lat, lon, clim)
        train_loss = load_loss(train_loss, metainfo)
    elif isinstance(train_loss, Callable):
        print("Using custom training loss")
    else:
        raise TypeError("'train_loss' must be str or Callable")
    # Load validation loss
    if not isinstance(val_loss, Iterable):
        raise TypeError("'val_loss' must be an iterable")
    elif all([isinstance(vl, str) for vl in val_loss]):
        clim = get_climatology(data_module, "val")
        metainfo = MetricsMetaInfo(in_vars, out_vars, lat, lon, clim)
        val_losses = []
        for vl in val_loss:
            print(f"Loading validation loss: {vl}")
            val_losses.append(load_loss(vl, metainfo))
    elif all([isinstance(vl, Callable) for vl in val_loss]):
        print("Using custom validation losses")
        val_losses = val_loss
    # Load test loss
    if not isinstance(test_loss, Iterable):
        raise TypeError("'test_loss' must be an iterable")
    elif all([isinstance(vl, str) for vl in test_loss]):
        clim = get_climatology(data_module, "val")
        metainfo = MetricsMetaInfo(in_vars, out_vars, lat, lon, clim)
        test_losses = []
        for vl in test_loss:
            print(f"Loading test loss: {vl}")
            test_losses.append(load_loss(vl, metainfo))
    elif all([isinstance(vl, Callable) for vl in test_loss]):
        print("Using custom test losses")
        test_losses = test_loss
    # Instantiate Lightning Module
    model_module = LitModule(
        model,
        optimizer,
        lr_scheduler,
        train_loss,
        val_losses,
        test_losses
    )
    return model_module
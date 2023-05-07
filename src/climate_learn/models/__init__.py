# Standard library
from typing import Any, Callable, Dict, List, Optional, Union

# Local application
from ..data import IterDataModule, DataModule
from ..data.task.args import ForecastingArgs
from .modules import ForecastLitModule, DownscaleLitModule
from .components import (
    Climatology,
    Interpolation,
    LinearRegression,
    Persistence,
    ResNet
)
from ..utils.datetime import Hours

# Third party
import pytorch_lightning as pl
import torch


def load_forecasting_module(
    data_module: pl.LightningDataModule,
    preset: Optional[str] = None,
    model: Optional[str] = None,
    model_kwargs: Optional[Dict[str, Any]] = None,
    optim: Optional[str] = None,
    optim_kwargs: Optional[Dict[str, Any]] = None,
    net: Optional[torch.nn.Module] = None,
    optimizer: Optional[Union[torch.optim.Optimizer, Dict[str, torch.optim.Optimizer]]] = None,
    train_loss: Optional[Union[Callable, List[Callable]]] = None,
    val_loss: Optional[Union[Callable, List[Callable]]] = None,
    test_loss: Optional[Union[Callable, List[Callable]]] = None
):
    in_vars = data_module.hparams.train_dataset_args.task_args.in_vars
    out_vars = data_module.hparams.train_dataset_args.task_args.out_vars
    history = data_module.hparams.train_dataset_args.task_args.history
    _validate_model_kwargs(preset, model, model_kwargs, net)
    if model == "climatology":
        train_climatology = data_module.get_climatology(split="train")
        train_climatology = torch.stack(tuple(train_climatology.values()))
        net = Climatology(train_climatology)
    elif model == "persistence":
        if not set(out_vars).issubset(in_vars):
            raise RuntimeError(
                "Persistence requires the output variables to be a subset of"
                " the input variables."
            )
        net = Persistence()
    elif model == "linear-regression":
        train_climatology = data_module.get_climatology(split="train")
        train_climatology = torch.stack(tuple(train_climatology.values()))
        in_features = train_climatology.shape.flatten() * history
        test_climatology = data_module.get_climatology(split="test")
        test_climatology = torch.stack(tuple(train_climatology.values()))
        out_features = test_climatology.shape.flatten()
        net = LinearRegression(in_features, out_features)
    elif model == "rasp-theurey-2020":
        net = ResNet(
            in_channels=len(in_vars),
            history=3,
            hidden_channels=128,
            activation="leaky",
            out_channels=len(out_vars),
            norm=True,
            dropout=0.1,
            n_blocks=19
        )
    elif model is not None:
        raise NotImplementedError(
            f"{model} is not an implemented model. If you think it should be,"
            " please raise an issue at"
            " https://github.com/aditya-grover/climate-learn/issues."
        )
    optimizer = _load_optimizer(optim, optim_kwargs, optimizer)
    model_module = ForecastLitModule(net, optimizer, train_loss, val_loss, test_loss)
    set_climatology(model_module, data_module)
    return model_module

def load_downscaling_module(
    data_module: pl.LightningDataModule,
    preset: Optional[str] = None,
    model: Optional[str] = None,
    model_kwargs: Optional[Dict[str, Any]] = None,
    optim: Optional[str] = None,
    optim_kwargs: Optional[Dict[str, Any]] = None,
    net: Optional[torch.nn.Module] = None,
    optimizer: Optional[Union[torch.optim.Optimizer, Dict[str, torch.optim.Optimizer]]] = None,
    train_loss: Optional[Union[Callable, List[Callable]]] = None,
    val_loss: Optional[Union[Callable, List[Callable]]] = None,
    test_loss: Optional[Union[Callable, List[Callable]]] = None
):
    _validate_model_kwargs(preset, model, model_kwargs, net)
    if model in ("nearest", "linear", "bilinear"):
        train_climatology = data_module.get_climatology(split="train")
        train_climatology = torch.stack(tuple(train_climatology.values()))
        size = train_climatology.shape
        net = Interpolation(size, model)
    elif model is not None:
        raise NotImplementedError(
            f"{model} is not an implemented model. If you think it should be,"
            " please raise an issue at"
            " https://github.com/aditya-grover/climate-learn/issues."
        )
    optimizer = _load_optimizer(optim, optim_kwargs, optimizer)
    model_module = DownscaleLitModule(net, optimizer, train_loss, val_loss, test_loss)
    set_climatology(model_module, data_module)
    return model_module

def _validate_model_kwargs(
    preset: Optional[str] = None,
    model: Optional[str] = None,
    model_kwargs: Optional[Dict[str, Any]] = None,
    net: Optional[torch.nn.Module] = None
):
    if (preset is None) and (model is None) and (net is None):
        raise RuntimeError("Please specify one of 'preset', 'model', or 'net'")
    if preset:
        if model is not None:
            raise RuntimeWarning("Ignoring 'model' since 'preset' was specified")
        model = preset
    if model and net:
        raise RuntimeWarning("Ignoring 'net' since one of 'preset' or 'model' was specified")
    if net and model_kwargs:
        raise RuntimeWarning("Ignoring 'model_kwargs' since 'net' was specified")
    return

def _load_optimizer(
    optim: Optional[str] = None,
    optim_kwargs: Optional[Dict[str, Any]] = None,
    optimizer: Optional[Union[torch.optim.Optimizer, Dict[str, torch.optim.Optimizer]]] = None,
):
    if (optim is not None) and (optimizer is not None):
        raise RuntimeError("Please specify one of 'optim' or 'optimizer'")
    if optim and optimizer:
        raise RuntimeWarning("Ignoring 'optimizer' since 'optim' was specified")
    if optimizer and optim_kwargs:
        raise RuntimeWarning("Ignoring 'optim_kwargs' since 'optimizer' was specified")
    if optim == "SGD":
        optimizer = None
    elif optim == "Adam":
        optimizer = None
    elif optim == "AdamW":
        optimizer = None
    elif optim is not None:
        raise NotImplementedError(
            f"{optim} is not an implemented optimizer. If you think it should"
            " be, please raise an issue at"
            " https://github.com/aditya-grover/climate-learn/issues"
        )
    return optimizer

def set_climatology(model_module, data_module):
    normalization = data_module.get_out_transforms()
    ## Hotfix for the models to work with dict style data
    mean_norm = torch.tensor([normalization[k].mean for k in normalization.keys()])
    std_norm = torch.tensor([normalization[k].std for k in normalization.keys()])
    mean_denorm, std_denorm = -mean_norm / std_norm, 1 / std_norm
    model_module.set_denormalization(mean_denorm, std_denorm)
    model_module.set_lat_lon(*data_module.get_lat_lon())
    if isinstance(data_module, IterDataModule):
        model_module.set_pred_range(data_module.hparams.pred_range)
    elif isinstance(data_module, DataModule):
        if isinstance(
            data_module.hparams.train_dataset_args.task_args,
            ForecastingArgs,
        ):
            model_module.set_pred_range(
                Hours(data_module.hparams.train_dataset_args.task_args.pred_range)
            )
        else:
            model_module.set_pred_range(Hours(1))

    train_climatology = data_module.get_climatology(split="train")
    ## Hotfix for the models to work with dict style data
    train_climatology = torch.stack(tuple(train_climatology.values()))
    model_module.set_train_climatology(train_climatology)
    val_climatology = data_module.get_climatology(split="val")
    ## Hotfix for the models to work with dict style data
    val_climatology = torch.stack(tuple(val_climatology.values()))
    model_module.set_val_climatology(val_climatology)
    test_climatology = data_module.get_climatology(split="test")
    ## Hotfix for the models to work with dict style data
    test_climatology = torch.stack(tuple(test_climatology.values()))
    model_module.set_test_climatology(test_climatology)


def fit_lin_reg_baseline(model_module, data_module, reg_hparam=1.0):
    model_module.fit_lin_reg_baseline(data_module.train_dataset, reg_hparam)

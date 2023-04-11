from .components import *
from .modules import *
from climate_learn.data import IterDataModule, DataModule
from climate_learn.data.task.args import ForecastingArgs
from climate_learn.utils.datetime import Hours
import torch


def load_model(name, task, model_kwargs, optim_kwargs):
    if name == "vit":
        model_cls = VisionTransformer
    elif name == "resnet":
        model_cls = ResNet
    elif name == "unet":
        model_cls = Unet

    model = model_cls(**model_kwargs)

    if task == "forecasting":
        module = ForecastLitModule(model, **optim_kwargs)
    elif task == "downscaling":
        module = DownscaleLitModule(model, **optim_kwargs)
    else:
        raise NotImplementedError("Only support foreacasting and downscaling")

    return module


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

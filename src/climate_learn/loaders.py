# Standard library
from typing import Any, Callable, Dict, Iterable, Optional, Tuple, Union
from functools import partial
import warnings

# Local application
from .data import DataModule, IterDataModule
from .models import LitModule, MODEL_REGISTRY
from .models.hub import (
    Climatology,
    Interpolation,
    LinearRegression,
    Persistence,
    ResNet,
    ViTPretrained,
    ViTPretrainedClimaXEmb,
    ViTPretrainedLevelEmb,
    VisionTransformer,
    TimeSformerPretrained,
    SwinPretrainedSegmentation,
    Mask2Former
)
from .models.lr_scheduler import LinearWarmupCosineAnnealingLR
from .transforms import TRANSFORMS_REGISTRY
from .metrics import MetricsMetaInfo, METRICS_REGISTRY

# Third party
import torch
import torch.nn as nn
import transformers
from torch.optim.lr_scheduler import _LRScheduler as LRScheduler


def load_model_module(
    task: str,
    data_module: Union[DataModule, IterDataModule],
    preset: Optional[str] = None,
    model: Optional[Union[str, nn.Module]] = None,
    model_kwargs: Optional[Dict[str, Any]] = None,
    optim: Optional[Union[str, torch.optim.Optimizer]] = None,
    optim_kwargs: Optional[Dict[str, Any]] = None,
    sched: Optional[Union[str, LRScheduler]] = None,
    sched_kwargs: Optional[Dict[str, Any]] = None,
    train_loss: Optional[Union[str, Callable]] = None,
    val_loss: Optional[Iterable[Union[str, Callable]]] = None,
    test_loss: Optional[Iterable[Union[str, Callable]]] = None,
    train_target_transform: Optional[Union[str, Callable]] = None,
    val_target_transform: Optional[Iterable[Union[str, Callable]]] = None,
    test_target_transform: Optional[Iterable[Union[str, Callable]]] = None,
    cfg: Optional[Dict[str, Any]] = None,
):
    # Temporary fix, per this discussion:
    # https://github.com/aditya-grover/climate-learn/pull/100#discussion_r1192812343
    lat, lon = data_module.get_lat_lon()
    if lat is None and lon is None:
        raise RuntimeError("Data module has not been set up yet.")
    # Load the model
    if preset is None and model is None:
        raise RuntimeError("Please specify 'preset' or 'model'")
    elif preset:
        print(f"Loading preset: {preset}")
        model, optimizer, lr_scheduler = load_preset(task, data_module, preset, cfg)
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
        optimizer = optim
        print("Using custom optimizer")
    else:
        raise TypeError("'optim' must be str or torch.optim.Optimizer")
    # Load the LR scheduler, if specified
    if preset:
        print("Using preset learning rate scheduler")
    elif sched is None:
        lr_scheduler = None
    elif isinstance(sched, str):
        print(f"Loading learning rate scheduler: {sched}")
        lr_scheduler = load_lr_scheduler(sched, optimizer, sched_kwargs)
    elif isinstance(sched, LRScheduler) or isinstance(sched, torch.optim.lr_scheduler.ReduceLROnPlateau):
        lr_scheduler = sched
        print("Using custom learning rate scheduler")
    else:
        raise TypeError("'sched' must be str, None, or torch.optim.lr_scheduler._LRScheduler")
    # Load training loss
    in_vars, out_vars = get_data_variables(data_module)
    lat, lon = data_module.get_lat_lon()
    if isinstance(train_loss, str):
        print(f"Loading training loss: {train_loss}")
        clim = get_climatology(data_module, "train")
        metainfo = MetricsMetaInfo(in_vars, out_vars, lat, lon, clim)
        train_loss = load_loss(train_loss, True, metainfo)
    elif isinstance(train_loss, Callable):
        print("Using custom training loss")
    else:
        raise TypeError("'train_loss' must be str or Callable")
    # Load validation loss
    if not isinstance(val_loss, Iterable):
        raise TypeError("'val_loss' must be an iterable")
    val_losses = []
    for vl in val_loss:
        if isinstance(vl, str):
            clim = get_climatology(data_module, "val")
            metainfo = MetricsMetaInfo(in_vars, out_vars, lat, lon, clim)
            print(f"Loading validation loss: {vl}")
            val_losses.append(load_loss(vl, False, metainfo))
        elif isinstance(vl, Callable):
            print("Using custom validation loss")
            val_losses.append(vl)
        else:
            raise TypeError("each 'val_loss' must be str or Callable")
    # Load test loss
    if not isinstance(test_loss, Iterable):
        raise TypeError("'test_loss' must be an iterable")
    test_losses = []
    for tl in test_loss:
        if isinstance(tl, str):
            clim = get_climatology(data_module, "test")
            metainfo = MetricsMetaInfo(in_vars, out_vars, lat, lon, clim)
            print(f"Loading test loss: {tl}")
            test_losses.append(load_loss(tl, False, metainfo))
        elif isinstance(tl, Callable):
            print("Using custom testing loss")
            test_losses.append(tl)
        else:
            raise TypeError("each 'test_loss' must be str or Callable")
    # Load training transform
    if isinstance(train_target_transform, str):
        print(f"Loading training transform: {train_target_transform}")
        train_transform = load_transform(train_target_transform, data_module)
    elif isinstance(train_target_transform, Callable):
        print("Using custom training transform")
        train_transform = train_target_transform
    elif train_target_transform is None:
        train_transform = train_target_transform
    else:
        raise TypeError("'train_target_transform' must be str, callable, or None")
    # Load validation transform
    val_transforms = []
    if isinstance(val_target_transform, Iterable):
        for vt in val_target_transform:
            if isinstance(vt, str):
                print(f"Loading validation transform: {vt}")
                val_transforms.append(load_transform(vt, data_module))
            elif isinstance(vt, Callable):
                print("Using custom validation transform")
                val_transforms.append(vt)
            else:
                raise TypeError("each 'val_transform' must be str or Callable")
    elif val_target_transform is None:
        val_transforms = val_target_transform
    else:
        raise TypeError(
            "'val_target_transform' must be an iterable of strings/callables,"
            " or None"
        )
    # Load test transform
    test_transforms = []
    if isinstance(test_target_transform, Iterable):
        for tt in test_target_transform:
            if isinstance(tt, str):
                print(f"Loading test transform: {tt}")
                test_transforms.append(load_transform(tt, data_module))
            elif isinstance(tt, Callable):
                print("Using custom test transform")
                test_transforms.append(tt)
            else:
                raise TypeError("each 'test_transform' must be str or Callable")
    elif test_target_transform is None:
        test_transforms = test_target_transform
    else:
        raise TypeError(
            "'test_target_transform' must be an iterable of strings/callables,"
            " or None"
        )
    # Instantiate Lightning Module
    model_module = LitModule(
        model,
        optimizer,
        lr_scheduler,
        train_loss,
        val_losses,
        test_losses,
        train_transform,
        val_transforms,
        test_transforms,
    )
    return model_module


load_forecasting_module = partial(
    load_model_module,
    task="forecasting",
    train_loss="lat_mse",
    val_loss=["lat_mse", "lat_rmse", "lat_acc"],
    test_loss=["lat_rmse", "lat_acc"],
    train_target_transform=None,
    val_target_transform=[nn.Identity(), "denormalize", "denormalize"], # first metric used for early stopping
    test_target_transform=["denormalize", "denormalize"],
)

load_climatebench_module = partial(
    load_model_module,
    task="forecasting",
    train_loss="mse",
    val_loss=["mse"],
    test_loss=["lat_nrmses", "lat_nrmseg", "lat_nrmse"],
    train_target_transform=None,
    val_target_transform=[nn.Identity()],
    test_target_transform=[nn.Identity(), nn.Identity(), nn.Identity()],
)

load_downscaling_module = partial(
    load_model_module,
    task="downscaling",
    train_loss="mse",
    val_loss=["rmse", "pearson", "mean_bias"],
    test_loss=["rmse", "pearson", "mean_bias"],
    train_target_transform=None,
    val_target_transform=["denormalize", "denormalize"],
    test_target_transform=["denormalize", "denormalize"],
)


def load_preset(task, data_module, preset, cfg=None):
    in_vars, out_vars = get_data_variables(data_module)
    in_shape, out_shape = get_data_dims(data_module)

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
            norm = data_module.get_out_transforms()
            mean_norm = torch.tensor([norm[k].mean for k in norm.keys()])
            std_norm = torch.tensor([norm[k].std for k in norm.keys()])
            clim = get_climatology(data_module, "train")
            model = Climatology(clim, mean_norm, std_norm)
            optimizer = lr_scheduler = None
        elif preset == "persistence":
            if not set(out_vars).issubset(in_vars):
                raise RuntimeError(
                    "Persistence requires the output variables to be a subset of"
                    " the input variables."
                )
            channels = [in_vars.index(o) for o in out_vars]
            model = Persistence(channels)
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
                n_blocks=19,
            )
            optimizer = load_optimizer(
                model, "Adam", {"lr": 1e-5, "weight_decay": 1e-5}
            )
            lr_scheduler = None
        elif preset.lower() == 'vit_cl':
            model = VisionTransformer(
                img_size=(in_height, in_width),
                in_channels=in_channels,
                out_channels=out_channels,
                history=history,
                patch_size=cfg['patch_size'],
                drop_path=cfg['drop_path'],
                drop_rate=cfg['drop_rate'],
                learn_pos_emb=cfg['learn_pos_emb'],
                embed_dim=cfg['embed_dim'],
                depth=cfg['depth'],
                decoder_depth=cfg['decoder_depth'],
                num_heads=cfg['num_heads'],
                mlp_ratio=cfg['mlp_ratio'],
            )
            optimizer = load_optimizer(
                    model, "AdamW", {"lr": cfg['lr'], "weight_decay": cfg['weight_decay'], "betas": cfg['betas']}
            )
            lr_scheduler = load_lr_scheduler(
                "linear-warmup-cosine-annealing",
                optimizer,
                {"warmup_epochs": cfg['warmup_epochs'], "max_epochs": cfg['num_epochs'], "warmup_start_lr": cfg['warmup_start_lr'], "eta_min": cfg['eta_min']}
            ) 
            # lr_scheduler = None
        elif preset.lower() == 'vit_pretrained':
            model = ViTPretrained(
                in_img_size = cfg['in_img_size'],
                out_img_size = (in_height, in_width),
                in_channels = in_channels,
                out_channels = out_channels,
                history=history,
                learn_pos_emb=cfg['learn_pos_emb'],
                patch_size = cfg['patch_size'],
                embed_dim = cfg['embed_dim'],
                decoder_depth=cfg['decoder_depth'],
                use_pretrained_weights=cfg['use_pretrained_weights'],
                use_pretrained_embeddings=cfg['use_pretrained_embeddings'],
                use_n_blocks=cfg['use_n_blocks'],
                freeze_backbone=cfg['freeze_backbone'],
                freeze_embeddings=cfg['freeze_embeddings'],
                resize_img=cfg['resize_img'],
                pretrained_model=cfg['pretrained_model'],
                mlp_embed_depth=cfg['mlp_embed_depth']
            )
            optimizer = load_optimizer(
                    model, "AdamW", {"lr": cfg['lr'], "weight_decay": cfg['weight_decay'], "betas": cfg['betas']}
            )
            lr_scheduler = load_lr_scheduler(
                "linear-warmup-cosine-annealing",
                optimizer,
                {"warmup_epochs": cfg['warmup_epochs'], "max_epochs": cfg['num_epochs'], "warmup_start_lr": cfg['warmup_start_lr'], "eta_min": cfg['eta_min']}
            )
        elif preset.lower() == 'vit_pretrained_climax_emb':
            model = ViTPretrainedClimaXEmb(
                in_img_size = cfg['in_img_size'],
                out_img_size = (in_height, in_width),
                in_channels = in_channels,
                out_channels = out_channels,
                history=history,
                learn_pos_emb=cfg['learn_pos_emb'],
                patch_size = cfg['patch_size'],
                embed_dim = cfg['embed_dim'],
                decoder_depth=cfg['decoder_depth'],
                use_pretrained_weights=cfg['use_pretrained_weights'],
                use_pretrained_climax=cfg['use_pretrained_climax'],
                use_n_blocks=cfg['use_n_blocks'],
                freeze_backbone=cfg['freeze_backbone'],
                freeze_embeddings=cfg['freeze_embeddings'],
                resize_img=cfg['resize_img'],
                pretrained_model=cfg['pretrained_model'],
                mlp_embed_depth=cfg['mlp_embed_depth']
            )
            optimizer = load_optimizer(
                    model, "AdamW", {"lr": cfg['lr'], "weight_decay": cfg['weight_decay'], "betas": cfg['betas']}
            )
            lr_scheduler = load_lr_scheduler(
                "linear-warmup-cosine-annealing",
                optimizer,
                {"warmup_epochs": cfg['warmup_epochs'], "max_epochs": cfg['max_epochs'], "warmup_start_lr": cfg['warmup_start_lr'], "eta_min": cfg['eta_min']}
            )
        elif preset.lower() == 'vit_pretrained_level_emb':
            model = ViTPretrainedLevelEmb(
                in_img_size = cfg['in_img_size'],
                out_img_size = (in_height, in_width),
                in_channels = in_channels,
                out_channels = out_channels,
                history=history,
                learn_pos_emb=cfg['learn_pos_emb'],
                patch_size = cfg['patch_size'],
                embed_dim = cfg['embed_dim'],
                decoder_depth=cfg['decoder_depth'],
                use_pretrained_weights=cfg['use_pretrained_weights'],
                use_n_blocks=cfg['use_n_blocks'],
                freeze_backbone=cfg['freeze_backbone'],
                freeze_embeddings=cfg['freeze_embeddings'],
                resize_img=cfg['resize_img'],
                pretrained_model=cfg['pretrained_model'],
                mlp_embed_depth=cfg['mlp_embed_depth']
            )
            optimizer = load_optimizer(
                    model, "AdamW", {"lr": cfg['lr'], "weight_decay": cfg['weight_decay'], "betas": cfg['betas']}
            )
            lr_scheduler = load_lr_scheduler(
                "linear-warmup-cosine-annealing",
                optimizer,
                {"warmup_epochs": cfg['warmup_epochs'], "max_epochs": cfg['max_epochs'], "warmup_start_lr": cfg['warmup_start_lr'], "eta_min": cfg['eta_min']}
            )
        elif preset.lower() == 'timesformer_pretrained':
            model = TimeSformerPretrained(
                in_img_size = cfg['in_img_size'],
                in_channels = in_channels,
                out_channels = out_channels,
                history=history,
                pretrained_model=cfg['pretrained_model'],
                ckpt_path=cfg['ckpt_path'],
                # learn_pos_emb=cfg['learn_pos_emb'],
                patch_size = cfg['patch_size'],
                decoder_depth=cfg['decoder_depth'],
                use_pretrained_weights=cfg['use_pretrained_weights'],
                use_pretrained_embeddings=cfg['use_pretrained_embeddings'],
                use_n_blocks=cfg['use_n_blocks'],
                freeze_backbone=cfg['freeze_backbone'],
                freeze_embeddings=cfg['freeze_embeddings'],
                mlp_embed_depth=cfg['mlp_embed_depth']
            )
            optimizer = load_optimizer(
                    model, "AdamW", {"lr": cfg['lr'], "weight_decay": cfg['weight_decay'], "betas": cfg['betas']}
            )
            lr_scheduler = load_lr_scheduler(
                "linear-warmup-cosine-annealing",
                optimizer,
                {"warmup_epochs": cfg['warmup_epochs'], "max_epochs": cfg['num_epochs'], "warmup_start_lr": cfg['warmup_start_lr'], "eta_min": cfg['eta_min']}
            )
        elif preset.lower() == 'swin_pretrained_segmentation':
            model = SwinPretrainedSegmentation(
                in_img_size=cfg['in_img_size'],
                input_channels=in_channels,
                out_channels=out_channels,
                embed_type=cfg['embed_type'],
                mlp_embed_depth=cfg['mlp_embed_depth'],
                decoder_depth=cfg['decoder_depth'],
                ckpt_path=cfg['ckpt_path'],
                freeze_backbone=cfg['freeze_backbone'],
                freeze_embeddings=cfg['freeze_embeddings'],
                # Backbone
                patch_size=cfg['patch_size'],
                embed_dim=cfg['embed_dim'],
                depths=cfg['depths'],
                num_heads=cfg['num_heads'],
                window_size=cfg['window_size'],
                mlp_ratio=cfg['mlp_ratio'],
                qkv_bias=cfg['qkv_bias'],
                qk_scale=cfg['qk_scale'],
                drop_rate=cfg['drop_rate'],
                attn_drop_rate=cfg['attn_drop_rate'],
                drop_path_rate=cfg['drop_path_rate'],
                ape=cfg['ape'],
                patch_norm=cfg['patch_norm'],
                out_indices=cfg['out_indices'],
                use_checkpoint=cfg['use_checkpoint'],

                # UPerHead
                in_channels=cfg['in_channels'],
                in_index=cfg['in_index'],
                pool_scales=cfg['pool_scales'],
                channels=cfg['channels'],
                dropout_ratio=cfg['dropout_ratio'],
                num_classes=cfg['num_classes'],
                align_corners=cfg['align_corners'],

                # FCN Head
                fcn_in_channels=cfg['fcn_in_channels'],
                fcn_in_index=cfg['fcn_in_index'],
                fcn_channels=cfg['fcn_channels'],
                num_convs=cfg['num_convs'],
                concat_input=cfg['concat_input'],
                fcn_dropout_ratio=cfg['fcn_dropout_ratio'],
                fcn_num_classes=cfg['fcn_num_classes'],
                fcn_align_corners=cfg['fcn_align_corners'],
            )
            
            optimizer = load_optimizer(
                    model, "AdamW", {"lr": cfg['lr'], "weight_decay": cfg['weight_decay'], "betas": cfg['betas']}
            )
            lr_scheduler = load_lr_scheduler(
                "linear-warmup-cosine-annealing",
                optimizer,
                {"warmup_epochs": cfg['warmup_epochs'], "max_epochs": cfg['num_epochs'], "warmup_start_lr": cfg['warmup_start_lr'], "eta_min": cfg['eta_min']}
            )
        elif preset.lower() == 'mask2former':
            model = Mask2Former(
                in_img_size = cfg['in_img_size'],
                in_channels = in_channels,
                out_channels = out_channels,
                embed_type=cfg['embed_type'], # normal or climax
                mlp_embed_depth=cfg['mlp_embed_depth'],
                decoder_depth=cfg['decoder_depth'],
                freeze_backbone=cfg['freeze_backbone'],
                freeze_embeddings=cfg['freeze_embeddings'],
                use_pretrained_weights=cfg['use_pretrained_weights'],
                patch_size=cfg['patch_size'], 
                embed_dim=cfg['embed_dim'],
                out_embed_dim=cfg['out_embed_dim'], 
                embed_norm=cfg['embed_norm'],
            )
            optimizer = load_optimizer(
                    model, "AdamW", {"lr": cfg['lr'], "weight_decay": cfg['weight_decay'], "betas": cfg['betas']}
            )
            lr_scheduler = load_lr_scheduler(
                "linear-warmup-cosine-annealing",
                optimizer,
                {"warmup_epochs": cfg['warmup_epochs'], "max_epochs": cfg['num_epochs'], "warmup_start_lr": cfg['warmup_start_lr'], "eta_min": cfg['eta_min']}
            )
        # elif preset.lower() == 'cli-vit':
        #     model = ClimaX(
        #         default_vars=cfg['in_variables'] + cfg['constants'],
        #         out_vars=cfg['out_variables'],
        #         img_size = [in_height, in_width],
        #         use_pretrained_weights=cfg['use_pretrained_weights'],
        #         freeze_backbone=cfg['freeze_backbone'],
        #     )
        #     optimizer = load_optimizer(
        #         model, "Adam", {"lr": cfg['lr'], "weight_decay": cfg['weight_decay'], "betas": cfg['betas']}
        #     )
        #     lr_scheduler = None
        else:
            raise_not_impl()
    elif task == "downscaling":
        in_channels, in_height, in_width = in_shape[1:]
        out_channels, out_height, out_width = out_shape[1:]
        if preset.lower() in (
            "bilinear-interpolation",
            "nearest-interpolation",
        ):
            if set(out_vars) != set(in_vars):
                raise RuntimeError(
                    "Interpolation requires the output variables to match the"
                    " input variables."
                )
            interpolation_mode = preset.split("-")[0]
            model = Interpolation(out_height * out_width, interpolation_mode)
            optimizer = lr_scheduler = None
        else:
            raise_not_impl()
    return model, optimizer, lr_scheduler


def load_optimizer(net: torch.nn.Module, optim: str, optim_kwargs: Dict[str, Any] = {}):
    if len(list(net.parameters())) == 0:
        warnings.warn("Net has no trainable parameters, setting optimizer to `None`")
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
    sched: str, optimizer: torch.optim.Optimizer, sched_kwargs: Dict[str, Any] = {}
):
    if optimizer is None:
        warnings.warn("Optimizer is `None`, setting LR scheduler to `None` too")
        lr_scheduler = None
    if sched == "constant":
        lr_scheduler = torch.optim.lr_scheduler.ConstantLR(optimizer, **sched_kwargs)
    elif sched == "linear":
        lr_scheduler = torch.optim.lr_scheduler.LinearLR(optimizer, **sched_kwargs)
    elif sched == "exponential":
        lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, **sched_kwargs)
    elif sched == "linear-warmup-cosine-annealing":
        lr_scheduler = LinearWarmupCosineAnnealingLR(optimizer, **sched_kwargs)
    elif sched == 'reduce-lr-on-plateau':
        lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, **sched_kwargs)
    else:
        raise NotImplementedError(
            f"{sched} is not an implemented learning rate scheduler. If you"
            " think it should be, please raise an issue at"
            " https://github.com/aditya-grover/climate-learn/issues"
        )
    return lr_scheduler


def load_loss(loss_name, aggregate_only, metainfo):
    loss_cls = METRICS_REGISTRY.get(loss_name, None)
    if loss_cls is None:
        raise NotImplementedError(
            f"{loss_name} is not an implemented loss. If you think it should be,"
            " please raise an issue at"
            " https://gtihub.com/aditya-grover/climate-learn/issues"
        )
    loss = loss_cls(aggregate_only=aggregate_only, metainfo=metainfo)
    return loss


def load_transform(transform_name, data_module):
    transform_cls = TRANSFORMS_REGISTRY.get(transform_name, None)
    if transform_cls is None:
        raise NotImplementedError(
            f"{transform_name} is not an implemented transform. If you think"
            " it should be, please raise an issue at"
            " https://github.com/aditya-grover/climate-learn/issues"
        )
    transform = transform_cls(data_module)
    return transform

def get_data_dims(data_module):
    return data_module.get_data_dims()

def get_data_variables(data_module):
    return data_module.get_data_variables()

# def get_data_dims(data_module):
#     for batch in data_module.train_dataloader():
#         x, y, _, _ = batch
#         break
#     return x.shape, y.shape

# def get_data_variables(data_module):
#     in_vars = data_module.train_dataset.task.in_vars
#     out_vars = data_module.train_dataset.task.out_vars
#     return in_vars, out_vars

def get_climatology(data_module, split):
    clim = data_module.get_climatology(split=split)
    if clim is None:
        raise RuntimeError("Climatology has not yet been set.")
    # Hotfix to work with dict style data
    clim = torch.stack(tuple(clim.values()))    
    return clim

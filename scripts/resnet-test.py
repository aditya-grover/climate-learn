# Standard library
from argparse import ArgumentParser

# Third party
import climate_learn as cl
from ..src.climate_learn.data.cmip6_itermodule import IterDataModule
from climate_learn.utils.datetime import Hours
from ..src.climate_learn.data.climate_dataset.cmip6.constants import *
import torch.multiprocessing


def main():
    torch.multiprocessing.set_sharing_strategy("file_system")

    parser = ArgumentParser()
    parser.add_argument("pred_range", type=int)
    parser.add_argument("root_dir")
    parser.add_argument("gpu", type=int)
    parser.add_argument("checkpoint")
    args = parser.parse_args()

    variables = [
        "air_temperature",
        "geopotential",
        "temperature",
        "specific_humidity",
        "u_component_of_wind",
        "v_component_of_wind",
    ]
    in_vars = []
    for var in variables:
        if var in PRESSURE_LEVEL_VARS:
            for level in DEFAULT_PRESSURE_LEVELS:
                in_vars.append(var + "_" + str(level))
        else:
            in_vars.append(var)

    out_variables = ["air_temperature", "geopotential_500", "temperature_850"]
    out_vars = []
    for var in out_variables:
        if var in PRESSURE_LEVEL_VARS:
            for level in DEFAULT_PRESSURE_LEVELS:
                out_vars.append(var + "_" + str(level))
        else:
            out_vars.append(var)

    history = 3
    subsample = Hours(1)
    window = 6
    pred_range = Hours(args.pred_range)
    batch_size = 32

    dm = IterDataModule(
        "forecasting",
        args.root_dir,
        args.root_dir,
        in_vars,
        out_vars,
        history,
        window,
        pred_range,
        subsample,
        batch_size=batch_size,
        num_workers=8,
    )
    dm.setup()
    in_vars, out_vars = dm.get_data_variables()
    lat, lon = dm.get_lat_lon()
    clim = cl.get_climatology(dm, "test")
    metainfo = cl.MetricsMetaInfo(in_vars, out_vars, lat, lon, clim)

    resnet = cl.models.hub.ResNet(
        in_channels=36,
        out_channels=3,
        history=history,
        hidden_channels=128,
        activation="leaky",
        norm=True,
        dropout=0.1,
        n_blocks=19,
    )
    optimizer = cl.load_optimizer(resnet, "Adam", {"lr": 1e-4, "weight_decay": 1e-5})
    lr_scheduler = cl.load_lr_scheduler(
        "linear-warmup-cosine-annealing",
        optimizer,
        {"warmup_epochs": 1000, "max_epochs": 64},
    )
    test_losses = [cl.load_loss(ln, False, metainfo) for ln in ["lat_rmse", "lat_acc"]]
    test_transforms = [
        cl.load_transform(tn, dm) for tn in ["denormalize", "denormalize"]
    ]
    model = cl.LitModule.load_from_checkpoint(
        args.checkpoint,
        net=resnet,
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
        train_loss=None,
        val_loss=None,
        test_loss=test_losses,
        test_target_transforms=test_transforms,
    )
    trainer = cl.Trainer(
        accelerator="gpu",
        devices=[args.gpu],
    )
    trainer.test(model, dm)


if __name__ == "__main__":
    main()

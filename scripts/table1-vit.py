# Standard library
from argparse import ArgumentParser

# Third party
import climate_learn as cl
from climate_learn.data.cmip6_itermodule import CMIP6IterDataModule
from climate_learn.utils.datetime import Hours
from climate_learn.data.climate_dataset.cmip6.constants import *
import torch.multiprocessing


def main():
    torch.multiprocessing.set_sharing_strategy("file_system")

    parser = ArgumentParser()
    parser.add_argument("pred_range", type=int)
    parser.add_argument("root_dir")
    parser.add_argument("gpu", type=int)
    args = parser.parse_args()
    
    variables = [
        "air_temperature",
        "geopotential",
        "temperature",
        "specific_humidity",
        "u_component_of_wind",
        "v_component_of_wind"
    ]
    in_vars = []
    for var in variables:
        if var in PRESSURE_LEVEL_VARS:
            for level in DEFAULT_PRESSURE_LEVELS:
                in_vars.append(var + "_" + str(level))
        else:
            in_vars.append(var)

    out_variables = [
        "air_temperature",
        "geopotential_500",
        "temperature_850"
    ]
    out_vars = []
    for var in out_variables:
        if var in PRESSURE_LEVEL_VARS:
            for level in DEFAULT_PRESSURE_LEVELS:
                out_vars.append(var + "_" + str(level))
        else:
            out_vars.append(var)
    
    history = 3
    subsample = Hours(6)
    window = 6
    pred_range = Hours(args.pred_range)
    batch_size = 32
    
    dm = CMIP6IterDataModule(
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
        num_workers=8
    )
    dm.setup()
    
    vit_kwargs = {
        "img_size": (32,64),
        "in_channels": 36,
        "out_channels": 3,
        "history": 3,
        "patch_size": 2,
        "embed_dim": 256,
        "depth": 8,
        "decoder_depth": 2,
        "num_heads": 16,
        "mlp_ratio": 4
    }
    vit = cl.load_forecasting_module(
        data_module=dm,
        model="vit",
        model_kwargs=vit_kwargs,
        optim="adamw",
        optim_kwargs={"lr": 1e-5},
        sched="linear-warmup-cosine-annealing",
        sched_kwargs={"warmup_epochs": 1000, "max_epochs": 64}
    )
    trainer = cl.Trainer(
        early_stopping="lat_rmse:aggregate",
        patience=5,
        accelerator="gpu",
        devices=[args.gpu],
        max_epochs=64,
        default_root_dir=f"vit_forecasting_{args.pred_range}"
    )
    
    trainer.fit(vit, dm)
    trainer.test(vit, dm)

    
if __name__ == "__main__":
    main()
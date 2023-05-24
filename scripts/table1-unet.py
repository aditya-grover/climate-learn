# Standard library
from argparse import ArgumentParser

# Third party
import climate_learn as cl
from climate_learn.data import IterDataModule
from climate_learn.utils.datetime import Hours
from climate_learn.data.climate_dataset.era5.constants import *
import torch.multiprocessing


def main():
    torch.multiprocessing.set_sharing_strategy("file_system")

    parser = ArgumentParser()
    parser.add_argument("pred_range", type=int)
    parser.add_argument("root_dir")
    parser.add_argument("gpu", type=int)
    args = parser.parse_args()
    
    variables = [
        "2m_temperature",
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
        "2m_temperature",
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
        num_workers=8
    )
    dm.setup()
    
    unet_kwargs = {
        "in_channels": 36,
        "out_channels": 3,
        "history": 3,
        "ch_mults": [1,1,2],
        "n_blocks": 4
    }
    unet = cl.load_forecasting_module(
        data_module=dm,
        model="unet",
        model_kwargs=unet_kwargs,
        optim="adamw",
        optim_kwargs={"lr": 1e-5}
    )
    trainer = cl.Trainer(
        early_stopping="lat_rmse:aggregate",
        patience=5,
        accelerator="gpu",
        devices=[args.gpu]
    )
    
    trainer.fit(unet, dm)
    trainer.test(unet, dm)

    
if __name__ == "__main__":
    main()
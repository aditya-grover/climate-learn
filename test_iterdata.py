# Standard library
from argparse import ArgumentParser

# Third party
import climate_learn as cl
from climate_learn.data import IterDataModule
from climate_learn.utils.datetime import Hours
from climate_learn.data.climate_dataset.era5.constants import *

import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')

def main():
    parser = ArgumentParser()
    parser.add_argument("pred_range", type=int)
    parser.add_argument("gpu", type=int)
    args = parser.parse_args()
    
    inp_root_dir = "/data0/datasets/weatherbench/data/weatherbench/era5/5.625deg_npz"
    out_root_dir = "/data0/datasets/weatherbench/data/weatherbench/era5/5.625deg_npz"
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
    
    

    # Data module
    dm = IterDataModule(
        task="forecasting",
        inp_root_dir = inp_root_dir,
        out_root_dir = out_root_dir,
        in_vars = in_vars,
        out_vars = out_vars,
        history = history,
        window = window,
        pred_range = pred_range,
        subsample = subsample,
        batch_size=batch_size,
        num_workers=0,
    )

    dm.setup()
    
    # Fit baselines
    climatology = cl.load_forecasting_module(
        data_module=dm, preset="climatology"
    )
    persistence = cl.load_forecasting_module(
        data_module=dm, preset="persistence"
    )
    linreg = cl.load_forecasting_module(
        data_module=dm, preset="linear-regression"
    )
    trainer = cl.Trainer(
        early_stopping="lat_rmse:aggregate",
        patience=5,
        accelerator="gpu",
        devices=[args.gpu],
        strategy="ddp_spawn",
        max_epochs=5
    )
    # trainer.fit(linreg, dm)
    
    # Evaluate baselines
    trainer.test(climatology, dm)
    trainer.test(persistence, dm)
    trainer.test(linreg, dm)

if __name__ == '__main__':
    main()
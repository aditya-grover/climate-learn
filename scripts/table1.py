# Standard library
from argparse import ArgumentParser

# Third party
import climate_learn as cl
from climate_learn.data import DataModule
from climate_learn.data.climate_dataset.args import ERA5Args
from climate_learn.data.dataset.args import ShardDatasetArgs
from climate_learn.data.task.args import ForecastingArgs


def main():
    parser = ArgumentParser()
    parser.add_argument("pred_range", type=int)
    parser.add_argument("gpu", type=int)
    args = parser.parse_args()
    
    root = "/data0/datasets/weatherbench/data/weatherbench/era5/5.625deg"
    dataset = "era5"
    variables = [
        "2m_temperature",
        # "geopotential",
        # "temperature",
        # "specific_humidity",
        # "u_component_of_wind",
        # "v_component_of_wind"
    ]
    in_vars = [f"{dataset}:{var}" for var in variables]
    out_vars = [f"{dataset}:{var}" for var in variables]
    total_years = range(1979, 2018)
    train_years = range(1979, 2015)
    val_years = range(2015, 2017)
    test_years = range(2017, 2019)
    history = 3
    subsample = 6
    pred_range = args.pred_range
    
    # General arguments
    data_args = ERA5Args(root, variables, total_years, name=dataset)
    forecasting_args = ForecastingArgs(
        in_vars,
        out_vars,
        history=history,
        pred_range=pred_range,
        subsample=subsample
    )
    shard_dataset_args = ShardDatasetArgs(data_args, forecasting_args, 5)

    # Split-specific arguments
    train_dataset_args = shard_dataset_args.create_copy(
        {"climate_dataset_args": {"years": train_years}}
    )
    val_dataset_args = shard_dataset_args.create_copy(
        {"climate_dataset_args": {"years": val_years}, "n_chunks": 1}
    )
    test_dataset_args = shard_dataset_args.create_copy(
        {"climate_dataset_args": {"years": test_years}, "n_chunks": 1}
    )

    # Data module
    dm = DataModule(
        train_dataset_args,
        val_dataset_args,
        test_dataset_args,
        batch_size=32,
        num_workers=8
    )
    
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
        max_epochs=5
    )
    trainer.fit(linreg, dm)
    
    # Evaluate baselines
    trainer.test(climatology, dm)
    trainer.test(persistence, dm)
    trainer.test(linreg, dm)

if __name__ == '__main__':
    main()
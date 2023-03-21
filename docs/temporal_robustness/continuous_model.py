import os
# export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/username/anaconda3/lib/
os.environ['LD_LIBRARY_PATH']
from climate_learn.data import download
from climate_learn.utils.data import load_dataset, view
from climate_learn.utils.datetime import Year, Days, Hours
from climate_learn.data import DataModule
from climate_learn.models import set_climatology
from climate_learn.models import load_model
from climate_learn.data import IterDataModule
from climate_learn.training import Trainer, WandbLogger
from climate_learn.models import load_model
import torch

torch.cuda.empty_cache()

pred_range = Hours(6)

forecast_data_module = IterDataModule(
    task="forecasting",
    inp_root_dir="/data0/datasets/weatherbench/data/weatherbench/era5/5.625deg_npz",
    out_root_dir="/data0/datasets/weatherbench/data/weatherbench/era5/5.625deg_npz",
    in_vars = ['2m_temperature',  'geopotential_500', 'temperature_850', 'geopotential_50', 'geopotential_250', 'geopotential_600', 'geopotential_700', 'geopotential_850', 'geopotential_925',  'u_component_of_wind_50', 'u_component_of_wind_250', 'u_component_of_wind_500', 'u_component_of_wind_600', 'u_component_of_wind_700', 'u_component_of_wind_850', 'u_component_of_wind_925', 'v_component_of_wind_50', 'v_component_of_wind_250', 'v_component_of_wind_500', 'v_component_of_wind_600', 'v_component_of_wind_700', 'v_component_of_wind_850', 'v_component_of_wind_925', 'temperature_50', 'temperature_250', 'temperature_500', 'temperature_600', 'temperature_700', 'temperature_925', 'specific_humidity_50', 'specific_humidity_250', 'specific_humidity_500', 'specific_humidity_600', 'specific_humidity_700', 'specific_humidity_850', 'specific_humidity_925', 'time'],
    out_vars=['2m_temperature', 'geopotential_500', 'temperature_850'],
    pred_range=pred_range,
    subsample=Hours(6),
    batch_size=32,
    num_workers=16,
    min_cont_time=6,
    max_cont_time=120
)

num_steps = 1

forecast_model_kwargs = {
    "in_channels": len(forecast_data_module.hparams.in_vars),
    "out_channels": len(forecast_data_module.hparams.out_vars),
    "n_blocks": 19,
    "num_steps": num_steps
}

forecast_optim_kwargs = {
    "lr": 1e-4,
    "weight_decay": 1e-5,
    "warmup_epochs": 1,
    "max_epochs": 64
}
forecast_model_module = load_model(
    name="resnet",
    task="forecasting",
    model_kwargs=forecast_model_kwargs,
    optim_kwargs=forecast_optim_kwargs
)

set_climatology(forecast_model_module, forecast_data_module)

project_name = "Continuous_model_5.6_64epochs_37vars"

forecast_trainer = Trainer(
    seed=0,
    accelerator="gpu",
    devices=[7],
    precision=32,
    max_epochs=64,
    logger = WandbLogger(project = project_name, name = "era"),
    early_stopping=False,
    task = "forecasting"
)

forecast_trainer.fit(forecast_model_module, forecast_data_module)

forecast_trainer.test(forecast_model_module, forecast_data_module)

# forecast_trainer.test(forecast_model_module, forecast_data_module, "path")
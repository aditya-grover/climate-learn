import os
# export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/srikeerthibolli/anaconda3/lib/
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
    inp_root_dir="/data0/datasets/weatherbench/data/weatherbench/era5/2.8125deg_npz",
    out_root_dir="/data0/datasets/weatherbench/data/weatherbench/era5/2.8125deg_npz",
    in_vars = ['2m_temperature',  'geopotential_500', 'temperature_850', 'geopotential_50', 'geopotential_250', 'geopotential_600', 'geopotential_700', 'geopotential_850', 'geopotential_925',  'u_component_of_wind_50', 'u_component_of_wind_250', 'u_component_of_wind_500', 'u_component_of_wind_600', 'u_component_of_wind_700', 'u_component_of_wind_850', 'u_component_of_wind_925', 'v_component_of_wind_50', 'v_component_of_wind_250', 'v_component_of_wind_500', 'v_component_of_wind_600', 'v_component_of_wind_700', 'v_component_of_wind_850', 'v_component_of_wind_925', 'temperature_50', 'temperature_250', 'temperature_500', 'temperature_600', 'temperature_700', 'temperature_925', 'specific_humidity_50', 'specific_humidity_250', 'specific_humidity_500', 'specific_humidity_600', 'specific_humidity_700', 'specific_humidity_850', 'specific_humidity_925'],
    out_vars = ['2m_temperature',  'geopotential_500', 'temperature_850', 'geopotential_50', 'geopotential_250', 'geopotential_600', 'geopotential_700', 'geopotential_850', 'geopotential_925',  'u_component_of_wind_50', 'u_component_of_wind_250', 'u_component_of_wind_500', 'u_component_of_wind_600', 'u_component_of_wind_700', 'u_component_of_wind_850', 'u_component_of_wind_925', 'v_component_of_wind_50', 'v_component_of_wind_250', 'v_component_of_wind_500', 'v_component_of_wind_600', 'v_component_of_wind_700', 'v_component_of_wind_850', 'v_component_of_wind_925', 'temperature_50', 'temperature_250', 'temperature_500', 'temperature_600', 'temperature_700', 'temperature_925', 'specific_humidity_50', 'specific_humidity_250', 'specific_humidity_500', 'specific_humidity_600', 'specific_humidity_700', 'specific_humidity_850', 'specific_humidity_925'],
    pred_range=pred_range,
    subsample=Hours(6),
    batch_size=32,
    num_workers=16,
)

isIterative = True
num_steps = 1
short_range = 6

if isIterative :
    if isinstance(pred_range, Hours):
        num_steps = int(Hours.hours(pred_range)/short_range)
    else:
        num_steps = int(Days.hours(pred_range)/short_range)
print(num_steps)

forecast_model_kwargs = {
    "in_channels": len(forecast_data_module.hparams.in_vars),
    "out_channels": len(forecast_data_module.hparams.out_vars),
    "n_blocks": 19,
    "num_steps": num_steps,
    "iter_model_out_channels": 3
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
    optim_kwargs=forecast_optim_kwargs,
    iterative_model=True,
    iter_model_out_vars=3,
)
set_climatology(forecast_model_module, forecast_data_module)

project_name = "Iterative_model_2.8_64epochs_36vars"

forecast_trainer = Trainer(
    seed=0,
    accelerator="gpu",
    devices=[1],
    precision=16,
    max_epochs=64,
    logger = WandbLogger(project = project_name, name = "era"),
    early_stopping=False,
    task = "forecasting"
)
forecast_trainer.fit(forecast_model_module, forecast_data_module)
forecast_trainer.test(forecast_model_module, forecast_data_module)

# forecast_trainer.test(forecast_model_module, forecast_data_module, "path")
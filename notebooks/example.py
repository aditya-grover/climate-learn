import climate_learn as cl
from climate_learn.data.climate_dataset.args import ERA5Args
from climate_learn.data.task.args import ForecastingArgs
from climate_learn.data.dataset.args import MapDatasetArgs

import wandb
import argparse
from pytorch_lightning import seed_everything
from pytorch_lightning.loggers import WandbLogger
from datetime import datetime
from transformers import AutoConfig, ViTImageProcessor

now = datetime.now()
now = now.strftime("%H-%M-%S_%d-%m-%Y")

config = AutoConfig.from_pretrained('google/vit-large-patch16-224-in21k')
processor = ViTImageProcessor.from_pretrained('google/vit-large-patch16-224-in21k')
print(processor)
exit()


def load_data():
    seed_everything(42, workers=True)

    root = "/data0/datasets/weatherbench/data/weatherbench/era5/5.625deg/"
    variables = ["geopotential_500"]#, "temperature_850", "2m_temperature"]
    # variables = ['2m_temperature']
    in_vars = out_vars = [f"era5:{v}" for v in variables]
    train_years = range(1979, 2016)
    val_years = range(2016, 2017)
    test_years = range(2017, 2019)

    forecasting_args = ForecastingArgs(
        in_vars,
        out_vars,
        pred_range=72,
        subsample=6
    )

    train_dataset_args = MapDatasetArgs(
        ERA5Args(root, variables, train_years),
        forecasting_args
    )

    val_dataset_args = MapDatasetArgs(
        ERA5Args(root, variables, val_years),
        forecasting_args
    )

    test_dataset_args = MapDatasetArgs(
        ERA5Args(root, variables, test_years),
        forecasting_args
    )

    dm = cl.data.DataModule(
        train_dataset_args,
        val_dataset_args,
        test_dataset_args,
        batch_size=32,
        num_workers=8
    )

    return dm


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", type=int, default=0)
    args = parser.parse_args()

    dm = load_data()

    # climatology is the average value over the training period
    climatology = cl.load_forecasting_module(data_module=dm, preset="climatology")

    # persistence returns its input as its prediction
    persistence = cl.load_forecasting_module(data_module=dm, preset="persistence")

    #VIT Pretrained
    vit_pretrained = cl.load_forecasting_module(
        data_module=dm, 
        preset="vit", 
        use_pretrained_backbone=False, 
        use_pretrained_embeddings=False, 
        freeze_backbone=False,
        freeze_embeddings=False,
    )

    wandb.init(project='Climate', name=f'VIT Fresh Backbone, New Embeddings (Patch Size = 4) {now}')
    logger = WandbLogger()

    gpu_num = args.gpu

    trainer = cl.Trainer(
        # stop when latitude-weighted RMSE, a validation metric, stops improving
        early_stopping="lat_rmse:aggregate [val]",
        # wait for 10 epochs of no improvement
        patience=10,
        # uncomment to use gpu acceleration
        accelerator="gpu",
        devices=[gpu_num],
        # max epochs
        max_epochs=50,
        # log to wandb
        logger=logger,
        # Print model summary
        enable_model_summary=False,
    )

    trainer.fit(vit_pretrained, dm)

    # print('Testing Climatology')
    # trainer.test(climatology, dm)

    # print('Testing Persistence')
    # trainer.test(persistence, dm)

    # print('Testing VIT Pretrained')
    # trainer.test(vit_pretrained, dm)


if __name__ == "__main__":
    main()
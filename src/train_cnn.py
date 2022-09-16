import os

from pytorch_lightning.utilities.cli import LightningCLI

from src.models.cnn_module import CNNLitModule
from src.datamodules.era5_datamodule import ERA5DataModule


def main():
    cli = LightningCLI(
        model_class=CNNLitModule,
        datamodule_class=ERA5DataModule,
        seed_everything_default=42,
        save_config_overwrite=True,
        run=False,
        auto_registry=True,
        parser_kwargs={"parser_mode": "omegaconf", "error_handler": None},
    )
    os.makedirs(cli.trainer.default_root_dir, exist_ok=True)

    normalization = cli.datamodule.get_out_transforms()
    mean_norm, std_norm = normalization.mean, normalization.std
    mean_denorm, std_denorm = -mean_norm / std_norm, 1 / std_norm
    cli.model.set_denormalization(mean_denorm, std_denorm)
    cli.model.set_lat_lon(*cli.datamodule.get_lat_lon())
    cli.model.set_pred_range(cli.datamodule.hparams.pred_range)

    # fit() runs the training
    cli.trainer.fit(cli.model, datamodule=cli.datamodule)


if __name__ == "__main__":
    main()

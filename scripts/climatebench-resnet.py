# Standard library
from argparse import ArgumentParser

# Third party
import climate_learn as cl
from climate_learn.data import ClimateDataModule
import torch.multiprocessing
from pytorch_lightning.loggers.tensorboard import TensorBoardLogger


def main():
    torch.multiprocessing.set_sharing_strategy("file_system")

    parser = ArgumentParser()
    parser.add_argument("root_dir")
    parser.add_argument("out_var", type=str) # tas, diurnal_temperature_range, pr, pr90
    parser.add_argument("gpu", type=int)
    args = parser.parse_args()
    
    variables=[
        'CO2',
        'SO2',
        'CH4',
        'BC'
    ]
    out_variables = args.out_var

    history = 10
    train_ratio = 0.9
    batch_size = 16
    default_root_dir=f"results/resnet_climatebech_new_setting_50_epochs_{args.out_var}"
    
    dm = ClimateDataModule(
        args.root_dir,
        variables=variables,
        out_variables=out_variables,
        train_ratio=train_ratio,
        history=history,
        batch_size=batch_size,
        num_workers=1
    )
    
    model = cl.models.hub.ResNet(
        in_channels=4,
        out_channels=1,
        history=history,
        hidden_channels=128,
        activation="leaky",
        norm=True,
        dropout=0.1,
        n_blocks=28,
    )
    optimizer = cl.load_optimizer(
        model, "AdamW", {"lr": 5e-4, "weight_decay": 1e-5, "betas": (0.9, 0.99)}
    )
    lr_scheduler = cl.load_lr_scheduler(
        "linear-warmup-cosine-annealing",
        optimizer,
        {"warmup_epochs": 5, "max_epochs": 50, "warmup_start_lr": 1e-8, "eta_min": 1e-8}
    )
    resnet = cl.load_climatebench_module(
        data_module=dm,
        model=model,
        optim=optimizer,
        sched=lr_scheduler
    )
    logger = TensorBoardLogger(
        save_dir=f"{default_root_dir}/logs"
    )
    trainer = cl.Trainer(
        early_stopping="val/mse:aggregate",
        patience=10,
        accelerator="gpu",
        devices=[args.gpu],
        precision=16,
        max_epochs=50,
        default_root_dir=default_root_dir,
        logger=logger,
        log_every_n_steps=1
    )
    
    trainer.fit(resnet, datamodule=dm)
    trainer.test(resnet, datamodule=dm, ckpt_path="best")

    
if __name__ == "__main__":
    main()
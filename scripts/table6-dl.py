# Standard library
from argparse import ArgumentParser
import os

# Third party
import climate_learn as cl
from climate_learn.models.hub import VisionTransformer, Interpolation
from climate_learn.transforms import Mask, Denormalize
from climate_learn.data import ERA5ToPrism
from pytorch_lightning.loggers.tensorboard import TensorBoardLogger
import torch.nn as nn


def main():
    parser = ArgumentParser()
    parser.add_argument("preset")
    parser.add_argument("gpu", type=int)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--checkpoint", default=None)
    args = parser.parse_args()
    
    dm = ERA5ToPrism(
        os.path.join(os.environ["PRISM_DIR"], "era5_cropped"),
        os.path.join(os.environ["PRISM_DIR"], "prism_processed"),
        batch_size=args.batch_size,
        num_workers=args.num_workers
    )
    dm.setup()

    mask = Mask(dm.get_out_mask().to(device=f"cuda:{args.gpu}"))
    denorm = Denormalize(dm)
    denorm_mask = lambda x: denorm(mask(x))
    
    if args.preset == "vit":
        net = nn.Sequential(
            Interpolation((32,64), "bilinear"),
            VisionTransformer(
                img_size=(32,64),
                in_channels=1,
                out_channels=1,
                history=1,
                patch_size=2,
                learn_pos_emb=True,
                embed_dim=128,
                depth=8,
                decoder_depth=2,
                num_heads=4
            )
        )
        optim_kwargs = {
            "lr": 1e-5,
            "weight_decay": 1e-5,
            "betas": (0.9, 0.99)
        }
        sched_kwargs = {
            "warmup_epochs": 5,
            "max_epochs": 50,
            "warmup_start_lr": 1e-8,
            "eta_min": 1e-8
        }
        model = cl.load_downscaling_module(
            data_module=dm,
            model=net,
            optim="adamw",
            optim_kwargs=optim_kwargs,
            sched="linear-warmup-cosine-annealing",
            sched_kwargs=sched_kwargs,
            train_target_transform=mask,
            val_target_transform=[denorm_mask, denorm_mask, denorm_mask, mask],
            test_target_transform=[denorm_mask, denorm_mask, denorm_mask]
        )
    else:
        model = cl.load_downscaling_module(
            data_module=dm,
            preset=args.preset,
            train_target_transform=mask,
            val_target_transform=[denorm_mask, denorm_mask, denorm_mask, mask],
            test_target_transform=[denorm_mask, denorm_mask, denorm_mask]
        )
    default_root_dir = f"{args.preset}_downscaling_prism"
    logger = TensorBoardLogger(save_dir=f"{default_root_dir}/logs")

    if args.checkpoint:
        trainer = cl.Trainer(
            accelerator="gpu",
            devices=[args.gpu],
            logger=logger,
            precision="16",
            summary_depth=1
        )
        model = cl.LitModule.load_from_checkpoint(
            args.checkpoint,
            net=model.net,
            optimizer=model.optimizer,
            lr_scheduler=None,
            train_loss=None,
            val_loss=None,
            test_loss=model.test_loss,
            test_target_tranfsorms=model.test_target_transforms
        )
        trainer.test(model, datamodule=dm)
    else:
        trainer = cl.Trainer(
            early_stopping="val/mse:aggregate",
            patience=5,
            accelerator="gpu",
            devices=[args.gpu],
            max_epochs=50,
            default_root_dir=default_root_dir,
            logger=logger,
            precision="16",
            summary_depth=1
        )
        trainer.fit(model, datamodule=dm)
        trainer.test(model, datamodule=dm, ckpt_path="best")

    
if __name__ == "__main__":
    main()
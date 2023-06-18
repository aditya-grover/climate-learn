# Standard library
from argparse import ArgumentParser
import os

# Third party
import climate_learn as cl
from climate_learn.transforms import Mask, Denormalize
from climate_learn.data import ERA5ToPrism


def main():
    parser = ArgumentParser()
    parser.add_argument("gpu", type=int)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=4)
    args = parser.parse_args()

    dm = ERA5ToPrism(
        os.path.join(os.environ["PRISM_DIR"], "era5_cropped"),
        os.path.join(os.environ["PRISM_DIR"], "prism_processed"),
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )
    dm.setup()

    mask = Mask(dm.get_out_mask().to(device=f"cuda:{args.gpu}"))
    denorm = Denormalize(dm)
    denorm_mask = lambda x: denorm(mask(x))

    nearest = cl.load_downscaling_module(
        data_module=dm,
        preset="nearest-interpolation",
        train_target_transform=mask,
        val_target_transform=[denorm_mask, denorm_mask, denorm_mask, mask],
        test_target_transform=[denorm_mask, denorm_mask, denorm_mask],
    )
    bilinear = cl.load_downscaling_module(
        data_module=dm,
        preset="bilinear-interpolation",
        train_target_transform=mask,
        val_target_transform=[denorm_mask, denorm_mask, denorm_mask, mask],
        test_target_transform=[denorm_mask, denorm_mask, denorm_mask],
    )
    trainer = cl.Trainer(accelerator="gpu", devices=[args.gpu])
    trainer.test(nearest, dm)
    trainer.test(bilinear, dm)


if __name__ == "__main__":
    main()

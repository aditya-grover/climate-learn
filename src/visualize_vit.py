import os

import imageio
import matplotlib.pyplot as plt
import numpy as np
import torch
from pytorch_lightning.utilities.cli import LightningCLI
from torchvision.transforms import transforms
from tqdm import tqdm

from src.datamodules.era5_datamodule import ERA5DataModule
from src.models.vit_module import ViTLitModule


def get_data_traj(dataset, start_idx, steps):
    # visualize a trajectory from the dataset, starting from start_idx
    dataset_traj = []
    inv_normalize = transforms.Normalize(
        mean=[-277.0595 / 21.289722, 0.05025468 / 5.5454874, -0.18755548 / 4.764006],
        std=[1 / 21.289722, 1 / 5.5454874, 1 / 4.764006],
    )
    for i in range(steps):
        idx = start_idx + i
        inp, _ = dataset[idx]
        dataset_traj.append(inv_normalize(inp).numpy())
    dataset_traj = np.array(dataset_traj)
    return dataset_traj, get_seq_imgs(dataset_traj, "Data")


def get_model_traj(model, dataset, gt_traj, start_idx, steps):
    inv_normalize = transforms.Normalize(
        mean=[-277.0595 / 21.289722, 0.05025468 / 5.5454874, -0.18755548 / 4.764006],
        std=[1 / 21.289722, 1 / 5.5454874, 1 / 4.764006],
    )
    x = dataset[start_idx][0].unsqueeze(0)
    pred_traj = [inv_normalize(x.squeeze()).numpy()]

    pbar = tqdm(range(steps))
    pbar.set_description("Rolling out predictions")
    for i in pbar:
        x = model.forward(x)
        pred_traj.append(inv_normalize(x.squeeze()).numpy())
    pred_traj = np.array(pred_traj)
    diff_traj = pred_traj - gt_traj
    return get_seq_imgs(pred_traj, "Model"), get_seq_imgs(diff_traj, "Difference")


def get_seq_imgs(data_traj, caption):
    temp_traj = data_traj[:, 0]  # (20, 128, 256)
    wind_u_traj = data_traj[:, 1]  # (20, 128, 256)
    wind_v_traj = data_traj[:, 2]  # (20, 128, 256)
    # create gif visualization
    steps = len(temp_traj)
    all_steps = []
    for i in range(steps):
        fig, axes = plt.subplots(
            3, 1, figsize=(10, 12)
        )  # Caution, figsize will also influence positions.
        im1 = axes[0].imshow(temp_traj[i])
        im1.set_cmap(cmap=plt.cm.RdBu)
        fig.colorbar(im1, ax=axes[0])
        axes[0].set_title("temperature")

        im2 = axes[1].imshow(wind_u_traj[i])
        im2.set_cmap(cmap=plt.cm.RdBu)
        fig.colorbar(im2, ax=axes[1])
        axes[1].set_title("wind_u")

        im3 = axes[2].imshow(wind_v_traj[i])
        im3.set_cmap(cmap=plt.cm.RdBu)
        fig.colorbar(im3, ax=axes[2])
        axes[2].set_title("wind_v")
        axes[2].set_xlabel(caption, fontsize=15)

        fig.tight_layout()

        fig.canvas.draw()  # draw the canvas, cache the renderer
        image = np.frombuffer(fig.canvas.tostring_rgb(), dtype="uint8")
        image = image.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        all_steps.append(image)

        plt.close(fig)

    return all_steps


class MyLightningCLI(LightningCLI):
    def add_arguments_to_parser(self, parser):
        parser.add_argument("--steps", type=int, default=20)
        parser.add_argument("--ckpt", type=str, required=True)
        parser.add_argument(
            "--save_dir",
            type=str,
            default="/home/t-tungnguyen/climate_pretraining/visualization_vit",
        )
        parser.add_argument("--filename", type=str, default="model.gif")


def main(model, dataset, args):
    os.makedirs(args.save_dir, exist_ok=True)

    dataset.setup()
    dataset = dataset.data_test

    steps = args.steps  # number of rollouts to the future
    start_idx = np.random.randint(
        low=0, high=len(dataset) - steps
    )  # choose a random index from the dataset

    data_traj, data_img_traj = get_data_traj(
        dataset, start_idx, steps + 1
    )  # ground truth
    pred_traj, diff_traj = get_model_traj(
        model, dataset, data_traj, start_idx, steps
    )  # prediction
    traj = [
        np.concatenate((data_img_traj[i], pred_traj[i], diff_traj[i]), axis=1)
        for i in range(len(data_traj))
    ]
    os.makedirs(args.save_dir, exist_ok=True)
    imageio.mimsave(os.path.join(args.save_dir, args.filename), traj, fps=5)


if __name__ == "__main__":
    cli = MyLightningCLI(
        model_class=ViTLitModule,
        datamodule_class=ERA5DataModule,
        seed_everything_default=42,
        save_config_overwrite=True,
        run=False,
        auto_registry=True,
        parser_kwargs={"parser_mode": "omegaconf", "error_handler": None},
    )
    state_dict = torch.load(cli.config.ckpt)["state_dict"]
    msg = cli.model.load_state_dict(state_dict)
    print(msg)

    main(cli.model, cli.datamodule, cli.config)

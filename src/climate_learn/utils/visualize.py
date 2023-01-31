# Standard library
from datetime import datetime
import os
import random

# Third party
import matplotlib.pyplot as plt
import numpy as np
import torch
from tqdm import tqdm

# TODO: include exceptions in docstrings


def visualize(model_module, data_module, split="test", samples=2, save_dir=None):
    """Visualizes model bias.

    :param model_module: A ClimateLearn model.
    :type model_module: LightningModule
    :param data_module: A ClimateLearn dataset.
    :type data_module: LightningDataModule
    :param split: "train", "val", or "test".
    :type split: str, optional
    :param samples: The exact days or the number of days to visualize. If provided as
        exact days, this should be a list of datetime strings, each formatted as
        "YYYY-mm-dd:HH". If provided as the number of days, it must be an int n. In
        this case, n days are randomly sampled from the given split.
    :type samples: List[str]|int, optional
    :param save_dir: The directory to save the visualization to. Defaults to `None`,
        meaning the visualization is not saved.
    :type save_dir: str, optional
    """
    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)

    # dataset.setup()
    dataset = eval(f"data_module.{split}_dataset")

    if type(samples) == int:
        idxs = random.sample(range(0, len(dataset)), samples)
    elif type(samples) == list:
        idxs = [
            np.searchsorted(
                dataset.time, np.datetime64(datetime.strptime(dt, "%Y-%m-%d:%H"))
            )
            for dt in samples
        ]
    else:
        raise Exception(
            "Invalid type for samples; Allowed int or list[datetime.datetime or np.datetime64]"
        )

    fig, axes = plt.subplots(len(idxs), 4, figsize=(30, 3 * len(idxs)), squeeze=False)

    for index, idx in enumerate(idxs):
        x, y, _, _ = dataset[idx]  # 1, 1, 32, 64
        pred = model_module.forward(x.unsqueeze(0))  # 1, 1, 32, 64

        inv_normalize = model_module.denormalization
        init_condition, gt = inv_normalize(x), inv_normalize(y)
        pred = inv_normalize(pred)
        bias = pred - gt

        for i, tensor in enumerate([init_condition, gt, pred, bias]):
            ax = axes[index][i]
            im = ax.imshow(tensor.detach().squeeze().cpu().numpy())
            im.set_cmap(cmap=plt.cm.RdBu)
            fig.colorbar(im, ax=ax)

        if data_module.hparams.task == "forecasting":
            axes[index][0].set_title("Initial condition [Kelvin]")
            axes[index][1].set_title("Ground truth [Kelvin]")
            axes[index][2].set_title("Prediction [Kelvin]")
            axes[index][3].set_title("Bias [Kelvin]")
        elif data_module.hparams.task == "downscaling":
            axes[index][0].set_title("Low resolution data [Kelvin]")
            axes[index][1].set_title("High resolution data [Kelvin]")
            axes[index][2].set_title("Downscaled [Kelvin]")
            axes[index][3].set_title("Bias [Kelvin]")
        else:
            raise NotImplementedError

    fig.tight_layout()

    if save_dir is not None:
        plt.savefig(os.path.join(save_dir, "visualize.png"))
    else:
        plt.show()


def visualize_mean_bias(model_module, data_module, save_dir=None):
    """Visualizes mean model bias on the test set.

    :param model_module: A ClimateLearn model.
    :type model_module: LightningModule
    :param data_module: A ClimateLearn dataset.
    :type data_module: LightningDataModule
    :param save_dir: The directory to save the visualization to. Defaults to `None`,
        meaning the visualization is not saved.
    :type save_dir: str, optional
    """
    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)

    loader = data_module.test_dataloader()

    all_mean_bias = []
    for batch in tqdm(loader):
        x, y, _, _ = batch  # B, 1, 32, 64
        x = x.to(model_module.device)
        y = y.to(model_module.device)
        pred = model_module.forward(x)  # B, 1, 32, 64

        inv_normalize = model_module.denormalization
        init_condition, gt = inv_normalize(x), inv_normalize(y)
        pred = inv_normalize(pred)
        bias = pred - gt  # B, 1, 32, 64
        mean_bias = bias.mean(dim=0)
        all_mean_bias.append(mean_bias)

    all_mean_bias = torch.stack(all_mean_bias, dim=0)
    mean_bias = torch.mean(all_mean_bias, dim=0)

    fig, axes = plt.subplots(1, 1, figsize=(12, 4), squeeze=False)
    ax = axes[0, 0]

    im = ax.imshow(mean_bias.detach().squeeze().cpu().numpy())
    im.set_cmap(cmap=plt.cm.RdBu)
    fig.colorbar(im, ax=ax)
    ax.set_title("Mean bias [Kelvin]")

    fig.tight_layout()

    if save_dir is not None:
        plt.savefig(os.path.join(save_dir, "visualize_mean_bias.png"))
    else:
        plt.show()

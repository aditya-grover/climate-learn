import os

import matplotlib.pyplot as plt
import numpy as np

from .models.forecast_module import ForecastLitModule
from .datamodules.era5_datamodule import ERA5DataModule


def visualize_forecast(module: ForecastLitModule, datamodule: ERA5DataModule, save_dir=None):
    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)

    # dataset.setup()
    dataset = datamodule.test_dataset

    random_idx = np.random.randint(
        low=0, high=len(dataset)
    )  # choose a random index from the dataset

    x, y, _, _ = dataset[random_idx] # 1, 1, 32, 64
    pred = module.forward(x.unsqueeze(0)) # 1, 1, 32, 64

    inv_normalize = module.denormalization
    init_condition, gt = inv_normalize(x), inv_normalize(y)
    pred = inv_normalize(pred)
    bias = pred - gt

    fig, axes = plt.subplots(1, 4, figsize=(20, 2))

    for i, tensor in enumerate([init_condition, gt, pred, bias]):
        ax = axes[i]
        im = ax.imshow(tensor.squeeze().cpu().numpy())
        im.set_cmap(cmap=plt.cm.RdBu)
        fig.colorbar(im, ax=ax)

    axes[0].set_title("Initial condition")
    axes[1].set_title("Ground truth")
    axes[2].set_title("Prediction")
    axes[3].set_title("Bias")

    fig.tight_layout()
    
    if save_dir is not None:
        plt.savefig(os.path.join(save_dir, 'visualize.png'))
    else:
        plt.show()

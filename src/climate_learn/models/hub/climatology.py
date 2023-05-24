# Local application
from .utils import register

# Third party
from torch import nn
from torchvision import transforms


@register("climatology")
class Climatology(nn.Module):
    def __init__(self, clim, mean, std):
        super().__init__()
        self.norm = transforms.Normalize(mean, std)
        self.clim = clim  # clim.shape = [C,H,W]

    def forward(self, x):
        # x.shape = [B,T,C,H,W]
        yhat = self.norm(self.clim).unsqueeze(0).repeat(x.shape[0], 1, 1, 1)
        # yhat.shape = [B,C,H,W]
        return yhat

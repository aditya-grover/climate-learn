from .climatology import Climatology
from .interpolation import Interpolation
from .linear_regression import LinearRegression
from .persistence import Persistence
from .resnet import ResNet
from .unet import Unet
from .vit import VisionTransformer

from .registry import MODEL_REGISTRY

__all__ = list(MODEL_REGISTRY.values())
from .components.vit import VisionTransformer
from .components.resnet import ResNet
from .components.unet import Unet
from .forecast_module import ForecastLitModule

def load_model(name, task, model_kwargs, optim_kwargs):
    if name == 'vit':
        model_cls = VisionTransformer
    elif name == 'resnet':
        model_cls = ResNet
    elif name == 'unet':
        model_cls = Unet
    model = model_cls(**model_kwargs)

    if task == 'forecast':
        module = ForecastLitModule(model, **optim_kwargs)
    else:
        raise NotImplementedError('Only support forecasting for now')
    
    return module
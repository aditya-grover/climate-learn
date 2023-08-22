# Local application
from .components.cnn_blocks import PeriodicConv2D, ResidualBlock
from .utils import register
from .resnet import ResNet

# Third party
import os
import torch
from torch import nn
from torch.distributions.normal import Normal
from glob import glob


@register("resnet_ensemble")
class ResNetEnsemble(nn.Module):
    def __init__(
        self,
        resnet_base_args,
        pretrained_prefix,
    ) -> None:
        super().__init__()
        self.nets = nn.ModuleList()
        
        pretrained_paths = glob(pretrained_prefix + '*')
        for p in pretrained_paths:
            print ('Loading ensemble ' + p)
            net = ResNet(**resnet_base_args)
            ckpt_path = glob(os.path.join(p, 'checkpoints', 'epoch_*'))[0]
            state_dict = torch.load(ckpt_path, map_location='cpu')['state_dict']
            state_dict = {k[4:]: v for k, v in state_dict.items()}
            msg = net.load_state_dict(state_dict)
            print (msg)
            self.nets.append(net)

    def forward(self, x):
        preds = [net(x) for net in self.nets]
        preds = torch.stack(preds, dim=0)
        mean_pred = torch.mean(preds, dim=0)
        std_pred = torch.std(preds, dim=0)
        return Normal(mean_pred, std_pred)
    

# resnet_base_args = {
#     'in_channels': 49,
#     'out_channels': 3,
#     'history': 3,
#     'hidden_channels': 128,
#     'activation': "leaky",
#     'norm': True,
#     'dropout': 0.1,
#     'n_blocks': 28,
# }
# pretrained_prefix = '/home/tungnd/climate-learn/results_rebuttal/resnet_120_ensemble'
# resnet_ensemble = ResNetEnsemble(resnet_base_args, pretrained_prefix).cuda()
# x = torch.randn(4, 147, 32, 64).cuda()
# out = resnet_ensemble(x)
# print (out)

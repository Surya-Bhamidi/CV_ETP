"""Local initializer used only before strict-loading the complete checkpoint."""
from torch import nn
import torch


@torch.no_grad()
def default_init_weights(module_list, scale=1, bias_fill=0, **kwargs):
    if not isinstance(module_list, list):
        module_list = [module_list]
    for module in module_list:
        for layer in module.modules():
            if isinstance(layer, (nn.Conv2d, nn.Linear)):
                nn.init.kaiming_normal_(layer.weight, **kwargs)
                layer.weight.mul_(scale)
                if layer.bias is not None:
                    layer.bias.fill_(bias_fill)

import torch
from torch import nn
from spikingjelly.activation_based import layer, functional
import copy

def conv_bn(in_channels, out_channels, kernel_size, stride, padding, groups=1):
    result = nn.Sequential()
    result.add_module('conv', layer.Conv2d(in_channels=in_channels, out_channels=out_channels,
                                                  kernel_size=kernel_size, stride=stride, padding=padding, groups=groups, bias=False))
    result.add_module('bn', layer.BatchNorm2d(num_features=out_channels))
    return result

def repvgg_model_convert(model:nn.Module, save_path=None, do_copy=True):
    if do_copy:
        model = copy.deepcopy(model)
    functional.reset_net(model)
    for module in model.modules():
        if hasattr(module, 'switch_to_deploy'):
            module.switch_to_deploy()
    if save_path is not None:
        torch.save(model.state_dict(), save_path)
    return model
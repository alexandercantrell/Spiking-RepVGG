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

def add_cnf(x,y):
    return torch.add(x,y)

def and_cnf(x,y):
    return nn.functional.relu(torch.subtract(torch.add(x,y),1))

def iand_cnf(x,y):
    return nn.functional.relu(torch.subtract(x,y))

def or_cnf(x,y):
    z = torch.add(x,y)
    return torch.where(z>1.0,1.0,z)

def xor_cnf(x,y):
    return torch.remainder(torch.add(x,y),2)

class ConnectingFunction(nn.Module):
    def __init__(self,cnf):
        super(ConnectingFunction,self).__init__()
        if cnf == 'ADD':
            self.cnf = lambda x,y: x+y
        elif cnf == 'AND':
            self.cnf = and_cnf
        elif cnf == 'IAND':
            self.cnf = iand_cnf
        elif cnf == 'OR':
            self.cnf = or_cnf
        elif cnf == 'XOR':
            self.cnf = xor_cnf
        else:
            raise NotImplementedError(f'{cnf} is a connecting function that has not been implemented.')
    def forward(self,x,y):
        return self.cnf(x,y)

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
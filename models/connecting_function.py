import torch
from torch import nn
from spikingjelly.activation_based import layer

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

def dand_cnf(x,y):
    return x * y

def iand_cnf(x,y):
    return nn.functional.relu(torch.subtract(x,y))

def diand_cnf(x,y):
    return x * (1. - y)

def or_cnf(x,y):
    z = torch.add(x,y)
    return torch.where(z>1.0,1.0,z)

def dor_cnf(x,y):
    return x+y-(x*y)

def xor_cnf(x,y):
    return torch.remainder(torch.add(x,y),2)

def dxor_cnf(x,y):
    return x+y-(2*x*y)

class ConnectingFunction(nn.Module):
    def __init__(self,cnf):
        super(ConnectingFunction,self).__init__()
        if cnf == 'ADD':
            self.cnf = lambda x,y: x+y
        elif cnf == 'AND':
            self.cnf = and_cnf
        elif cnf == 'DAND':
            self.cnf = dand_cnf
        elif cnf == 'IAND':
            self.cnf = iand_cnf
        elif cnf == 'DIAND':
            self.cnf = diand_cnf
        elif cnf == 'OR':
            self.cnf = or_cnf
        elif cnf == 'DOR':
            self.cnf = dor_cnf
        elif cnf == 'XOR':
            self.cnf = xor_cnf
        elif cnf == 'DXOR':
            self.cnf = dxor_cnf
        else:
            raise NotImplementedError(f'{cnf} is a connecting function that has not been implemented.')
    def forward(self,x,y):
        return self.cnf(x,y)

import torch
import torch.nn as nn
from spikingjelly.activation_based import layer, neuron, surrogate
from connecting_functions import ConnectingFunction
from connecting_neuron import ParaConnLIFNode

def convrelupxp(in_channels, out_channels, stride=1):
    if stride != 1:
        return nn.Sequential(
            layer.Conv2d(in_channels, in_channels, kernel_size=3, stride=stride,
                        groups=in_channels, padding=1, bias=False),
            layer.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, bias=False),
            layer.BatchNorm2d(out_channels),
            nn.ReLU()
        )
    else:
        return nn.Sequential(
            layer.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, bias=False),
            layer.BatchNorm2d(out_channels),
            nn.ReLU()
        )
    
class SpikeResNetBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride, groups=1, deploy=False):
        super(SpikeResNetBlock, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.stride = stride
        self.groups = groups
        self.deploy = deploy
        self.conv1 = nn.Sequential(
            layer.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride,
                        groups=groups, padding=1, bias=False),
            layer.BatchNorm2d(out_channels),
            neuron.ParametricLIFNode(v_threshold=1.0, detach_reset=True, surrogate_function=surrogate.ATan()),
        )
        self.conv2 = nn.Sequential(
            layer.Conv2d(out_channels, out_channels, kernel_size=3, stride=1,
                        groups=groups, padding=1, bias=False),
            layer.BatchNorm2d(out_channels),
        )
        self.sn = neuron.ParametricLIFNode(v_threshold=1.0, detach_reset=True, surrogate_function=surrogate.ATan())
        
        if stride==1 and in_channels==out_channels:
            self.identity = nn.Identity()
            if not deploy:
                self.aac = nn.Identity()
        else:
            self.identity = nn.Sequential(
                layer.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                layer.BatchNorm2d(out_channels)
            )
            if not deploy:
                self.aac = convrelupxp(in_channels, out_channels, stride=stride)
        
    def forward(self, x):
        if self.deploy:
            return self.sn(self.conv2(self.conv1(x)) + self.identity(x))
        else:
            x,y = x
            out = self.conv2(self.conv1(x))
            id = self.identity(x)
            if y is not None:
                y = self.aac(y) + out
            else:
                y = out
            return self.sn(out+id), y
        
    def switch_to_deploy(self):
        if self.deploy:
            return
        if hasattr(self, 'aac'):
            self.__delattr__('aac')
        self.deploy=True
            
        
        
class SEWResNetBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride, groups=1, deploy=False, cnf='AND'):
        super(SEWResNetBlock, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.stride = stride
        self.groups = groups
        self.deploy = deploy
        self.conv1 = nn.Sequential(
            layer.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride,
                        groups=groups, padding=1, bias=False),
            layer.BatchNorm2d(out_channels),
            neuron.ParametricLIFNode(v_threshold=1.0, detach_reset=True, surrogate_function=surrogate.ATan()),
        )
        self.conv2 = nn.Sequential(
            layer.Conv2d(out_channels, out_channels, kernel_size=3, stride=1,
                        groups=groups, padding=1, bias=False),
            layer.BatchNorm2d(out_channels),
            neuron.ParametricLIFNode(v_threshold=1.0, detach_reset=True, surrogate_function=surrogate.ATan())
        )
        
        if stride==1 and in_channels==out_channels:
            self.identity = nn.Identity()
            if not deploy:
                self.aac = nn.Identity()
        else:
            self.identity = nn.Sequential(
                layer.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                layer.BatchNorm2d(out_channels),
                neuron.ParametricLIFNode(v_threshold=1.0, detach_reset=True, surrogate_function=surrogate.ATan())
            )
            if not deploy:
                self.aac = convrelupxp(in_channels, out_channels, stride=stride)

        self.cnf = ConnectingFunction(cnf)
        
    def forward(self, x):
        if self.deploy:
            return self.cnf(self.conv2(self.conv1(x)), self.identity(x))
        else:
            x,y = x
            out = self.conv2(self.conv1(x))
            id = self.identity(x)
            if y is not None:
                y = self.aac(y) + out
            else:
                y = out
            return self.cnf(out,id), y
        
    def switch_to_deploy(self):
        if self.deploy:
            return
        if hasattr(self, 'aac'):
            self.__delattr__('aac')
        self.deploy=True
            
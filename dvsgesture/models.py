import torch
import torch.nn as nn
from spikingjelly.activation_based import layer, neuron, surrogate
from connecting_neuron import SpikeParaConnLIFNode

class SpikeBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride):
        super(SpikeBlock, self).__init__()
        self.identity = stride == 1 and in_channels == out_channels
        self.conv3x3 = layer.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, stride=stride, bias=False)
        self.bn3x3 = layer.BatchNorm2d(out_channels)
        self.conv1x1 = layer.Conv2d(in_channels, out_channels, kernel_size=1, padding=0, stride=stride, bias=False)
        self.bn = layer.BatchNorm2d(out_channels)
        self.sn = neuron.ParametricLIFNode(init_tau=2.0, detach_reset=True, surrogate_function=surrogate.ATan())

    def forward(self, x: torch.Tensor):
        out = self.bn(self.conv1x1(x) + self.bn3x3(self.conv3x3(x)))
        if self.identity:
            out = out + x
        return self.sn(out)
    
class ConnBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride):
        super(ConnBlock, self).__init__()
        self.identity = in_channels == out_channels and stride == 1
        self.conv3x3 = layer.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, stride=stride, bias=False)#
        self.bn3x3 = layer.BatchNorm2d(out_channels)
        self.conv1x1 = layer.Conv2d(in_channels, out_channels, kernel_size=1, padding=0, stride=stride, bias=False)#
        self.bn = layer.BatchNorm2d(out_channels)
        if self.identity:
            self.sn = SpikeParaConnLIFNode(init_tau=2.0, detach_reset=True, surrogate_function=surrogate.ATan())#
        else:
            self.sn = neuron.ParametricLIFNode(init_tau=2.0, detach_reset=True, surrogate_function=surrogate.ATan())

    def forward(self, x: torch.Tensor):
        out = self.bn(self.conv1x1(x) + self.bn3x3(self.conv3x3(x)))
        if self.identity:
            out = self.sn(out, x)
        else:
            out = self.sn(out)
        return out    

class Spike7BRVGGNet(nn.Module):
    def __init__(self, cfg_dict, num_classes):
        super(SpikeRVGGNet, self).__init__()
        if cfg_dict['block_type'] == 'spike':
            self.block = SpikeBlock
        elif cfg_dict['block_type'] == 'conn':
            self.block = ConnBlock
        else:
            raise NotImplementedError
        
        in_channels = 2
        layer_list = cfg_dict['layers']
        self.in_layer = self._build_layer(in_channels, layer_list[0])
        in_channels = layer_list[0]['channels']

        convs = nn.Sequential()
        for layer_dict in layer_list[1:]:
            convs.extend(self._build_layer(in_channels, layer_dict))
            in_channels = layer_dict['channels']

        self.convs = convs

        self.avgpool = layer.AdaptiveAvgPool2d((1, 1))
        
        self.flatten = nn.Flatten(2)

        self.out = layer.Linear(in_channels, num_classes, bias=True)

    def _build_layer(self, in_channels, layer_dict):
        channels = layer_dict['channels']
        stride = layer_dict['stride']
        convs = nn.Sequential()
        convs.append(self.block(in_channels, channels, 1))
        for _ in range(layer_dict['num_blocks'] - 1):
            convs.append(self.block(channels, channels, 1))
        if stride > 1:
            convs.append(layer.MaxPool2d(stride))
        return convs
    
    def forward(self, x: torch.Tensor):
        x = x.permute(1,0,2,3,4)
        x = self.in_layer(x)
        x = self.convs(x)
        x = self.avgpool(x)
        x = self.flatten(x)
        return self.out(x.mean(0))

class SpikeRVGGNet(nn.Module):
    def __init__(self, cfg_dict, num_classes):
        super(SpikeRVGGNet, self).__init__()
        if cfg_dict['block_type'] == 'spike':
            self.block = SpikeBlock
        elif cfg_dict['block_type'] == 'conn':
            self.block = ConnBlock
        else:
            raise NotImplementedError
        
        in_channels = 2
        layer_list = cfg_dict['layers']
        self.in_layer = self._build_layer(in_channels, layer_list[0])
        in_channels = layer_list[0]['channels']
        
        convs = nn.Sequential()
        for layer_dict in layer_list[1:]:
            convs.extend(self._build_layer(in_channels, layer_dict))
            in_channels = layer_dict['channels']

        self.convs = convs
        self.avgpool = layer.AdaptiveAvgPool2d((1, 1))
        self.flatten = nn.Flatten(2)
        self.out = layer.Linear(in_channels, num_classes, bias=True)

    def _build_layer(self, in_channels, layer_dict):
        channels = layer_dict['channels']
        stride = layer_dict['stride']
        convs = nn.Sequential()
        convs.append(self.block(in_channels, channels, stride))
        for _ in range(layer_dict['num_blocks'] - 1):
            convs.append(self.block(channels, channels, 1))
        return convs

    def forward(self, x: torch.Tensor):
        x = x.permute(1,0,2,3,4)
        x = self.in_layer(x)
        x = self.convs(x)
        x = self.avgpool(x)
        x = self.flatten(x)
        return self.out(x.mean(0))
    
def Spiking7BNet(num_classes):
    cfg_dict = {
        'layers': [
            {'channels': 64, 'num_blocks': 3, 'stride': 2},
            {'channels': 64, 'num_blocks': 2, 'stride': 2},
            {'channels': 64, 'num_blocks': 2, 'stride': 2},
            {'channels': 64, 'num_blocks': 2, 'stride': 2},
            {'channels': 128, 'num_blocks': 3, 'stride': 2},
            {'channels': 128, 'num_blocks': 2, 'stride': 2},
            {'channels': 128, 'num_blocks': 2, 'stride': 2},
        ],
        'block_type': 'spike',
    }
    return Spike7BRVGGNet(cfg_dict, num_classes)

def SpikingConn7BNet(num_classes):
    cfg_dict = {
        'layers': [
            {'channels': 64, 'num_blocks': 3, 'stride': 2},
            {'channels': 64, 'num_blocks': 2, 'stride': 2},
            {'channels': 64, 'num_blocks': 2, 'stride': 2},
            {'channels': 64, 'num_blocks': 2, 'stride': 2},
            {'channels': 128, 'num_blocks': 3, 'stride': 2},
            {'channels': 128, 'num_blocks': 2, 'stride': 2},
            {'channels': 128, 'num_blocks': 2, 'stride': 2},
        ],
        'block_type': 'conn',
    }
    return Spike7BRVGGNet(cfg_dict, num_classes)

def SpikingRVGGNet(num_classes):
    cfg_dict = {
        'layers': [
            {'channels': 64, 'num_blocks': 1, 'stride': 1},#128->128
            {'channels': 64, 'num_blocks': 6, 'stride': 2},#128->64
            {'channels': 64, 'num_blocks': 6, 'stride': 2},#64->32
            {'channels': 128, 'num_blocks': 5, 'stride': 2},#32->16
            {'channels': 128, 'num_blocks': 1, 'stride': 2},#16->8
        ],
        'block_type': 'spike',
    }
    return SpikeRVGGNet(cfg_dict, num_classes)

def SpikingConnRVGGNet(num_classes):
    cfg_dict = {
        'layers': [
            {'channels': 64, 'num_blocks': 1, 'stride': 1},#128->128
            {'channels': 64, 'num_blocks': 6, 'stride': 2},#128->64
            {'channels': 64, 'num_blocks': 6, 'stride': 2},
            {'channels': 128, 'num_blocks': 5, 'stride': 2},#32->16
            {'channels': 256, 'num_blocks': 1, 'stride': 2},#16->8
        ],
        'block_type': 'conn',
    }
    return SpikeRVGGNet(cfg_dict, num_classes)

model_dict = {
    'spiking7b': Spiking7BNet,
    'spikingconn7b': SpikingConn7BNet,
    'spiking': SpikingRVGGNet,
    'spikingconn': SpikingConnRVGGNet,
}

def get_model_by_name(name):
    return model_dict[name]
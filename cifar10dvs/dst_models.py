import torch
import torch.nn as nn
from spikingjelly.activation_based import layer, neuron, surrogate
from connecting_functions import ConnectingFunction
from connecting_neuron import ParaConnLIFNode, SpikeParaConnLIFNode

def convrelu3x3(in_channels, out_channels, stride):
    return nn.Sequential(
        layer.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, stride=stride, bias=False),
        layer.BatchNorm2d(out_channels),
        nn.ReLU()
    )

class SpikeBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride):
        super(SpikeBlock, self).__init__()
        self.identity = stride == 1 and in_channels == out_channels
        self.conv3x3 = layer.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, stride=stride, bias=False)
        self.bn3x3 = layer.BatchNorm2d(out_channels)
        self.conv1x1 = layer.Conv2d(in_channels, out_channels, kernel_size=1, padding=0, stride=stride, bias=False)
        self.bn = layer.BatchNorm2d(out_channels)
        self.sn = neuron.ParametricLIFNode(init_tau=2.0, detach_reset=True, surrogate_function=surrogate.ATan())
        if not self.identity:
            self.aac = convrelu3x3(in_channels, out_channels, stride)

    def forward(self, x: torch.Tensor, y: torch.Tensor = None):
        out = self.bn(self.conv1x1(x) + self.bn3x3(self.conv3x3(x)))
        if self.identity:
            if y is not None:
                y = y + out
            out = out + x
        elif y is not None:
            y = self.aac(y) + out
        return self.sn(out), y
    
class ConnBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride):
        super(ConnBlock, self).__init__()
        self.identity = stride == 1 and in_channels == out_channels
        self.conv3x3 = layer.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, stride=stride, bias=False)
        self.bn3x3 = layer.BatchNorm2d(out_channels)
        self.conv1x1 = layer.Conv2d(in_channels, out_channels, kernel_size=1, padding=0, stride=stride, bias=False)
        self.bn = layer.BatchNorm2d(out_channels)
        if self.identity:
            self.sn = SpikeParaConnLIFNode(init_tau=2.0, detach_reset=True, surrogate_function=surrogate.ATan())
        else:
            self.sn = neuron.ParametricLIFNode(init_tau=2.0, detach_reset=True, surrogate_function=surrogate.ATan())
            self.aac = convrelu3x3(in_channels, out_channels, stride)

    def forward(self, x: torch.Tensor, y: torch.Tensor = None):
        out = self.bn(self.conv1x1(x) + self.bn3x3(self.conv3x3(x)))
        if self.identity:
            if y is not None:
                y = y + out
            out = self.sn(out, x)
        else:
            if y is not None:
                y = self.aac(y) + out
            out = self.sn(out)
        return out, y

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
        
        convs = nn.ModuleList()
        for layer_dict in layer_list[1:]:
            convs.extend(self._build_layer(in_channels, layer_dict))
            in_channels = layer_dict['channels']
        
        convs.append(nn.Flatten(2))

        self.convs = convs

        x=y=128
        for layer_dict in layer_list:
            if isinstance(layer_dict['stride'], tuple):
                x = x // layer_dict['stride'][0]
                y = y // layer_dict['stride'][1]
            else:
                x = x // layer_dict['stride']
                y = y // layer_dict['stride']
        out_features = x * y * in_channels

        self.out = layer.Linear(out_features, num_classes, bias=True)
        self.acc = layer.Linear(out_features, num_classes, bias=True)

    def _build_layer(self, in_channels, layer_dict):
        channels = layer_dict['channels']
        stride = layer_dict['stride']
        convs = nn.ModuleList()
        convs.append(self.block(in_channels, channels, stride))
        for _ in range(layer_dict['num_blocks'] - 1):
            convs.append(self.block(channels, channels, 1))
        return convs
    
    def forward(self, x: torch.Tensor):
        x = x.permute(1,0,2,3,4)
        y = None
        for conv in self.in_layer:
            x, y = conv(x,y)
        if self.training:
            y = x
        for conv in self.convs:
            if isinstance(conv, nn.Flatten):
                x = conv(x)
                y = conv(y)
            else:
                x, y = conv(x,y)
        if y is not None:
            y = self.acc(y.mean(0))
        return self.out(x.mean(0)), y
    
def Spiking7BNetFSNN(num_classes):
    cfg_dict = {
        'layers': [
            {'channels': 64, 'num_blocks': 1, 'stride': 1},
            {'channels': 64, 'num_blocks': 2, 'stride': 1},
            {'channels': 64, 'num_blocks': 2, 'stride': 2},
            {'channels': 64, 'num_blocks': 2, 'stride': 2},
            {'channels': 64, 'num_blocks': 2, 'stride': 2},
            {'channels': 128, 'num_blocks': 2, 'stride': 2},
            {'channels': 128, 'num_blocks': 2, 'stride': 2},
            {'channels': 128, 'num_blocks': 2, 'stride': 2},
            {'channels': 128, 'num_blocks': 1, 'stride': 2}
        ],
        'block_type': 'spike',
    }
    return SpikeRVGGNet(cfg_dict, num_classes)

def SpikingConn7BNetFSNN(num_classes):
    cfg_dict = {
        'layers': [
            {'channels': 64, 'num_blocks': 1, 'stride': 1},
            {'channels': 64, 'num_blocks': 2, 'stride': 1},
            {'channels': 64, 'num_blocks': 2, 'stride': 2},
            {'channels': 64, 'num_blocks': 2, 'stride': 2},
            {'channels': 64, 'num_blocks': 2, 'stride': 2},
            {'channels': 128, 'num_blocks': 2, 'stride': 2},
            {'channels': 128, 'num_blocks': 2, 'stride': 2},
            {'channels': 128, 'num_blocks': 2, 'stride': 2},
            {'channels': 128, 'num_blocks': 1, 'stride': 2}
        ],
        'block_type': 'conn',
    }
    return SpikeRVGGNet(cfg_dict, num_classes)

def SpikingRVGGNetFSNN(num_classes):
    cfg_dict = {
        'layers': [
            {'channels': 64, 'num_blocks': 1, 'stride': 2},#128->64
            {'channels': 64, 'num_blocks': 2, 'stride': 2},#64->32
            {'channels': 64, 'num_blocks': 3, 'stride': 2},#32->16
            {'channels': 64, 'num_blocks': 3, 'stride': 2},#16->8
            {'channels': 128, 'num_blocks': 3, 'stride': 2},#8->4
            {'channels': 128, 'num_blocks': 3, 'stride': 2},#4->2
            {'channels': 128, 'num_blocks': 1, 'stride': 2},
        ],
        'block_type': 'spike',
    }
    return SpikeRVGGNet(cfg_dict, num_classes)

def SpikingConnRVGGNetFSNN(num_classes):
    cfg_dict = {
        'layers': [
            {'channels': 64, 'num_blocks': 1, 'stride': 2},#128->64
            {'channels': 64, 'num_blocks': 2, 'stride': 2},#64->32
            {'channels': 64, 'num_blocks': 3, 'stride': 2},#32->16
            {'channels': 64, 'num_blocks': 3, 'stride': 2},#16->8
            {'channels': 128, 'num_blocks': 3, 'stride': 2},#8->4
            {'channels': 128, 'num_blocks': 3, 'stride': 2},#4->2
            {'channels': 128, 'num_blocks': 1, 'stride': 2},
        ],
        'block_type': 'conn',
    }
    return SpikeRVGGNet(cfg_dict, num_classes)


model_dict = {
    'spiking7bfsnn': Spiking7BNetFSNN,
    'spikingconn7bfsnn': SpikingConn7BNetFSNN,
    'spikingfsnn': SpikingRVGGNetFSNN,
    'spikingconnfsnn': SpikingConnRVGGNetFSNN,
}

def get_model_by_name(name):
    return model_dict[name]

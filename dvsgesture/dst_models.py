import torch
import torch.nn as nn
import torch.nn.functional as F
from spikingjelly.activation_based import layer, neuron, surrogate
from connecting_neuron import SpikeParaConnLIFNode

def convrelu1x1(in_channels, out_channels):
    return nn.Sequential(
        layer.Conv2d(in_channels, out_channels, kernel_size=1, padding=0, stride=1, bias=False),
        layer.BatchNorm2d(out_channels),
        nn.ReLU()
    )

def conv1x1(in_channels, out_channels):
    return nn.Sequential(
            layer.Conv2d(in_channels, out_channels, kernel_size=1, padding=0, stride=1, bias=False),
            )

def conv3x3(in_channels, stride):
    return nn.Sequential(
        layer.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, stride = stride, bias=False, groups=in_channels),
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
        self.abn = layer.BatchNorm2d(out_channels)
        if not self.identity:
            self.aac = []#[layer.BatchNorm2d(in_channels)]
            #if stride != 1:
            self.aac.append(conv3x3(in_channels, stride))
            #    self.aac.append(layer.MaxPool2d(stride))
            #if in_channels != out_channels:
                #self.aac.append(convrelu1x1(in_channels, out_channels))
            self.aac.append(conv1x1(in_channels, out_channels))
            #self.aac.append(layer.Conv2d(in_channels, out_channels, kernel_size=1, padding=0, stride=stride, bias=False))
            #self.aac.append(layer.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, stride=stride, bias=False))
            self.aac = nn.Sequential(*self.aac)

    def forward(self, x):
        x, y = x
        out = self.conv1x1(x) + self.bn3x3(self.conv3x3(x))
        #out = self.bn(out)
        if self.identity:
            if y is not None:
                y = y + out
                y = self.abn(y)
                y = F.relu(y)
            elif self.training:
                y = out
                y = self.abn(y)#
                y = F.relu(y)
            out = out + x
        else:
            if y is not None:
                y = self.aac(y) + out
                y = self.abn(y)
                y = F.relu(y)
            elif self.training:
                y = out
                y = self.abn(y)#
                y = F.relu(y)
        out = self.bn(out)
        return self.sn(out), y
    
class ConnBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride):
        super(ConnBlock, self).__init__()
        self.identity = stride == 1 and in_channels == out_channels
        self.conv3x3 = layer.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, stride=stride, bias=False)
        self.bn3x3 = layer.BatchNorm2d(out_channels)
        self.conv1x1 = layer.Conv2d(in_channels, out_channels, kernel_size=1, padding=0, stride=stride, bias=False)
        self.bn = layer.BatchNorm2d(out_channels)
        self.abn = layer.BatchNorm2d(out_channels)
        if self.identity:
            self.sn = SpikeParaConnLIFNode(init_tau=2.0, detach_reset=True, surrogate_function=surrogate.ATan())
        else:
            self.sn = neuron.ParametricLIFNode(init_tau=2.0, detach_reset=True, surrogate_function=surrogate.ATan())
            self.aac = []
            if stride != 1:
                self.aac.append(conv3x3(in_channels, stride))
            #    self.aac.append(layer.MaxPool2d(stride))
            #if in_channels != out_channels:
            self.aac.append(convrelu1x1(in_channels, out_channels))
            self.aac = nn.Sequential(*self.aac)

    def forward(self, x):
        x, y = x
        out = self.bn(self.conv1x1(x) + self.bn3x3(self.conv3x3(x)))
        #out = self.conv1x1(x) + self.bn3x3(self.conv3x3(x))
        if self.identity:
            #out = self.sn(out,x)
            if y is not None:
                #y = F.relu(self.abn(y + out))
                #y = self.abn(y+out)
                y = y + out
            elif self.training:#
                y = out
                #y = self.abn(out)
                #y = F.relu(self.abn(out))#
            #out = self.bn(out)
            out = self.sn(out, x)
        else:
            #out = self.sn(out)
            if y is not None:
                #y = F.relu(self.abn(self.aac(y) + out))
                #y = self.abn(self.aac(y) + out)
                y = self.aac(y) + out
            elif self.training:#
                #y = F.relu(self.abn(out))#
                #y = self.abn(out)
                y = out
            #out = self.bn(out)
            out = self.sn(out)
        return out, y
    
class MaxPoolBlock(nn.Module):
    def __init__(self, stride):
        super(MaxPoolBlock, self).__init__()
        self.mp = layer.MaxPool2d(stride)
    def forward(self, x):
        x, y = x
        x = self.mp(x)
        if y is not None:
            y = self.mp(y)
        return x, y

class Spike7BRVGGNet(nn.Module):
    def __init__(self, cfg_dict, num_classes):
        super(Spike7BRVGGNet, self).__init__()
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

        self.avgpool = layer.AdaptiveAvgPool2d((1,1))

        self.flatten = nn.Flatten(2)


        self.out = layer.Linear(in_channels, num_classes, bias=True)
        self.acc = layer.Linear(in_channels, num_classes, bias=True)

    def _build_layer(self, in_channels, layer_dict):
        channels = layer_dict['channels']
        stride = layer_dict['stride']
        convs = nn.Sequential()
        convs.append(self.block(in_channels, channels, 1))
        for _ in range(layer_dict['num_blocks'] - 1):
            convs.append(self.block(channels, channels, 1))
        if stride > 1:
            convs.append(MaxPoolBlock(stride))
        return convs
    
    def forward(self, x: torch.Tensor):
        x = x.permute(1,0,2,3,4)
        y = None
        x,y = self.in_layer((x,y))
        if self.training and y is None:
            y = x
        x,y = self.convs((x,y))
        x = self.avgpool(x)
        x = self.flatten(x)
        if y is not None:
            y = self.avgpool(y)
            y = self.flatten(y)
            y = self.acc(y.mean(0))
        return self.out(x.mean(0)), y

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
        self.avgpool = layer.AdaptiveAvgPool2d((1,1))
        self.flatten = nn.Flatten(2)

        self.out = layer.Linear(in_channels, num_classes, bias=True)
        self.acc = layer.Linear(in_channels, num_classes, bias=True)

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
        y = None
        x,y = self.in_layer((x,y))
        if self.training and y is None:
            y = x
        x,y = self.convs((x,y))
        x = self.avgpool(x)
        x = self.flatten(x)
        if y is not None:
            y = self.avgpool(y)
            y = self.flatten(y)
            y = self.acc(y.mean(0))
        return self.out(x.mean(0)), y
    
def Spiking7BNetFSNN(num_classes):
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

def SpikingConn7BNetFSNN(num_classes):
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

def SpikingRVGGNetFSNN(num_classes):
    cfg_dict = {
        'layers': [
            {'channels': 48, 'num_blocks': 3, 'stride': 1},#64->32
            {'channels': 48, 'num_blocks': 3, 'stride': 2},#32->16
            {'channels': 48, 'num_blocks': 3, 'stride': 2},#16->8
            {'channels': 96, 'num_blocks': 3, 'stride': 2},#4->2
            {'channels': 96, 'num_blocks': 3, 'stride': 2},
            {'channels': 96, 'num_blocks': 3, 'stride': 2},
            {'channels': 128, 'num_blocks': 3, 'stride': 2},
        ],
        'block_type': 'spike',
    }
    return SpikeRVGGNet(cfg_dict, num_classes)

def SpikingConnRVGGNetFSNN(num_classes):
    cfg_dict = {
        'layers': [
            {'channels': 64, 'num_blocks': 1, 'stride': 1},#128->64
            {'channels': 64, 'num_blocks': 7, 'stride': 2},#64->32
            {'channels': 64, 'num_blocks': 6, 'stride': 2},#32->16
            {'channels': 64, 'num_blocks': 5, 'stride': 2},#16->8
            {'channels': 64, 'num_blocks': 1, 'stride': 2},#8->4
           # {'channels': 48, 'num_blocks': 3, 'stride': 2},
            #{'channels': 64, 'num_blocks': 3, 'stride': 2},
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

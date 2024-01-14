import torch
import torch.nn as nn
import torch.nn.functional as F
from spikingjelly.activation_based import layer, neuron, surrogate
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

class SpikeRepVGGBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride, groups=1):
        super(SpikeRepVGGBlock, self).__init__()
        self.identity = stride == 1 and in_channels == out_channels
        self.conv3x3 = layer.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=stride, groups=groups,padding=1, bias=False)
        self.bn3x3 = layer.BatchNorm2d(out_channels)
        self.conv1x1 = layer.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=stride, groups=groups, padding=0, bias=False)
        self.bn = layer.BatchNorm2d(out_channels)
        self.sn = neuron.ParametricLIFNode(v_threshold=1.0, detach_reset=True, surrogate_function=surrogate.ATan())
        if not self.identity:
            self.aac = convrelupxp(in_channels, out_channels, stride)

    def forward(self, x):
        x, y = x
        out = self.bn(self.conv1x1(x) + self.bn3x3(self.conv3x3(x)))
        if self.identity:
            if y is not None:
                y = y + out
            elif self.training:
                y = out
            out = out + x
        else:
            if y is not None:
                y = self.aac(y) + out
            elif self.training:
                y = out
        return self.sn(out), y
    
class SpikeConnRepVGGBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride, groups=1):
        super(SpikeConnRepVGGBlock, self).__init__()
        self.identity = stride == 1 and in_channels == out_channels

        self.conv3x3 = layer.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=stride, groups=groups, padding=1, bias=False)
        self.bn3x3 = layer.BatchNorm2d(out_channels)
        self.conv1x1 = layer.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=stride, groups=groups, padding=0, bias=False)
        self.bn = layer.BatchNorm2d(out_channels)
        if self.identity:
            self.sn = ParaConnLIFNode(v_threshold=1.0, detach_reset=True, surrogate_function=surrogate.ATan())
        else:
            self.aac = convrelupxp(in_channels, out_channels, stride)
            self.sn = neuron.ParametricLIFNode(v_threshold=1.0, detach_reset=True, surrogate_function=surrogate.ATan())
            
    def forward(self, x):
        x, y = x
        out = self.bn(self.conv1x1(x) + self.bn3x3(self.conv3x3(x)))
        if self.identity:
            if y is not None:
                y = y + out
            elif self.training:
                y = out
            out = self.sn(out, x)
        else:
            if y is not None:
                y = self.aac(y) + out
            elif self.training:
                y = out
            out = self.sn(out)
        return out, y

class SRepVGG(nn.Module):
    def __init__(self, cfg_dict, num_classes):
        super(SRepVGG, self).__init__()
        if cfg_dict['block_type'] == 'spike':
            self.block = SpikeRepVGGBlock
        elif cfg_dict['block_type'] == 'spike_connecting':
            self.block = SpikeConnRepVGGBlock
        else:
            raise NotImplementedError
        
        self.override_groups_map = cfg_dict.get('override_groups_map', {})
        self.cur_layer_idx=0
        
        in_channels=3
        layer_list = cfg_dict['layers']
        convs = nn.Sequential()
        for layer_dict in layer_list:
            convs.extend(self._build_layer(in_channels, layer_dict))
            in_channels = layer_dict['channels']
        self.convs = convs

        self.avgpool = layer.AdaptiveAvgPool2d((1,1))
        self.flatten = nn.Flatten(2)
        self.fc = layer.Linear(in_channels, num_classes)
        self.aac = layer.Linear(in_channels, num_classes)

    def _build_layer(self, in_channels, layer_dict):
        channels = layer_dict['channels']
        stride = layer_dict['stride']
        convs = nn.Sequential()
        convs.append(self.block(in_channels, channels, stride, groups = self.override_groups_map.get(self.cur_layer_idx, 1)))
        self.cur_layer_idx += 1
        for _ in range(layer_dict['num_blocks'] - 1):
            convs.append(self.block(channels, channels, 1, groups = self.override_groups_map.get(self.cur_layer_idx, 1)))
            self.cur_layer_idx += 1
        return convs
    
    def forward(self, x: torch.Tensor):
        x,y = self.convs((x,None))
        x = self.avgpool(x)
        x = self.flatten(x)
        x = self.fc(x.mean(0))
        if y is not None:
            y = self.avgpool(y)
            y = self.flatten(y)
            y = self.aac(y.mean(0))
        return x,y

def SRepVGG_N0(num_classes, block_type='spike_connecting'):
    cfg_dict = {
        'block_type': block_type,
        'layers': [
            {'channels': 64, 'num_blocks': 2, 'stride': 2},
            {'channels': 64, 'num_blocks': 2, 'stride': 2},
            {'channels': 64, 'num_blocks': 2, 'stride': 2},
            {'channels': 64, 'num_blocks': 2, 'stride': 2},
            {'channels': 64, 'num_blocks': 2, 'stride': 2},
        ],
    }
    return SRepVGG(cfg_dict, num_classes)

def SRepVGG_N1(num_classes, block_type='spike_connecting'):
    cfg_dict = {
        'block_type': block_type,
        'layers': [
            {'channels': 64, 'num_blocks': 1, 'stride': 1},
            {'channels': 64, 'num_blocks': 6, 'stride': 2},
            {'channels': 64, 'num_blocks': 6, 'stride': 2},
            {'channels': 128, 'num_blocks': 5, 'stride': 2},
            {'channels': 128, 'num_blocks': 1, 'stride': 2},
        ],
    }
    return SRepVGG(cfg_dict, num_classes)

model_dict = {
    'SRepVGG_N0': SRepVGG_N0,
    'SRepVGG_N1': SRepVGG_N1,
}

def get_model_by_name(name):
    return model_dict[name]

import torch
import torch.nn as nn
from spikingjelly.activation_based import layer, neuron, surrogate
from connecting_functions import ConnectingFunction

def conv_bn(in_channels, out_channels, kernel_size, stride, padding, padding_mode='zeros', groups=1):
    result = nn.Sequential()
    result.add_module('conv', layer.Conv2d(in_channels=in_channels, out_channels=out_channels,
                                                  kernel_size=kernel_size, stride=stride, padding=padding, padding_mode=padding_mode, groups=groups, bias=False))
    result.add_module('bn', layer.BatchNorm2d(num_features=out_channels))
    return result

class SEWRRepVGGBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, cnf=None):
        super(SEWRRepVGGBlock, self).__init__()
        self.conv3x3 = conv_bn(in_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.conv1x1 = conv_bn(in_channels, out_channels, kernel_size=1, stride=stride, padding=0)
        self.sn = neuron.ParametricLIFNode(init_tau=2.0, detach_reset=True, surrogate_function=surrogate.ATan())
        self.cnf = None if cnf is None else ConnectingFunction(cnf)

    def forward(self, x):
        out = self.sn(self.conv3x3(x) + self.conv1x1(x))
        if self.cnf is not None:
            out = self.cnf(x,out)
        return out

class QASEWRepVGGBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride, cnf=None):
        super(QASEWRepVGGBlock, self).__init__()
        self.conv3x3 = conv_bn(in_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.conv1x1 = layer.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, padding=0, bias=False)
        self.bn = layer.BatchNorm2d(out_channels)
        self.sn = neuron.ParametricLIFNode(init_tau=2.0, detach_reset=True, surrogate_function=surrogate.ATan())
        self.cnf = None if cnf is None else ConnectingFunction(cnf)

    def forward(self, x):
        out = self.sn(self.bn(self.conv3x3(x) + self.conv1x1(x)))
        if self.cnf is not None:
            out = self.cnf(x,out)
        return out

def get_block(block):
    if block == 'sew':
        return SEWRRepVGGBlock
    elif block == 'qasew':
        return QASEWRepVGGBlock
    else:
        raise NotImplementedError

class SpikingRepVGGN(nn.Module):
    def __init__(self, layer_list, num_classes, cnf=None):
        super(SpikingRepVGGN, self).__init__()
        in_channels = 2
        conv = []
        for cfg_dict in layer_list:
            channels = cfg_dict['channels']
            if 'out_channels' in cfg_dict:
                out_channels = cfg_dict['out_channels']
            else:
                out_channels = in_channels

            block = get_block(cfg_dict['block'])

            if channels != in_channels:
                conv.append(block(in_channels, channels, stride=1, cnf=cnf))
            in_channels = channels

            for _ in range(cfg_dict['num_blocks']):
                conv.append(block(in_channels, out_channels, stride=1, cnf=cnf))
            
            if 'k_pool' in cfg_dict:
                k_pool = cfg_dict['k_pool']
                conv.append(nn.MaxPool2d(k_pool,k_pool))

        conv.append(nn.Flatten(2))
        self.conv = nn.Sequential(*conv)

        with torch.no_grad():
            x = torch.zeros([1,1,128,128])
            for m in self.conv.modules():
                if isinstance(m, layer.MaxPool2d):
                    x = m(x)
            out_features = x.numel() * in_channels
        
        self.out = layer.Linear(out_features, num_classes, bias=True)

    def forward(self, x: torch.Tensor):
        x = x.permute(1,0,2,3,4)
        x = self.conv(x)
        return self.out(x.mean(0))

def SEWRepVGG(num_classes, cnf):
    layer_list = [
        {'channels': 64, 'out_channels': 64, 'num_blocks': 2, 'block': 'sew', 'k_pool': 2},
        {'channels': 64, 'out_channels': 64, 'num_blocks': 2, 'block': 'sew', 'k_pool': 2},
        {'channels': 64, 'out_channels': 64, 'num_blocks': 2, 'block': 'sew', 'k_pool': 2},
        {'channels': 64, 'out_channels': 64, 'num_blocks': 2, 'block': 'sew', 'k_pool': 2},
        {'channels': 128, 'out_channels': 128, 'num_blocks': 2, 'block': 'sew', 'k_pool': 2},
        {'channels': 128, 'out_channels': 128, 'num_blocks': 2, 'block': 'sew', 'k_pool': 2},
        {'channels': 128, 'out_channels': 128, 'num_blocks': 2, 'block': 'sew', 'k_pool': 2},
    ]  
    return SpikingRepVGGN(layer_list, num_classes, cnf)

def QASEWRepVGG(num_classes, cnf):
    layer_list = [
        {'channels': 64, 'out_channels': 64, 'num_blocks': 2, 'block': 'qasew', 'k_pool': 2},
        {'channels': 64, 'out_channels': 64, 'num_blocks': 2, 'block': 'qasew', 'k_pool': 2},
        {'channels': 64, 'out_channels': 64, 'num_blocks': 2, 'block': 'qasew', 'k_pool': 2},
        {'channels': 64, 'out_channels': 64, 'num_blocks': 2, 'block': 'qasew', 'k_pool': 2},
        {'channels': 128, 'out_channels': 128, 'num_blocks': 2, 'block': 'qasew', 'k_pool': 2},
        {'channels': 128, 'out_channels': 128, 'num_blocks': 2, 'block': 'qasew', 'k_pool': 2},
        {'channels': 128, 'out_channels': 128, 'num_blocks': 2, 'block': 'qasew', 'k_pool': 2},
    ]
    return SpikingRepVGGN(layer_list, num_classes, cnf)

def SEWRepVGGH(num_classes, cnf):
    layer_list = [
        {'channels': 64, 'out_channels': 64, 'num_blocks': 1, 'block': 'sew', 'k_pool': 2},
        {'channels': 64, 'out_channels': 64, 'num_blocks': 1, 'block': 'sew', 'k_pool': 2},
        {'channels': 64, 'out_channels': 64, 'num_blocks': 1, 'block': 'sew', 'k_pool': 2},
        {'channels': 64, 'out_channels': 64, 'num_blocks': 1, 'block': 'sew', 'k_pool': 2},
        {'channels': 128, 'out_channels': 128, 'num_blocks': 1, 'block': 'sew', 'k_pool': 2},
        {'channels': 128, 'out_channels': 128, 'num_blocks': 1, 'block': 'sew', 'k_pool': 2},
        {'channels': 128, 'out_channels': 128, 'num_blocks': 1, 'block': 'sew', 'k_pool': 2},
    ]  
    return SpikingRepVGGN(layer_list, num_classes, cnf)

def QASEWRepVGGH(num_classes, cnf):
    layer_list = [
        {'channels': 64, 'out_channels': 64, 'num_blocks': 1, 'block': 'qasew', 'k_pool': 2},
        {'channels': 64, 'out_channels': 64, 'num_blocks': 1, 'block': 'qasew', 'k_pool': 2},
        {'channels': 64, 'out_channels': 64, 'num_blocks': 1, 'block': 'qasew', 'k_pool': 2},
        {'channels': 64, 'out_channels': 64, 'num_blocks': 1, 'block': 'qasew', 'k_pool': 2},
        {'channels': 128, 'out_channels': 128, 'num_blocks': 1, 'block': 'qasew', 'k_pool': 2},
        {'channels': 128, 'out_channels': 128, 'num_blocks': 1, 'block': 'qasew', 'k_pool': 2},
        {'channels': 128, 'out_channels': 128, 'num_blocks': 1, 'block': 'qasew', 'k_pool': 2},
    ]
    return SpikingRepVGGN(layer_list, num_classes, cnf)

model_dict = {
    'sew': SEWRepVGG,
    'qasew': QASEWRepVGG,
    'sewh': SEWRepVGGH,
    'qasewh': QASEWRepVGGH,
}

def get_model_by_name(name):
    return model_dict[name]
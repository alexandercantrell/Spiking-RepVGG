import torch
import torch.nn as nn
from spikingjelly.activation_based import layer, neuron, surrogate
from connecting_functions import ConnectingFunction
from connecting_neuron import ParaConnLIFNode, SpikeParaConnLIFNode

def conv3x3(in_channels, out_channels):
    return nn.Sequential(
        layer.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, stride=1, bias=False),
        layer.BatchNorm2d(out_channels),
        neuron.ParametricLIFNode(init_tau=2.0, detach_reset=True, surrogate_function=surrogate.ATan())
    )

def conv1x1(in_channels, out_channels):
    return nn.Sequential(
        layer.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, bias=False),
        layer.BatchNorm2d(out_channels),
        neuron.ParametricLIFNode(init_tau=2.0, detach_reset=True, surrogate_function=surrogate.ATan())
    )

def convrelu3x3(in_channels, out_channels):
    return nn.Sequential(
        layer.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, stride=1, bias=False),
        layer.BatchNorm2d(out_channels),
        nn.ReLU()
    )

def convrelu1x1(in_channels, out_channels):
    return nn.Sequential(
        layer.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, bias=False),
        layer.BatchNorm2d(out_channels),
        nn.ReLU()
    )

class UpBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1):
        super(UpBlock, self).__init__()
        if kernel_size == 1:
            self.conv = conv1x1(in_channels, out_channels)
            self.aac = convrelu1x1(in_channels, out_channels)
        elif kernel_size == 3:
            self.conv = conv3x3(in_channels, out_channels)
            self.aac = convrelu3x3(in_channels, out_channels)
        else:
            raise NotImplementedError
    def forward(self, x: torch.Tensor, y: torch.Tensor = None):
        x = self.conv(x)
        if y is not None:
            y = self.aac(y) + x
        return x, y

class SEWBlock(nn.Module):
    def __init__(self, in_channels, mid_channels, cnf=None):
        super(SEWBlock, self).__init__()
        self.cnf = ConnectingFunction(cnf)
        self.conv = nn.Sequential(
            conv3x3(in_channels, mid_channels),
            conv3x3(mid_channels, in_channels),
        )

    def forward(self, x: torch.Tensor, y: torch.Tensor = None):
        out = self.conv(x)
        if y is not None:
            y = y + out
        out = self.cnf(out, x)
        return out, y
    
class ConnBlock(nn.Module):
    def __init__(self, in_channels, mid_channels):
        super(ConnBlock, self).__init__()
        self.conv = nn.Sequential(
            conv3x3(in_channels, mid_channels),
            layer.Conv2d(mid_channels, in_channels, kernel_size=3, padding=1, stride=1, bias=False),
            layer.BatchNorm2d(in_channels),
        )
        self.cn = ParaConnLIFNode(init_tau=2.0, detach_reset=True, surrogate_function=surrogate.ATan())

    def forward(self, x: torch.Tensor, y: torch.Tensor = None):
        out = self.conv(x)
        if y is not None:
            y = y + out
        out = self.cn(out, x)
        return out, y
    
class SpikeConnBlock(nn.Module):
    def __init__(self, in_channels, mid_channels):
        super(SpikeConnBlock, self).__init__()
        self.conv = nn.Sequential(
            conv3x3(in_channels, mid_channels),
            layer.Conv2d(mid_channels, in_channels, kernel_size=3, padding=1, stride=1, bias=False),
            layer.BatchNorm2d(in_channels),
        )
        self.cn = SpikeParaConnLIFNode(init_tau=2.0, detach_reset=True, surrogate_function=surrogate.ATan())
    def forward(self, x:torch.Tensor, y:torch.Tensor=None):
        out = self.conv(x)
        if y is not None:
            y = y + out
        out = self.cn(out, x)
        return out, y
    
class SpikingRVGGBlock(nn.Module):
    def __init__(self, in_channels, mid_channels):
        super(SpikingRVGGBlock, self).__init__()
        self.conv1_1x1 = layer.Conv2d(in_channels, mid_channels, kernel_size=1, padding=0, stride=1, bias=False)
        self.conv1_3x3 = layer.Conv2d(mid_channels, mid_channels, kernel_size=3, padding=1, stride=1, bias=False)
        self.bn1 = layer.BatchNorm2d(mid_channels)
        self.bn1_3x3 = layer.BatchNorm2d(mid_channels)
        self.sn1 = neuron.ParametricLIFNode(init_tau=2.0, detach_reset=True, surrogate_function=surrogate.ATan())
        self.conv2_1x1 = layer.Conv2d(mid_channels, in_channels, kernel_size=1, padding=0, stride=1, bias=False)
        self.conv2_3x3 = layer.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, stride=1, bias=False)
        self.bn2 = layer.BatchNorm2d(in_channels)
        self.bn2_3x3 = layer.BatchNorm2d(in_channels)
        self.sn2 = neuron.ParametricLIFNode(init_tau=2.0, detach_reset=True, surrogate_function=surrogate.ATan())

    def forward(self, x: torch.Tensor, y: torch.Tensor = None):
        out = self.bn1(self.conv1_1x1(x) + self.bn1_3x3(self.conv1_3x3(x)))
        if y is not None:
            y = y + out
        x = self.sn1(out + x)
        out = self.bn2(self.conv2_1x1(x) + self.bn2_3x3(self.conv2_3x3(x)))
        if y is not None:
            y = y + out
        out = self.sn2(out + x)
        return out, y
    
class SpikingConnRVGGBlock(nn.Module):
    def __init__(self, in_channels, mid_channels):
        super(SpikingConnRVGGBlock, self).__init__()
        self.conv1_1x1 = layer.Conv2d(in_channels, mid_channels, kernel_size=1, padding=0, stride=1, bias=False)
        self.conv1_3x3 = layer.Conv2d(mid_channels, mid_channels, kernel_size=3, padding=1, stride=1, bias=False)
        self.bn1 = layer.BatchNorm2d(mid_channels)
        self.bn1_3x3 = layer.BatchNorm2d(mid_channels)
        self.sn1 = SpikeParaConnLIFNode(init_tau=2.0, detach_reset=True, surrogate_function=surrogate.ATan())
        self.conv2_1x1 = layer.Conv2d(mid_channels, in_channels, kernel_size=1, padding=0, stride=1, bias=False)
        self.conv2_3x3 = layer.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, stride=1, bias=False)
        self.bn2 = layer.BatchNorm2d(in_channels)
        self.bn2_3x3 = layer.BatchNorm2d(in_channels)
        self.sn2 = SpikeParaConnLIFNode(init_tau=2.0, detach_reset=True, surrogate_function=surrogate.ATan())

    def forward(self, x:torch.Tensor, y: torch.Tensor = None):
        out = self.bn1(self.conv1_1x1(x) + self.bn1_3x3(self.conv1_3x3(x)))
        if y is not None:
            y = y + out
        x = self.sn1(out,x)
        out = self.bn2(self.conv2_1x1(x) + self.bn2_3x3(self.conv2_3x3(x)))
        if y is not None:
            y = y + out
        out = self.sn2(out, x)
        return out, y
    

class BasicBlock(nn.Module):
    def __init__(self, in_channels, mid_channels):
        super(BasicBlock, self).__init__()
        self.conv = nn.Sequential(
            conv3x3(in_channels, mid_channels),
            layer.Conv2d(mid_channels, in_channels, kernel_size=3, padding=1, stride=1, bias=False),
            layer.BatchNorm2d(in_channels),
        )
        self.sn = neuron.ParametricLIFNode(init_tau=2.0, detach_reset=True, surrogate_function=surrogate.ATan())

    def forward(self, x: torch.Tensor, y: torch.Tensor = None):
        out = self.conv(x)
        if y is not None:
            y = y + out
        out = self.sn(out + x)
        return out, y

class DSTResNetN(nn.Module):
    def __init__(self, layer_list, num_classes, cnf=None, dsnn=False):
        super(DSTResNetN, self).__init__()
        self.dsnn=dsnn

        in_channels = 2
        channels = layer_list[0]['channels']
        self.in_conv = UpBlock(in_channels,channels,layer_list[0]['up_kernel_size'])
        in_channels = channels

        conv = nn.ModuleList()

        for cfg_dict in layer_list:
            channels = cfg_dict['channels']

            if 'mid_channels' in cfg_dict:
                mid_channels = cfg_dict['mid_channels']
            else:
                mid_channels = channels

            if in_channels != channels:
                conv.append(UpBlock(in_channels, channels, cfg_dict['up_kernel_size']))

            in_channels = channels

            if 'num_blocks' in cfg_dict:
                num_blocks = cfg_dict['num_blocks']
                if cfg_dict['block_type'] == 'sew':
                    for _ in range(num_blocks):
                        conv.append(SEWBlock(in_channels, mid_channels, cnf=cnf))
                elif cfg_dict['block_type'] == 'conn':
                    for _ in range(num_blocks):
                        conv.append(ConnBlock(in_channels, mid_channels))
                elif cfg_dict['block_type'] == 'basic':
                    for _ in range(num_blocks):
                        conv.append(BasicBlock(in_channels, mid_channels))
                elif cfg_dict['block_type'] == 'spikeconn':
                    for _ in range(num_blocks):
                        conv.append(SpikeConnBlock(in_channels, mid_channels))
                elif cfg_dict['block_type'] == 'spikingrvgg':
                    for _ in range(num_blocks):
                        conv.append(SpikingRVGGBlock(in_channels, mid_channels))
                elif cfg_dict['block_type'] == 'spikingconnrvgg':
                    for _ in range(num_blocks):
                        conv.append(SpikingConnRVGGBlock(in_channels,mid_channels))
                else:
                    raise NotImplementedError

            if 'k_pool' in cfg_dict:
                k_pool = cfg_dict['k_pool']
                conv.append(layer.MaxPool2d(k_pool, k_pool))                   

        conv.append(nn.Flatten(2))

        self.conv = conv

        with torch.no_grad():
            x = torch.zeros([1, 1, 128, 128])
            for m in self.conv.modules():
                if isinstance(m, layer.MaxPool2d):                   
                    x = m(x) 
            out_features = x.numel() * in_channels                          

        self.out = layer.Linear(out_features, num_classes, bias=True)
        self.aac = layer.Linear(out_features, num_classes, bias=True)

    def forward(self, x: torch.Tensor):
        x = x.permute(1,0,2,3,4)
        x, y = self.in_conv(x)
        if self.training or self.dsnn:
            y = x
        
        for m in self.conv:
            if isinstance(m, layer.MaxPool2d) or isinstance(m, nn.Flatten):   
                x = m(x)
                if y is not None:
                    y = m(y)
            else:
                x, y = m(x, y)
        x = self.out(x.mean(0))
        if y is not None:
            y = self.aac(y.mean(0))
        return x, y
    
def SEWFSNN(num_classes, cnf):
    layer_list = [
        {'channels': 64, 'up_kernel_size': 1, 'mid_channels': 64, 'num_blocks': 1, 'block_type': 'sew', 'k_pool': 2},
        {'channels': 64, 'up_kernel_size': 1, 'mid_channels': 64, 'num_blocks': 1, 'block_type': 'sew', 'k_pool': 2},
        {'channels': 64, 'up_kernel_size': 1, 'mid_channels': 64, 'num_blocks': 1, 'block_type': 'sew', 'k_pool': 2},
        {'channels': 64, 'up_kernel_size': 1, 'mid_channels': 64, 'num_blocks': 1, 'block_type': 'sew', 'k_pool': 2},
        {'channels': 128, 'up_kernel_size': 1, 'mid_channels': 128, 'num_blocks': 1, 'block_type': 'sew', 'k_pool': 2},
        {'channels': 128, 'up_kernel_size': 1, 'mid_channels': 128, 'num_blocks': 1, 'block_type': 'sew', 'k_pool': 2},
        {'channels': 128, 'up_kernel_size': 1, 'mid_channels': 128, 'num_blocks': 1, 'block_type': 'sew', 'k_pool': 2},
    ]
    return DSTResNetN(layer_list, num_classes, cnf)

def SpikingFSNN(num_classes, *args, **kwargs):
    layer_list = [
        {'channels': 64, 'up_kernel_size': 1, 'mid_channels': 64, 'num_blocks': 1, 'block_type': 'basic', 'k_pool': 2},
        {'channels': 64, 'up_kernel_size': 1, 'mid_channels': 64, 'num_blocks': 1, 'block_type': 'basic', 'k_pool': 2},
        {'channels': 64, 'up_kernel_size': 1, 'mid_channels': 64, 'num_blocks': 1, 'block_type': 'basic', 'k_pool': 2},
        {'channels': 64, 'up_kernel_size': 1, 'mid_channels': 64, 'num_blocks': 1, 'block_type': 'basic', 'k_pool': 2},
        {'channels': 128, 'up_kernel_size': 1, 'mid_channels': 128, 'num_blocks': 1, 'block_type': 'basic', 'k_pool': 2},
        {'channels': 128, 'up_kernel_size': 1, 'mid_channels': 128, 'num_blocks': 1, 'block_type': 'basic', 'k_pool': 2},
        {'channels': 128, 'up_kernel_size': 1, 'mid_channels': 128, 'num_blocks': 1, 'block_type': 'basic', 'k_pool': 2},
    ]
    return DSTResNetN(layer_list, num_classes)

def ConnFSNN(num_classes, *args, **kwargs):
    layer_list = [
        {'channels': 64, 'up_kernel_size': 1, 'mid_channels': 64, 'num_blocks': 1, 'block_type': 'conn', 'k_pool': 2},
        {'channels': 64, 'up_kernel_size': 1, 'mid_channels': 64, 'num_blocks': 1, 'block_type': 'conn', 'k_pool': 2},
        {'channels': 64, 'up_kernel_size': 1, 'mid_channels': 64, 'num_blocks': 1, 'block_type': 'conn', 'k_pool': 2},
        {'channels': 64, 'up_kernel_size': 1, 'mid_channels': 64, 'num_blocks': 1, 'block_type': 'conn', 'k_pool': 2},
        {'channels': 128, 'up_kernel_size': 1, 'mid_channels': 128, 'num_blocks': 1, 'block_type': 'conn', 'k_pool': 2},
        {'channels': 128, 'up_kernel_size': 1, 'mid_channels': 128, 'num_blocks': 1, 'block_type': 'conn', 'k_pool': 2},
        {'channels': 128, 'up_kernel_size': 1, 'mid_channels': 128, 'num_blocks': 1, 'block_type': 'conn', 'k_pool': 2},
    ]
    return DSTResNetN(layer_list, num_classes)

def SpikeConnResNetFSNN(num_classes, *args, **kwargs):
    layer_list = [
        {'channels': 64, 'up_kernel_size': 1, 'mid_channels': 64, 'num_blocks': 1, 'block_type': 'spikeconn', 'k_pool': 2},
        {'channels': 64, 'up_kernel_size': 1, 'mid_channels': 64, 'num_blocks': 1, 'block_type': 'spikeconn', 'k_pool': 2},
        {'channels': 64, 'up_kernel_size': 1, 'mid_channels': 64, 'num_blocks': 1, 'block_type': 'spikeconn', 'k_pool': 2},
        {'channels': 64, 'up_kernel_size': 1, 'mid_channels': 64, 'num_blocks': 1, 'block_type': 'spikeconn', 'k_pool': 2},
        {'channels': 128, 'up_kernel_size': 1, 'mid_channels': 128, 'num_blocks': 1, 'block_type': 'spikeconn', 'k_pool': 2},
        {'channels': 128, 'up_kernel_size': 1, 'mid_channels': 128, 'num_blocks': 1, 'block_type': 'spikeconn', 'k_pool': 2},
        {'channels': 128, 'up_kernel_size': 1, 'mid_channels': 128, 'num_blocks': 1, 'block_type': 'spikeconn', 'k_pool': 2},
    ]
    return DSTResNetN(layer_list, num_classes)

def SpikingRVGGFSNN(num_classes, *args, **kwargs):
    layer_list = [
        {'channels': 64, 'up_kernel_size': 3, 'mid_channels': 64, 'num_blocks': 1, 'block_type': 'spikingrvgg', 'k_pool': 2},
        {'channels': 64, 'up_kernel_size': 3, 'mid_channels': 64, 'num_blocks': 1, 'block_type': 'spikingrvgg', 'k_pool': 2},
        {'channels': 64, 'up_kernel_size': 3, 'mid_channels': 64, 'num_blocks': 1, 'block_type': 'spikingrvgg', 'k_pool': 2},
        {'channels': 64, 'up_kernel_size': 3, 'mid_channels': 64, 'num_blocks': 1, 'block_type': 'spikingrvgg', 'k_pool': 2},
        {'channels': 128, 'up_kernel_size': 3, 'mid_channels': 128, 'num_blocks': 1, 'block_type': 'spikingrvgg', 'k_pool': 2},
        {'channels': 128, 'up_kernel_size': 3, 'mid_channels': 128, 'num_blocks': 1, 'block_type': 'spikingrvgg', 'k_pool': 2},
        {'channels': 128, 'up_kernel_size': 3, 'mid_channels': 128, 'num_blocks': 1, 'block_type': 'spikingrvgg', 'k_pool': 2},
    ]
    return DSTResNetN(layer_list, num_classes)

def SpikingConnRVGGFSNN(num_classes, *args, **kwargs):
    layer_list = [
        {'channels': 64, 'up_kernel_size': 3, 'mid_channels': 64, 'num_blocks': 1, 'block_type': 'spikingconnrvgg', 'k_pool': 2},
        {'channels': 64, 'up_kernel_size': 3, 'mid_channels': 64, 'num_blocks': 1, 'block_type': 'spikingconnrvgg', 'k_pool': 2},
        {'channels': 64, 'up_kernel_size': 3, 'mid_channels': 64, 'num_blocks': 1, 'block_type': 'spikingconnrvgg', 'k_pool': 2},
        {'channels': 64, 'up_kernel_size': 3, 'mid_channels': 64, 'num_blocks': 1, 'block_type': 'spikingconnrvgg', 'k_pool': 2},
        {'channels': 128, 'up_kernel_size': 3, 'mid_channels': 128, 'num_blocks': 1, 'block_type': 'spikingconnrvgg', 'k_pool': 2},
        {'channels': 128, 'up_kernel_size': 3, 'mid_channels': 128, 'num_blocks': 1, 'block_type': 'spikingconnrvgg', 'k_pool': 2},
        {'channels': 128, 'up_kernel_size': 3, 'mid_channels': 128, 'num_blocks': 1, 'block_type': 'spikingconnrvgg', 'k_pool': 2},
    ]
    return DSTResNetN(layer_list, num_classes)

def SEWDSNN(num_classes, cnf):
    layer_list = [
        {'channels': 64, 'up_kernel_size': 1, 'mid_channels': 64, 'num_blocks': 1, 'block_type': 'sew', 'k_pool': 2},
        {'channels': 64, 'up_kernel_size': 1, 'mid_channels': 64, 'num_blocks': 1, 'block_type': 'sew', 'k_pool': 2},
        {'channels': 64, 'up_kernel_size': 1, 'mid_channels': 64, 'num_blocks': 1, 'block_type': 'sew', 'k_pool': 2},
        {'channels': 64, 'up_kernel_size': 1, 'mid_channels': 64, 'num_blocks': 1, 'block_type': 'sew', 'k_pool': 2},
        {'channels': 128, 'up_kernel_size': 1, 'mid_channels': 128, 'num_blocks': 1, 'block_type': 'sew', 'k_pool': 2},
        {'channels': 128, 'up_kernel_size': 1, 'mid_channels': 128, 'num_blocks': 1, 'block_type': 'sew', 'k_pool': 2},
        {'channels': 128, 'up_kernel_size': 1, 'mid_channels': 128, 'num_blocks': 1, 'block_type': 'sew', 'k_pool': 2},
    ]
    return DSTResNetN(layer_list, num_classes, cnf, dsnn=True)

def SpikingDSNN(num_classes, *args, **kwargs):
    layer_list = [
        {'channels': 64, 'up_kernel_size': 1, 'mid_channels': 64, 'num_blocks': 1, 'block_type': 'basic', 'k_pool': 2},
        {'channels': 64, 'up_kernel_size': 1, 'mid_channels': 64, 'num_blocks': 1, 'block_type': 'basic', 'k_pool': 2},
        {'channels': 64, 'up_kernel_size': 1, 'mid_channels': 64, 'num_blocks': 1, 'block_type': 'basic', 'k_pool': 2},
        {'channels': 64, 'up_kernel_size': 1, 'mid_channels': 64, 'num_blocks': 1, 'block_type': 'basic', 'k_pool': 2},
        {'channels': 128, 'up_kernel_size': 1, 'mid_channels': 128, 'num_blocks': 1, 'block_type': 'basic', 'k_pool': 2},
        {'channels': 128, 'up_kernel_size': 1, 'mid_channels': 128, 'num_blocks': 1, 'block_type': 'basic', 'k_pool': 2},
        {'channels': 128, 'up_kernel_size': 1, 'mid_channels': 128, 'num_blocks': 1, 'block_type': 'basic', 'k_pool': 2},
    ]
    return DSTResNetN(layer_list, num_classes, dsnn=True)

def ConnDSNN(num_classes, *args, **kwargs):
    layer_list = [
        {'channels': 64, 'up_kernel_size': 1, 'mid_channels': 64, 'num_blocks': 1, 'block_type': 'conn', 'k_pool': 2},
        {'channels': 64, 'up_kernel_size': 1, 'mid_channels': 64, 'num_blocks': 1, 'block_type': 'conn', 'k_pool': 2},
        {'channels': 64, 'up_kernel_size': 1, 'mid_channels': 64, 'num_blocks': 1, 'block_type': 'conn', 'k_pool': 2},
        {'channels': 64, 'up_kernel_size': 1, 'mid_channels': 64, 'num_blocks': 1, 'block_type': 'conn', 'k_pool': 2},
        {'channels': 128, 'up_kernel_size': 1, 'mid_channels': 128, 'num_blocks': 1, 'block_type': 'conn', 'k_pool': 2},
        {'channels': 128, 'up_kernel_size': 1, 'mid_channels': 128, 'num_blocks': 1, 'block_type': 'conn', 'k_pool': 2},
        {'channels': 128, 'up_kernel_size': 1, 'mid_channels': 128, 'num_blocks': 1, 'block_type': 'conn', 'k_pool': 2},
    ]
    return DSTResNetN(layer_list, num_classes, dsnn=True)

def SpikeConnResNetDSNN(num_classes, *args, **kwargs):
    layer_list = [
        {'channels': 64, 'up_kernel_size': 1, 'mid_channels': 64, 'num_blocks': 1, 'block_type': 'spikeconn', 'k_pool': 2},
        {'channels': 64, 'up_kernel_size': 1, 'mid_channels': 64, 'num_blocks': 1, 'block_type': 'spikeconn', 'k_pool': 2},
        {'channels': 64, 'up_kernel_size': 1, 'mid_channels': 64, 'num_blocks': 1, 'block_type': 'spikeconn', 'k_pool': 2},
        {'channels': 64, 'up_kernel_size': 1, 'mid_channels': 64, 'num_blocks': 1, 'block_type': 'spikeconn', 'k_pool': 2},
        {'channels': 128, 'up_kernel_size': 1, 'mid_channels': 128, 'num_blocks': 1, 'block_type': 'spikeconn', 'k_pool': 2},
        {'channels': 128, 'up_kernel_size': 1, 'mid_channels': 128, 'num_blocks': 1, 'block_type': 'spikeconn', 'k_pool': 2},
        {'channels': 128, 'up_kernel_size': 1, 'mid_channels': 128, 'num_blocks': 1, 'block_type': 'spikeconn', 'k_pool': 2},
    ]
    return DSTResNetN(layer_list, num_classes, dsnn=True)

def SpikingRVGGDSNN(num_classes, *args, **kwargs):
    layer_list = [
        {'channels': 64, 'up_kernel_size': 3, 'mid_channels': 64, 'num_blocks': 1, 'block_type': 'spikingrvgg', 'k_pool': 2},
        {'channels': 64, 'up_kernel_size': 3, 'mid_channels': 64, 'num_blocks': 1, 'block_type': 'spikingrvgg', 'k_pool': 2},
        {'channels': 64, 'up_kernel_size': 3, 'mid_channels': 64, 'num_blocks': 1, 'block_type': 'spikingrvgg', 'k_pool': 2},
        {'channels': 64, 'up_kernel_size': 3, 'mid_channels': 64, 'num_blocks': 1, 'block_type': 'spikingrvgg', 'k_pool': 2},
        {'channels': 128, 'up_kernel_size': 3, 'mid_channels': 128, 'num_blocks': 1, 'block_type': 'spikingrvgg', 'k_pool': 2},
        {'channels': 128, 'up_kernel_size': 3, 'mid_channels': 128, 'num_blocks': 1, 'block_type': 'spikingrvgg', 'k_pool': 2},
        {'channels': 128, 'up_kernel_size': 3, 'mid_channels': 128, 'num_blocks': 1, 'block_type': 'spikingrvgg', 'k_pool': 2},
    ]
    return DSTResNetN(layer_list, num_classes, dsnn=True)

def SpikingConnRVGGDSNN(num_classes, *args, **kwargs):
    layer_list = [
        {'channels': 64, 'up_kernel_size': 3, 'mid_channels': 64, 'num_blocks': 1, 'block_type': 'spikingconnrvgg', 'k_pool': 2},
        {'channels': 64, 'up_kernel_size': 3, 'mid_channels': 64, 'num_blocks': 1, 'block_type': 'spikingconnrvgg', 'k_pool': 2},
        {'channels': 64, 'up_kernel_size': 3, 'mid_channels': 64, 'num_blocks': 1, 'block_type': 'spikingconnrvgg', 'k_pool': 2},
        {'channels': 64, 'up_kernel_size': 3, 'mid_channels': 64, 'num_blocks': 1, 'block_type': 'spikingconnrvgg', 'k_pool': 2},
        {'channels': 128, 'up_kernel_size': 3, 'mid_channels': 128, 'num_blocks': 1, 'block_type': 'spikingconnrvgg', 'k_pool': 2},
        {'channels': 128, 'up_kernel_size': 3, 'mid_channels': 128, 'num_blocks': 1, 'block_type': 'spikingconnrvgg', 'k_pool': 2},
        {'channels': 128, 'up_kernel_size': 3, 'mid_channels': 128, 'num_blocks': 1, 'block_type': 'spikingconnrvgg', 'k_pool': 2},
    ]
    return DSTResNetN(layer_list, num_classes, dsnn=True)

model_dict = {
    'sewfsnn': SEWFSNN,
    'spikingfsnn': SpikingFSNN,
    'connfsnn': ConnFSNN,
    'spikeconnfsnn': SpikeConnResNetFSNN,
    'sewdsnn': SEWDSNN,
    'spikingdsnn': SpikingDSNN,
    'conndsnn': ConnDSNN,
    'spikeconndsnn': SpikeConnResNetDSNN,
    'spikingrvggfsnn': SpikingRVGGFSNN,
    'spikingrvggdsnn': SpikingRVGGDSNN,
    'spikingconnrvggfsnn': SpikingConnRVGGFSNN,
    'spikingconnrvggdsnn': SpikingConnRVGGDSNN,
}

def get_model_by_name(name):
    return model_dict[name]

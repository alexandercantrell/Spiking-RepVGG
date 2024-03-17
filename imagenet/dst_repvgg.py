import torch
import torch.nn as nn
import torch.nn.functional as F
from spikingjelly.activation_based import layer, neuron, surrogate
from connecting_neuron import ConnIFNode
from batchnorm_neuron import BNIFNode
from collections import OrderedDict

V_THRESHOLD = 1.0

def convrelupxp(in_channels, out_channels, stride=1):
    if stride != 1:
        return nn.Sequential(
            OrderedDict([
                ('stride_conv',layer.Conv2d(in_channels, in_channels, kernel_size=3, stride=stride,groups=in_channels, padding=1, bias=False)),
                ('channel_conv',layer.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, bias=False)),
                ('bn', layer.BatchNorm2d(out_channels)),
                ('relu', nn.ReLU())
            ])
        )
    else:
        return nn.Sequential(
            OrderedDict([
                ('channel_conv',layer.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, bias=False)),
                ('bn', layer.BatchNorm2d(out_channels)),
                ('relu', nn.ReLU())
            ])
        )
    
class ConversionBlock(nn.Module):
    def __init__(self, in_channels, deploy=False, set_y=True):
        super(ConversionBlock, self).__init__()
        self.in_channels=in_channels
        self.deploy=deploy
        self.set_y=set_y
        if deploy:
            scale = torch.ones(1, in_channels, 1, 1)
            bias = torch.zeros(1, in_channels, 1, 1)
            self.sn = BNIFNode(scale, bias, v_threshold=V_THRESHOLD, detach_reset=True)
        else:
            self.bn = layer.BatchNorm2d(in_channels)
            self.sn = neuron.IFNode(v_threshold=V_THRESHOLD, detach_reset=True, surrogate_function=surrogate.ATan())

    def forward(self, x):
        if self.deploy:
            return self.sn(x)
        else:
            x, y = x
            out = self.bn(x)
            if self.training and self.set_y:
                y = out
            return self.sn(out), y
        
    def _bn_tensor(self, bn):
        running_mean = bn.running_mean
        running_var = bn.running_var
        gamma = bn.weight
        beta = bn.bias
        eps = bn.eps
        std = torch.sqrt(running_var + eps)
        t = std/gamma
        b = beta*t - running_mean
        return t.reshape(1,-1,1,1), b.reshape(1,-1,1,1)
    
    def switch_to_deploy(self):
        pass
        if isinstance(self.sn, BNIFNode):
            return
        scale, bias = self._bn_tensor(self.bn)
        self.sn = BNIFNode(scale, bias, v_threshold=V_THRESHOLD, detach_reset=True, step_mode=self.sn.step_mode).to(self.bn.weight)#TODO: fix cupy backend and add backend param later
        self.__delattr__('bn')
        self.deploy=True

class SpikeRepVGGBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride, groups=1, deploy=False, use_aac=True):
        super(SpikeRepVGGBlock, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.stride = stride
        self.groups = groups
        self.deploy = deploy
        self.identity = stride == 1 and in_channels == out_channels
        if deploy:
            self.reparam = layer.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=stride, groups=groups, padding=1, bias=True)
        else:
            self.conv3x3 = layer.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=stride, groups=groups,padding=1, bias=False)
            self.bn3x3 = layer.BatchNorm2d(out_channels)
            self.conv1x1 = layer.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=stride, groups=groups, padding=0, bias=False)
            self.bn = layer.BatchNorm2d(out_channels)
            if self.identity:
                self.aac = nn.Identity()
            elif use_aac:
                self.aac = convrelupxp(in_channels, out_channels, stride)
        self.sn = neuron.IFNode(v_threshold=1.0, detach_reset=True, surrogate_function=surrogate.ATan())

    def forward(self, x):
        if self.deploy:
            return self.sn(self.reparam(x))
        else:
            x, y = x
            out = self.bn(self.conv1x1(x) + self.bn3x3(self.conv3x3(x)))
            if y is not None:
                y = self.aac(y) + out
            elif self.training:
                y = out
            return self.sn(out), y
        
    def get_equivalent_kernel_bias(self):
        kernel3x3, bias3x3 = self._fuse_bn_tensor(self.conv3x3, self.bn3x3)
        kernel = kernel3x3 + self._pad_1x1_to_3x3_tensor(self.conv1x1.weight)
        bias = bias3x3
        kernel, bias = self._fuse_extra_bn_tensor(kernel, bias, self.bn)

        if self.identity:
            input_dim = self.in_channels // self.groups
            id_tensor = torch.zeros((self.in_channels, input_dim, 3, 3), dtype=kernel.dtype, device=kernel.device)
            for i in range(self.in_channels):
                id_tensor[i, i % input_dim, 1, 1] = 1
            kernel = kernel + id_tensor
        return kernel, bias

    def _fuse_bn_tensor(self, conv, bn):
        kernel = conv.weight
        running_mean = bn.running_mean
        running_var = bn.running_var
        gamma = bn.weight
        beta = bn.bias
        eps = bn.eps
        std = torch.sqrt(running_var + eps)
        t = (gamma / std).reshape(-1, 1, 1, 1)
        return kernel * t, beta - running_mean * gamma / std
    
    def _fuse_extra_bn_tensor(self, kernel, bias, bn):
        running_mean = bn.running_mean - bias
        running_var = bn.running_var
        gamma = bn.weight
        beta = bn.bias
        eps = bn.eps
        std = torch.sqrt(running_var + eps)
        t = (gamma / std).reshape(-1, 1, 1, 1)
        return kernel * t, beta - running_mean * gamma / std
    
    def _pad_1x1_to_3x3_tensor(self, kernel):
        return F.pad(kernel, [1,1,1,1])
    
    def switch_to_deploy(self):
        if hasattr(self, 'reparam'):
            return
        kernel, bias = self.get_equivalent_kernel_bias()
        self.reparam = layer.Conv2d(in_channels=self.in_channels, out_channels=self.out_channels, 
                                    kernel_size=3, stride=self.stride, 
                                    padding=1, groups=self.groups, bias=True, step_mode=self.conv3x3.step_mode).to(self.conv3x3.weight.device)
        self.reparam.weight.data = kernel
        self.reparam.bias.data = bias
        #for para in self.parameters(): #commented out for syops param count
        #    para.detach_()
        self.__delattr__('conv3x3')
        self.__delattr__('conv1x1')
        self.__delattr__('bn3x3')
        self.__delattr__('bn')
        self.__delattr__('aac')
        self.deploy=True
    
class SpikeConnRepVGGBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride, groups=1, deploy=False, use_aac=True):
        super(SpikeConnRepVGGBlock, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.stride = stride
        self.groups = groups
        self.deploy = deploy
        self.identity = stride == 1 and in_channels == out_channels
        if deploy:
            self.reparam = layer.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=stride, groups=groups, padding=1, bias=True)
            self.sn = neuron.IFNode(v_threshold=1.0, detach_reset=True, surrogate_function=surrogate.ATan())
        else:
            self.conv3x3 = layer.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=stride, groups=groups, padding=1, bias=False)
            self.bn3x3 = layer.BatchNorm2d(out_channels)
            self.conv1x1 = layer.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=stride, groups=groups, padding=0, bias=False)
            self.bn = layer.BatchNorm2d(out_channels)
            if self.identity:
                self.aac = nn.Identity()
                self.sn = ConnIFNode(v_threshold=1.0, detach_reset=True, surrogate_function=surrogate.ATan())
            else:
                if use_aac:
                    self.aac = convrelupxp(in_channels, out_channels, stride)
                self.sn = neuron.IFNode(v_threshold=1.0, detach_reset=True, surrogate_function=surrogate.ATan())
            
    def forward(self, x):
        if self.deploy:
            return self.sn(self.reparam(x))
        else:
            x, y = x
            out = self.bn(self.conv1x1(x) + self.bn3x3(self.conv3x3(x)))
            if y is not None:
                y = self.aac(y) + out
            elif self.training:
                y = out
            if self.identity:
                return self.sn(out, x), y
            else:
                return self.sn(out), y
    
    def get_equivalent_kernel_bias(self):
        kernel3x3, bias3x3 = self._fuse_bn_tensor(self.conv3x3, self.bn3x3)
        kernel = kernel3x3 + self._pad_1x1_to_3x3_tensor(self.conv1x1.weight)
        bias = bias3x3
        kernel, bias = self._fuse_extra_bn_tensor(kernel, bias, self.bn)

        if self.identity:
            identity_value = self.sn.y_multiplier
            input_dim = self.in_channels // self.groups
            id_tensor = torch.zeros((self.in_channels, input_dim, 3, 3), dtype=kernel.dtype, device=kernel.device)
            for i in range(self.in_channels):
                id_tensor[i, i % input_dim, 1, 1] = identity_value
            kernel = kernel + id_tensor
        return kernel, bias

    def _fuse_bn_tensor(self, conv, bn):
        kernel = conv.weight
        running_mean = bn.running_mean
        running_var = bn.running_var
        gamma = bn.weight
        beta = bn.bias
        eps = bn.eps
        std = torch.sqrt(running_var + eps)
        t = (gamma / std).reshape(-1, 1, 1, 1)
        return kernel * t, beta - running_mean * gamma / std
    
    def _fuse_extra_bn_tensor(self, kernel, bias, bn):
        running_mean = bn.running_mean - bias
        running_var = bn.running_var
        gamma = bn.weight
        beta = bn.bias
        eps = bn.eps
        std = torch.sqrt(running_var + eps)
        t = (gamma / std).reshape(-1, 1, 1, 1)
        return kernel * t, beta - running_mean * gamma / std
    
    def _pad_1x1_to_3x3_tensor(self, kernel):
        return F.pad(kernel, [1,1,1,1])
    
    def switch_to_deploy(self):
        if hasattr(self, 'reparam'):
            return
        kernel, bias = self.get_equivalent_kernel_bias()
        self.reparam = layer.Conv2d(in_channels=self.in_channels, out_channels=self.out_channels, 
                                    kernel_size=3, stride=self.stride, 
                                    padding=1, groups=self.groups, bias=True, step_mode=self.conv3x3.step_mode).to(self.conv3x3.weight.device)
        self.reparam.weight.data = kernel
        self.reparam.bias.data = bias
        self.sn = neuron.IFNode(v_threshold=1.0, detach_reset=True, surrogate_function=surrogate.ATan(), step_mode=self.sn.step_mode,backend=self.sn.backend).to(self.conv3x3.weight.device)
        #for para in self.parameters(): #commented out for syops param count
        #    para.detach_()
        self.__delattr__('conv3x3')
        self.__delattr__('conv1x1')
        self.__delattr__('bn3x3')
        self.__delattr__('bn')
        self.__delattr__('aac')
        self.deploy=True

class SRepVGG(nn.Module):
    def __init__(self, cfg_dict, num_classes, T=4, deploy=False, conversion=False, conversion_set_y=True):
        super(SRepVGG, self).__init__()
        self.deploy = deploy
        if cfg_dict['block_type'] == 'spike':
            self.block = SpikeRepVGGBlock
        elif cfg_dict['block_type'] == 'spike_connecting':
            self.block = SpikeConnRepVGGBlock
        else:
            raise NotImplementedError
        
        self.override_groups_map = cfg_dict.get('override_groups_map', {})
        self.cur_layer_idx=0
        self.T = T
        in_channels=3
        first_block=True #hack to avoid ddp crash from aap in first layer. TODO: fix
        layer_list = cfg_dict['layers']
        convs = nn.Sequential()
        if conversion:
            convs.append(ConversionBlock(in_channels, deploy=deploy))
            first_block=False
        for layer_dict in layer_list:
            convs.extend(self._build_layer(in_channels, layer_dict, first_block))
            in_channels = layer_dict['channels']
            first_block=False
        self.convs = convs

        self.avgpool = layer.AdaptiveAvgPool2d((1,1))
        self.flatten = nn.Flatten(2)
        self.fc = layer.Linear(in_channels, num_classes)
        if not self.deploy:
            self.aac = layer.Linear(in_channels, num_classes)

    def _build_layer(self, in_channels, layer_dict, first_block=False):
        channels = layer_dict['channels']
        stride = layer_dict['stride']
        convs = nn.Sequential()
        convs.append(self.block(in_channels, channels, stride, groups = self.override_groups_map.get(self.cur_layer_idx, 1), deploy=self.deploy, use_aac=not first_block))
        self.cur_layer_idx += 1
        for _ in range(layer_dict['num_blocks'] - 1):
            convs.append(self.block(channels, channels, 1, groups = self.override_groups_map.get(self.cur_layer_idx, 1), deploy=self.deploy))
            self.cur_layer_idx += 1
        return convs
    
    def forward(self, x: torch.Tensor):
        if len(x.shape) == 4 and self.T>0:
            x.unsqueeze(0).repeat(self.T,1,1,1,1)
        if self.deploy:
            out = self.convs(x)
            out = self.avgpool(out)
            out = self.flatten(out)
            out = self.fc(out.mean(0))
            return out
        else:
            x,y = self.convs((x,None))
            x = self.avgpool(x)
            x = self.flatten(x)
            x = self.fc(x.mean(0))
            if y is not None:
                y = self.avgpool(y)
                y = self.flatten(y)
                y = self.aac(y.mean(0))
            return x,y
    
    def switch_to_deploy(self):
        if self.deploy:
            return
        for layer in self.convs:
            if hasattr(layer, 'switch_to_deploy'):
                layer.switch_to_deploy()
        self.__delattr__('aac')
        self.deploy = True
    
optional_groupwise_layers = [2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26]
g2_map = {l: 2 for l in optional_groupwise_layers}
g4_map = {l: 4 for l in optional_groupwise_layers}

def SRepVGG_A0(num_classes, T=4, block_type='spike_connecting', conversion=False, conversion_set_y=True):
    cfg_dict = {
        'block_type': block_type,
        'layers': [
            {'channels': 48, 'num_blocks': 1, 'stride': 2},
            {'channels': 48, 'num_blocks': 2, 'stride': 2},
            {'channels': 96, 'num_blocks': 4, 'stride': 2},
            {'channels': 192, 'num_blocks': 14, 'stride': 2},
            {'channels': 1280, 'num_blocks': 1, 'stride': 2}
        ]
    }
    return SRepVGG(cfg_dict, num_classes, T=T, conversion=conversion, conversion_set_y=conversion_set_y)

def SRepVGG_A1(num_classes, T=4, block_type='spike_connecting', conversion=False, conversion_set_y=True):
    cfg_dict = {
        'block_type': block_type,
        'layers': [
            {'channels': 64, 'num_blocks': 1, 'stride': 2},
            {'channels': 64, 'num_blocks': 2, 'stride': 2},
            {'channels': 128, 'num_blocks': 4, 'stride': 2},
            {'channels': 256, 'num_blocks': 14, 'stride': 2},
            {'channels': 1280, 'num_blocks': 1, 'stride': 2}
        ]
    }
    return SRepVGG(cfg_dict, num_classes, T=T, conversion=conversion, conversion_set_y=conversion_set_y)

def SRepVGG_A2(num_classes, T=4, block_type='spike_connecting', conversion=False, conversion_set_y=True):
    cfg_dict = {
        'block_type': block_type,
        'layers': [
            {'channels': 64, 'num_blocks': 1, 'stride': 2},
            {'channels': 96, 'num_blocks': 2, 'stride': 2},
            {'channels': 192, 'num_blocks': 4, 'stride': 2},
            {'channels': 384, 'num_blocks': 14, 'stride': 2},
            {'channels': 1408, 'num_blocks': 1, 'stride': 2}
        ]
    }
    return SRepVGG(cfg_dict, num_classes, T=T, conversion=conversion, conversion_set_y=conversion_set_y)

def SRepVGG_B0(num_classes, T=4, block_type='spike_connecting', conversion=False, conversion_set_y=True):
    cfg_dict = {
        'block_type': block_type,
        'layers': [
            {'channels': 64, 'num_blocks': 1, 'stride': 1},
            {'channels': 64, 'num_blocks': 4, 'stride': 2},
            {'channels': 128, 'num_blocks': 6, 'stride': 2},
            {'channels': 256, 'num_blocks': 16, 'stride': 2},
            {'channels': 1280, 'num_blocks': 1, 'stride': 2}
        ]
    }
    return SRepVGG(cfg_dict, num_classes, T=T, conversion=conversion, conversion_set_y=conversion_set_y)

def SRepVGG_B1(num_classes, T=4, block_type='spike_connecting', conversion=False, conversion_set_y=True):
    cfg_dict = {
        'block_type': block_type,
        'layers': [
            {'channels': 64, 'num_blocks': 1, 'stride': 1},
            {'channels': 128, 'num_blocks': 4, 'stride': 2},
            {'channels': 256, 'num_blocks': 6, 'stride': 2},
            {'channels': 512, 'num_blocks': 16, 'stride': 2},
            {'channels': 2048, 'num_blocks': 1, 'stride': 2}
        ]
    }
    return SRepVGG(cfg_dict, num_classes, T=T, conversion=conversion, conversion_set_y=conversion_set_y)

def SRepVGG_B1g2(num_classes, T=4, block_type='spike_connecting', conversion=False, conversion_set_y=True):
    cfg_dict = {
        'block_type': block_type,
        'override_groups_map': g2_map,
        'layers': [
            {'channels': 64, 'num_blocks': 1, 'stride': 1},
            {'channels': 128, 'num_blocks': 4, 'stride': 2},
            {'channels': 256, 'num_blocks': 6, 'stride': 2},
            {'channels': 512, 'num_blocks': 16, 'stride': 2},
            {'channels': 2048, 'num_blocks': 1, 'stride': 2}
        ]
    }
    return SRepVGG(cfg_dict, num_classes, T=T, conversion=conversion, conversion_set_y=conversion_set_y)

def SRepVGG_B1g4(num_classes, T=4, block_type='spike_connecting', conversion=False, conversion_set_y=True):
    cfg_dict = {
        'block_type': block_type,
        'override_groups_map': g4_map,
        'layers': [
            {'channels': 64, 'num_blocks': 1, 'stride': 1},
            {'channels': 128, 'num_blocks': 4, 'stride': 2},
            {'channels': 256, 'num_blocks': 6, 'stride': 2},
            {'channels': 512, 'num_blocks': 16, 'stride': 2},
            {'channels': 2048, 'num_blocks': 1, 'stride': 2}
        ]
    }
    return SRepVGG(cfg_dict, num_classes, T=T, conversion=conversion, conversion_set_y=conversion_set_y)

def SRepVGG_B2(num_classes, T=4, block_type='spike_connecting', conversion=False, conversion_set_y=True):
    cfg_dict = {
        'block_type': block_type,
        'layers': [
            {'channels': 64, 'num_blocks': 1, 'stride': 1},
            {'channels': 160, 'num_blocks': 4, 'stride': 2},
            {'channels': 320, 'num_blocks': 6, 'stride': 2},
            {'channels': 640, 'num_blocks': 16, 'stride': 2},
            {'channels': 2560, 'num_blocks': 1, 'stride': 2}
        ]
    }
    return SRepVGG(cfg_dict, num_classes, T=T, conversion=conversion, conversion_set_y=conversion_set_y)

def SRepVGG_B2g2(num_classes, T=4, block_type='spike_connecting', conversion=False, conversion_set_y=True):
    cfg_dict = {
        'block_type': block_type,
        'override_groups_map': g2_map,
        'layers': [
            {'channels': 64, 'num_blocks': 1, 'stride': 1},
            {'channels': 160, 'num_blocks': 4, 'stride': 2},
            {'channels': 320, 'num_blocks': 6, 'stride': 2},
            {'channels': 640, 'num_blocks': 16, 'stride': 2},
            {'channels': 2560, 'num_blocks': 1, 'stride': 2}
        ]
    }
    return SRepVGG(cfg_dict, num_classes, T=T, conversion=conversion, conversion_set_y=conversion_set_y)

def SRepVGG_B2g4(num_classes, T=4, block_type='spike_connecting', conversion=False, conversion_set_y=True):
    cfg_dict = {
        'block_type': block_type,
        'override_groups_map': g4_map,
        'layers': [
            {'channels': 64, 'num_blocks': 1, 'stride': 1},
            {'channels': 160, 'num_blocks': 4, 'stride': 2},
            {'channels': 320, 'num_blocks': 6, 'stride': 2},
            {'channels': 640, 'num_blocks': 16, 'stride': 2},
            {'channels': 2560, 'num_blocks': 1, 'stride': 2}
        ]
    }
    return SRepVGG(cfg_dict, num_classes, T=T, conversion=conversion, conversion_set_y=conversion_set_y)

def SRepVGG_B3(num_classes, T=4, block_type='spike_connecting', conversion=False, conversion_set_y=True):
    cfg_dict = {
        'block_type': block_type,
        'layers': [
            {'channels': 64, 'num_blocks': 1, 'stride': 1},
            {'channels': 192, 'num_blocks': 4, 'stride': 2},
            {'channels': 384, 'num_blocks': 6, 'stride': 2},
            {'channels': 768, 'num_blocks': 16, 'stride': 2},
            {'channels': 2560, 'num_blocks': 1, 'stride': 2}
        ]
    }
    return SRepVGG(cfg_dict, num_classes, T=T, conversion=conversion, conversion_set_y=conversion_set_y)

def SRepVGG_B3g2(num_classes, T=4, block_type='spike_connecting', conversion=False, conversion_set_y=True):
    cfg_dict = {
        'block_type': block_type,
        'override_groups_map': g2_map,
        'layers': [
            {'channels': 64, 'num_blocks': 1, 'stride': 1},
            {'channels': 192, 'num_blocks': 4, 'stride': 2},
            {'channels': 384, 'num_blocks': 6, 'stride': 2},
            {'channels': 768, 'num_blocks': 16, 'stride': 2},
            {'channels': 2560, 'num_blocks': 1, 'stride': 2}
        ]
    }
    return SRepVGG(cfg_dict, num_classes, T=T, conversion=conversion, conversion_set_y=conversion_set_y)

def SRepVGG_B3g4(num_classes, T=4, block_type='spike_connecting', conversion=False, conversion_set_y=True):
    cfg_dict = {
        'block_type': block_type,
        'override_groups_map': g4_map,
        'layers': [
            {'channels': 64, 'num_blocks': 1, 'stride': 1},
            {'channels': 192, 'num_blocks': 4, 'stride': 2},
            {'channels': 384, 'num_blocks': 6, 'stride': 2},
            {'channels': 768, 'num_blocks': 16, 'stride': 2},
            {'channels': 2560, 'num_blocks': 1, 'stride': 2}
        ]
    }
    return SRepVGG(cfg_dict, num_classes, T=T, conversion=conversion, conversion_set_y=conversion_set_y)

model_dict = {
    'SRepVGG_A0': SRepVGG_A0,
    'SRepVGG_A1': SRepVGG_A1,
    'SRepVGG_A2': SRepVGG_A2,
    'SRepVGG_B0': SRepVGG_B0,
    'SRepVGG_B1': SRepVGG_B1,
    'SRepVGG_B1g2': SRepVGG_B1g2,
    'SRepVGG_B1g4': SRepVGG_B1g4,
    'SRepVGG_B2': SRepVGG_B2,
    'SRepVGG_B2g2': SRepVGG_B2g2,
    'SRepVGG_B2g4': SRepVGG_B2g4,
    'SRepVGG_B3': SRepVGG_B3,
    'SRepVGG_B3g2': SRepVGG_B3g2,
    'SRepVGG_B3g4': SRepVGG_B3g4,
}

def get_model_by_name(name):
    return model_dict[name]

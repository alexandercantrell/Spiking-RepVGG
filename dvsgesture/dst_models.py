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
    def __init__(self, in_channels, out_channels, stride, groups=1, deploy=False):
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
            if not self.identity:
                self.aac = convrelupxp(in_channels, out_channels, stride)
        self.sn = neuron.ParametricLIFNode(v_threshold=1.0, detach_reset=True, surrogate_function=surrogate.ATan())

    def forward(self, x):
        if self.deploy:
            return self.sn(self.reparam(x))
        else:
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
        if hasattr(self, 'aac'):
            self.__delattr__('aac')
        self.deploy=True
    
class SpikeConnRepVGGBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride, groups=1, deploy=False):
        super(SpikeConnRepVGGBlock, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.stride = stride
        self.groups = groups
        self.deploy = deploy
        self.identity = stride == 1 and in_channels == out_channels
        if deploy:
            self.reparam = layer.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=stride, groups=groups, padding=1, bias=True)
            self.sn = neuron.ParametricLIFNode(v_threshold=1.0, detach_reset=True, surrogate_function=surrogate.ATan())
        else:
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
        if self.deploy:
            return self.sn(self.reparam(x))
        else:
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
    
    def get_equivalent_kernel_bias(self):
        kernel3x3, bias3x3 = self._fuse_bn_tensor(self.conv3x3, self.bn3x3)
        kernel = kernel3x3 + self._pad_1x1_to_3x3_tensor(self.conv1x1.weight)
        bias = bias3x3
        kernel, bias = self._fuse_extra_bn_tensor(kernel, bias, self.bn)

        if self.identity:
            identity_value = self.sn.v_threshold/self.sn.w.sigmoid()
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
        w = self.sn.w.data
        self.sn = neuron.ParametricLIFNode(v_threshold=1.0, detach_reset=True, surrogate_function=surrogate.ATan(), step_mode=self.conv3x3.step_mode).to(self.conv3x3.weight.device)
        self.sn.w.data = w
        #for para in self.parameters(): #commented out for syops param count
        #    para.detach_()
        self.__delattr__('conv3x3')
        self.__delattr__('conv1x1')
        self.__delattr__('bn3x3')
        self.__delattr__('bn')
        if hasattr(self, 'aac'):
            self.__delattr__('aac')
        self.deploy=True

class SRepVGG(nn.Module):
    def __init__(self, cfg_dict, num_classes, deploy=False):
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
        
        in_channels=2
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
        convs.append(self.block(in_channels, channels, stride, groups = self.override_groups_map.get(self.cur_layer_idx, 1), deploy=self.deploy))
        self.cur_layer_idx += 1
        for _ in range(layer_dict['num_blocks'] - 1):
            convs.append(self.block(channels, channels, 1, groups = self.override_groups_map.get(self.cur_layer_idx, 1), deploy=self.deploy))
            self.cur_layer_idx += 1
        return convs
    
    def forward(self, x: torch.Tensor):
        x = x.permute(1,0,2,3,4)
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
        if hasattr(self,'aac'):
            self.__delattr__('aac')
        self.deploy = True

def SRepVGG_N0(num_classes, block_type='spike_connecting'):
    cfg_dict = {
        'block_type': block_type,
        'layers': [
            {'channels': 32, 'num_blocks': 1, 'stride': 1},
            {'channels': 32, 'num_blocks': 6, 'stride': 2},
            {'channels': 32, 'num_blocks': 6, 'stride': 2},
            {'channels': 32, 'num_blocks': 5, 'stride': 2},
            {'channels': 32, 'num_blocks': 1, 'stride': 2},
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

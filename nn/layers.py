import inspect
import torch 
import torch.nn as nn 
import numpy as np
from spikingjelly.activation_based import layer, neuron
from copy import deepcopy
from nn.connecting_functions import ConnectingFunction

def conv_bn(in_channels, out_channels, kernel_size, stride, padding, padding_mode='zeros', groups=1):
    result = nn.Sequential()
    result.add_module('conv', layer.Conv2d(in_channels=in_channels, out_channels=out_channels,
                                                  kernel_size=kernel_size, stride=stride, padding=padding, padding_mode=padding_mode, groups=groups, bias=False))
    result.add_module('bn', layer.BatchNorm2d(num_features=out_channels))
    return result

def make_neuron(spiking_neuron,**kwargs):
    if inspect.isclass(spiking_neuron) and issubclass(spiking_neuron,neuron.BaseNode):
        return spiking_neuron(**deepcopy(kwargs))
    elif isinstance(spiking_neuron,neuron.BaseNode):
        return spiking_neuron
    else:
        raise TypeError(f'{spiking_neuron} is not a subclass or instance of a subclass of the spikingjelly BaseNode.')

class SpikingRepVGGBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size,
                 stride=1, padding=0, dilation=1, groups=1, padding_mode='zeros', 
                 deploy=False, spiking_neuron = None, **kwargs):
        super(SpikingRepVGGBlock,self).__init__()
        assert kernel_size == 3
        assert padding == 1
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.padding_11 = padding - kernel_size // 2
        self.dilation = dilation
        self.groups = groups
        self.padding_mode = padding_mode
        self.deploy = deploy

        if deploy:
            self.rbr_reparam = layer.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride,
                                      padding=padding, padding_mode=padding_mode, dilation=dilation, groups=groups, bias=True)
        else:
            self.rbr_dense = conv_bn(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding, padding_mode=padding_mode, groups=groups)
            self.rbr_1x1 = conv_bn(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=stride, padding=self.padding_11, padding_mode=padding_mode, groups=groups)
            self.rbr_identity = nn.Identity() if out_channels == in_channels and stride == 1 else None

        self.sn = make_neuron(spiking_neuron,**kwargs)

    def forward(self, x):
        if hasattr(self,'rbr_reparam'):
            return self.sn(self.rbr_reparam(x))

        if self.rbr_identity is None: 
            identity = 0
        else: 
            identity = self.rbr_identity(x)
        return self.sn(self.rbr_dense(x) + self.rbr_1x1(x) + identity)
    
    def get_equivalent_kernel_bias(self):
        kernel3x3, bias3x3 = self._fuse_bn_tensor(self.rbr_dense)
        kernel1x1, bias1x1 = self._fuse_bn_tensor(self.rbr_1x1)
        kernel = kernel3x3 + self._pad_1x1_to_3x3_tensor(kernel1x1)
        bias = bias3x3 + bias1x1
        if self.rbr_identity is not None:
            input_dim = self.in_channels // self.groups
            kernel_value = np.zeros((self.in_channels, input_dim, 3, 3), dtype=np.float32)
            for i in range(self.in_channels):
                kernel_value[i, i % input_dim, 1, 1] = 1
            id_tensor = torch.from_numpy(kernel_value).to(self.rbr_1x1.weight.device)
            kernel = kernel + id_tensor
        return kernel, bias
    
    def _pad_1x1_to_3x3_tensor(self, kernel1x1):
        if kernel1x1 is None:
            return 0
        else:
            return torch.nn.functional.pad(kernel1x1, [1, 1, 1, 1])
    
    def _fuse_bn_tensor(self, branch):
        if branch is None:
            return 0, 0
        assert isinstance(branch, nn.Sequential)
        kernel = branch.conv.weight
        running_mean = branch.bn.running_mean
        running_var = branch.bn.running_var
        gamma = branch.bn.weight
        beta = branch.bn.bias
        eps = branch.bn.eps
        std = (running_var + eps).sqrt()
        t = (gamma / std).reshape(-1, 1, 1, 1)
        return kernel * t, beta - running_mean * gamma / std
    
    def switch_to_deploy(self):
        if hasattr(self, 'rbr_reparam'):
            return
        kernel, bias = self.get_equivalent_kernel_bias()
        self.rbr_reparam = layer.Conv2d(in_channels=self.in_channels, out_channels=self.out_channels,
                                     kernel_size=self.kernel_size, stride=self.stride,
                                     padding=self.padding, padding_mode=self.padding_mode, dilation=self.dilation, 
                                     groups=self.groups, bias=True)
        self.rbr_reparam.weight.data = kernel
        self.rbr_reparam.bias.data = bias
        for para in self.parameters():
            para.detach_()
        self.__delattr__('rbr_dense')
        self.__delattr__('rbr_1x1')
        if hasattr(self, 'rbr_identity'):
            self.__delattr__('rbr_identity')
        self.deploy = True

class QASpikingRepVGGBlock(SpikingRepVGGBlock):
    def __init__(self, in_channels, out_channels, kernel_size,
                 stride=1, padding=0, dilation=1, groups=1, padding_mode='zeros', 
                 deploy=False, spiking_neuron = None, **kwargs):
        super(QASpikingRepVGGBlock,self).__init__(in_channels, out_channels, kernel_size,
                    stride, padding, dilation, groups, padding_mode, 
                    deploy, spiking_neuron, **kwargs)
        if not deploy:
            self.rbr_1x1 = layer.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, groups=groups, bias=False, padding=self.padding_11, padding_mode=self.padding_mode)
            self.bn = layer.BatchNorm2d(out_channels)

    def forward(self, x):
        if hasattr(self,'rbr_reparam'):
            return self.sn(self.rbr_reparam(x))
        
        if self.rbr_identity is None:
            identity = 0
        else:
            identity = self.rbr_identity(x)
        return self.sn(self.bn(self.rbr_dense(x) + self.rbr_1x1(x)) + identity)

    def get_equivalent_kernel_bias(self):
        kernel3x3, bias3x3 = self._fuse_bn_tensor(self.rbr_dense)
        kernel = kernel3x3 + self._pad_1x1_to_3x3_tensor(self.rbr_1x1.weight)
        bias = bias3x3
        kernel, bias = self._fuse_extra_bn_tensor(kernel, bias, self.bn)
        if self.rbr_identity is not None:
            input_dim = self.in_channels // self.groups
            kernel_value = np.zeros((self.in_channels, input_dim, 3, 3), dtype=np.float32)
            for i in range(self.in_channels):
                kernel_value[i, i % input_dim, 1, 1] = 1
            id_tensor = torch.from_numpy(kernel_value).to(self.rbr_1x1.weight.device)
            kernel = kernel + id_tensor
        return kernel, bias

    def _fuse_extra_bn_tensor(self, kernel, bias, branch):
        assert isinstance(branch, nn.BatchNorm2d)
        #TODO: check if you should remove bias
        running_mean = branch.running_mean - bias # remove bias
        running_var = branch.running_var
        gamma = branch.weight
        beta = branch.bias
        eps = branch.eps
        std = (running_var + eps).sqrt()
        t = (gamma / std).reshape(-1, 1, 1, 1)
        return kernel * t, beta - running_mean * gamma / std
    
    def switch_to_deploy(self):
        if hasattr(self, 'rbr_reparam'):
            return
        kernel, bias = self.get_equivalent_kernel_bias()
        self.rbr_reparam = layer.Conv2d(in_channels=self.in_channels, out_channels=self.out_channels,
                                     kernel_size=self.kernel_size, stride=self.stride,
                                     padding=self.padding, padding_mode=self.padding_mode, dilation=self.dilation, 
                                     groups=self.groups, bias=True)
        self.rbr_reparam.weight.data = kernel
        self.rbr_reparam.bias.data = bias
        for para in self.parameters():
            para.detach_()
        self.__delattr__('rbr_dense')
        self.__delattr__('rbr_1x1')
        if hasattr(self, 'rbr_identity'):
            self.__delattr__('rbr_identity')
        if hasattr(self, 'bn'):
            self.__delattr__('bn')
        self.deploy = True
    
class SEWRepVGGBlock(SpikingRepVGGBlock):
    def __init__(self, in_channels, out_channels, kernel_size,
                 stride=1, padding=0, dilation=1, groups=1, padding_mode='zeros', 
                 deploy=False, cnf = None, spiking_neuron = None, **kwargs):
        super(SEWRepVGGBlock,self).__init__(in_channels, out_channels, kernel_size,
                    stride, padding, dilation, groups, padding_mode, 
                    deploy, spiking_neuron, **kwargs)

        self.rbr_identity = nn.Identity() if out_channels == in_channels and stride == 1 else None
        self.cnf = ConnectingFunction(cnf) if self.rbr_identity is not None else None

    def forward(self, x):
        if hasattr(self,'rbr_reparam'):
            out = self.sn(self.rbr_reparam(x))
        else:
            out = self.sn(self.rbr_dense(x) + self.rbr_1x1(x))
        if self.cnf is not None:
            out = self.cnf(self.rbr_identity(x),out)
        return out
    
    def get_equivalent_kernel_bias(self):
        kernel3x3, bias3x3 = self._fuse_bn_tensor(self.rbr_dense)
        kernel1x1, bias1x1 = self._fuse_bn_tensor(self.rbr_1x1)
        return kernel3x3 + self._pad_1x1_to_3x3_tensor(kernel1x1), bias3x3 + bias1x1

    def switch_to_deploy(self):
        if hasattr(self, 'rbr_reparam'):
            return
        kernel, bias = self.get_equivalent_kernel_bias()
        self.rbr_reparam = layer.Conv2d(in_channels=self.in_channels, out_channels=self.out_channels,
                                     kernel_size=self.kernel_size, stride=self.stride,
                                     padding=self.padding, padding_mode=self.padding_mode, dilation=self.dilation, 
                                     groups=self.groups, bias=True)
        self.rbr_reparam.weight.data = kernel
        self.rbr_reparam.bias.data = bias
        for para in self.parameters():
            para.detach_()
        self.__delattr__('rbr_dense')
        self.__delattr__('rbr_1x1')
        self.deploy = True

class IDSEWRepVGGBlock(SEWRepVGGBlock):
    def __init__(self, in_channels, out_channels, kernel_size,
                 stride=1, padding=0, dilation=1, groups=1, padding_mode='zeros', 
                 deploy=False, cnf = None, spiking_neuron = None, **kwargs):
        super(IDSEWRepVGGBlock,self).__init__(in_channels, out_channels, kernel_size,
                 stride, padding, dilation, groups, padding_mode, 
                 deploy, cnf, spiking_neuron, **kwargs)
        
        is_downsample = not(out_channels == in_channels and stride == 1)

        if deploy and is_downsample:
            self.rbr_identity = nn.Sequential()
            self.rbr_identity.add_module('conv', layer.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=stride,padding=self.padding_11, padding_mode=self.padding_mode, groups=groups))
            self.rbr_identity.add_module('sn', make_neuron(spiking_neuron, **kwargs))
        elif is_downsample:
            self.rbr_identity = nn.Sequential()
            self.rbr_identity.add_module('conv_bn', conv_bn(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=stride, padding=self.padding_11, padding_mode=self.padding_mode, groups=groups))
            self.rbr_identity.add_module('sn', make_neuron(spiking_neuron, **kwargs))
        else: self.rbr_identity = nn.Identity()
        self.cnf = ConnectingFunction(cnf)
    
    def forward(self, x):
        if hasattr(self,'rbr_reparam'):
            out = self.sn(self.rbr_reparam(x))
        else:
            out = self.sn(self.rbr_dense(x) + self.rbr_1x1(x))
        out = self.cnf(self.rbr_identity(x),out)
        return out
    
    def switch_to_deploy(self):
        if isinstance(self.rbr_identity, nn.Sequential):
            sn = deepcopy(self.rbr_identity.sn)
            kernel,bias = self._fuse_bn_tensor(self.rbr_identity.conv_bn)
            self.rbr_identity = nn.Sequential()
            self.rbr_identity.add_module('conv', layer.Conv2d(in_channels=self.in_channels, out_channels=self.out_channels,
                                    kernel_size=1, stride=self.stride,padding=self.padding, padding_mode=self.padding_mode,
                                    dilation=self.dilation, groups=self.groups, bias=True))
            self.rbr_identity.add_module('sn', sn)
            self.rbr_identity.conv.weight.data = kernel
            self.rbr_identity.conv.bias.data = bias
        super().switch_to_deploy()
    
class QASEWRepVGGBlock(QASpikingRepVGGBlock):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, padding_mode='zeros', deploy=False, cnf=None, spiking_neuron=None, **kwargs):
        super().__init__(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, padding_mode, deploy, cnf, spiking_neuron, **kwargs)
        if not deploy:
            self.rbr_1x1 = layer.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, groups=groups, bias=False,padding=self.padding_11,padding_mode=self.padding_mode)
            self.bn = layer.BatchNorm2d(out_channels)
        self.rbr_identity = nn.Identity() if out_channels == in_channels and stride == 1 else None
        self.cnf = ConnectingFunction(cnf) if self.rbr_identity is not None else None

    def forward(self, x):
        if hasattr(self, 'rbr_reparam'):
            out = self.sn(self.rbr_reparam(x))
        else:
            out = self.sn(self.bn(self.rbr_dense(x) + self.rbr_1x1(x)))
        if self.cnf is not None:
            out = self.cnf(self.rbr_identity(x), out)
        return out
    
    def get_equivalent_kernel_bias(self):
        kernel3x3, bias3x3 = self._fuse_bn_tensor(self.rbr_dense)
        kernel = kernel3x3 + self._pad_1x1_to_3x3_tensor(self.rbr_1x1.weight)
        bias = bias3x3
        return self._fuse_extra_bn_tensor(kernel, bias, self.bn)
    
    def switch_to_deploy(self):
        if hasattr(self, 'rbr_reparam'):
            return
        kernel, bias = self.get_equivalent_kernel_bias()
        self.rbr_reparam = layer.Conv2d(in_channels=self.in_channels, out_channels=self.out_channels,
                                     kernel_size=self.kernel_size, stride=self.stride,
                                     padding=self.padding, padding_mode=self.padding_mode, dilation=self.dilation, 
                                     groups=self.groups, bias=True)
        self.rbr_reparam.weight.data = kernel
        self.rbr_reparam.bias.data = bias
        for para in self.parameters():
            para.detach_()
        self.__delattr__('rbr_dense')
        self.__delattr__('rbr_1x1')
        if hasattr(self, 'bn'):
            self.__delattr__('bn')
        self.deploy = True

class IDQASEWRepVGGBlock(QASEWRepVGGBlock):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, padding_mode='zeros', deploy=False, cnf=None, spiking_neuron=None, **kwargs):
        super().__init__(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, padding_mode, deploy, cnf, spiking_neuron, **kwargs)
        
        is_downsample = not(out_channels == in_channels and stride == 1)

        if deploy and is_downsample:
            self.rbr_identity = nn.Sequential()
            self.rbr_identity.add_module('conv', layer.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=stride,padding=self.padding_11, padding_mode=self.padding_mode, groups=groups))
            self.rbr_identity.add_module('sn', make_neuron(spiking_neuron, **kwargs))
        elif is_downsample:
            self.rbr_identity = nn.Sequential()
            self.rbr_identity.add_module('conv_bn', conv_bn(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=stride, padding=self.padding_11, padding_mode=self.padding_mode, groups=groups))
            self.rbr_identity.add_module('sn', make_neuron(spiking_neuron, **kwargs))
        else: self.rbr_identity = nn.Identity()
        self.cnf = ConnectingFunction(cnf)
    
    def forward(self, x):
        if hasattr(self,'rbr_reparam'):
            out = self.sn(self.rbr_reparam(x))
        else:
            out = self.sn(self.bn(self.rbr_dense(x) + self.rbr_1x1(x)))
        out = self.cnf(self.rbr_identity(x),out)
        return out
    
    def switch_to_deploy(self):
        if isinstance(self.rbr_identity, nn.Sequential):
            sn = deepcopy(self.rbr_identity.sn)
            kernel,bias = self._fuse_bn_tensor(self.rbr_identity.conv_bn)
            self.rbr_identity = nn.Sequential()
            self.rbr_identity.add_module('conv', layer.Conv2d(in_channels=self.in_channels, out_channels=self.out_channels,
                                    kernel_size=1, stride=self.stride,padding=self.padding, padding_mode=self.padding_mode,
                                    dilation=self.dilation, groups=self.groups, bias=True))
            self.rbr_identity.add_module('sn', sn)
            self.rbr_identity.conv.weight.data = kernel
            self.rbr_identity.conv.bias.data = bias
        super().switch_to_deploy()

blocks = {
    'Spiking': SpikingRepVGGBlock,
    'QASpiking': QASpikingRepVGGBlock,
    'SEW': SEWRepVGGBlock,
    'IDSEW': IDSEWRepVGGBlock,
    'QASEW': QASEWRepVGGBlock,
    'IDQASEW': IDQASEWRepVGGBlock
}

def get_block_by_name(name):
    return blocks[name]
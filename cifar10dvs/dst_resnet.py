import torch
from collections import OrderedDict
from functools import partial
import torch.nn as nn
from spikingjelly.activation_based import layer, neuron, surrogate
from connecting_functions import ConnectingFunction

class AACBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, deploy=False):
        super(AACBlock, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.stride = stride
        self.deploy = deploy
        if stride != 1:
            self.stride_layer = layer.Conv2d(in_channels, in_channels, kernel_size=3, stride=stride, groups=in_channels, padding=1, bias=False)
        else:
            self.stride_layer = nn.Identity()

        if deploy:
            self.reparam = layer.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, bias=True)
        else:
            self.conv = layer.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, bias=False)
            self.bn = layer.BatchNorm2d(out_channels)
    
        self.relu = nn.ReLU()
    
    def forward(self, x):
        if self.deploy:
            return self.relu(self.reparam(self.stride_layer(x)))
        else:
            return self.relu(self.bn(self.conv(self.stride_layer(x))))
        
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
        
    def switch_to_deploy(self):
        if self.deploy:
            return
        kernel, bias = self._fuse_bn_tensor(self.conv, self.bn)
        self.reparam = layer.Conv2d(self.in_channels, self.out_channels, kernel_size=1, stride=1, bias=True, step_mode=self.conv.step_mode).to(self.conv.weight.device)
        self.reparam.weight.data = kernel
        self.reparam.bias.data = bias
        self.__delattr__('conv')
        self.__delattr__('bn')
        self.deploy=True
    
class DSTUpChannelBlock(nn.Module):
    def __init__(self, in_channels, out_channels, deploy=False, dsnn=False):
        super(DSTUpChannelBlock, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.deploy = deploy
        self.dsnn = dsnn
        if deploy:
            self.reparam = nn.Sequential(
                OrderedDict([
                    ('conv', layer.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, bias=True)),
                    ('sn', neuron.ParametricLIFNode(v_threshold=1.0, detach_reset=True, surrogate_function=surrogate.ATan()))
                ])
            )
            if self.dsnn:
                if in_channels != out_channels:
                    self.aac = nn.Identity()
                else:
                    self.aac = AACBlock(in_channels, out_channels, deploy=True)
        else:
            self.conv = nn.Sequential(
                OrderedDict([
                    ('conv', layer.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, bias=False)),
                    ('bn', layer.BatchNorm2d(out_channels)),
                    ('sn', neuron.ParametricLIFNode(v_threshold=1.0, detach_reset=True, surrogate_function=surrogate.ATan()))
                ])
            )
            if in_channels == out_channels:
                self.aac = nn.Identity()
            else:
                self.aac = AACBlock(in_channels, out_channels)
    
    def forward(self, x):
        if self.deploy:
            if self.dsnn:
                x, y = x
                out = self.reparam(x)
                if y is not None:
                    y = self.aac(y) + out
                else:
                    y = out
                return out, y
            else:
                return self.reparam(x)
        else:
            x,y = x
            out = self.conv(x)
            if y is not None:
                y = self.aac(y) + out
            elif self.training or self.dsnn:
                y = out
            return out, y
        
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

    def switch_to_deploy(self):
        if self.deploy:
            return
        kernel, bias = self._fuse_bn_tensor(self.conv.conv, self.conv.bn)
        self.reparam = nn.Sequential(
            OrderedDict([
                ('conv', layer.Conv2d(self.in_channels, self.out_channels, kernel_size=1, stride=1, bias=True, step_mode=self.conv.conv.step_mode).to(self.conv.conv.weight.device)),
                ('sn', neuron.ParametricLIFNode(v_threshold=1.0, detach_reset=True, surrogate_function=surrogate.ATan(), step_mode=self.conv.sn.step_mode).to(self.conv.conv.weight.device))
            ])
        )
        self.reparam.conv.weight.data = kernel
        self.reparam.conv.bias.data = bias
        self.reparam.sn.w.data = self.conv.sn.w.data
        if not self.dsnn:
            self.__delattr__('aac')
        else:
            self.aac.switch_to_deploy()
        self.__delattr__('conv')
        self.deploy=True

class DSTMaxPool2dBlock(nn.Module):
    def __init__(self, k_pool, dsnn=False):
        super(DSTMaxPool2dBlock, self).__init__()
        self.k_pool = k_pool
        self.pool = layer.MaxPool2d(k_pool)
        self.dsnn = dsnn
    
    def forward(self, x):
        if isinstance(x, tuple):
            x,y = x
            x = self.pool(x)
            if y is not None:
                y = self.pool(y)
            elif self.training or self.dsnn:
                y = x
            return x, y
        else:
            return self.pool(x)

class SpikeResNetBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride, groups=1, deploy=False, dsnn=False):
        super(SpikeResNetBlock, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.stride = stride
        self.groups = groups
        self.deploy = deploy
        self.dsnn = dsnn
        self.identity = stride==1 and in_channels==out_channels

        if self.deploy:
            self.reparam_conv1 = layer.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, groups=groups, padding=1, bias=True)
            self.sn1 = neuron.ParametricLIFNode(v_threshold=1.0, detach_reset=True, surrogate_function=surrogate.ATan())
            self.reparam_conv2 = layer.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, groups=groups, padding=1, bias=True)
            self.sn2 = neuron.ParametricLIFNode(v_threshold=1.0, detach_reset=True, surrogate_function=surrogate.ATan())
            
            if self.identity:
                self.reparam_skip = nn.Identity()
                if self.dsnn:
                    self.aac = nn.Identity()
            else:
                self.reparam_skip = layer.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=True)
                if self.dsnn:
                    self.aac = AACBlock(in_channels, out_channels, stride=stride, deploy=True)
        
        else:
            self.conv1 = layer.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, groups=groups, padding=1, bias=False)
            self.bn1 = layer.BatchNorm2d(out_channels)
            self.sn1 = neuron.ParametricLIFNode(v_threshold=1.0, detach_reset=True, surrogate_function=surrogate.ATan())
            self.conv2 = layer.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, groups=groups, padding=1, bias=False)
            self.bn2 = layer.BatchNorm2d(out_channels)
            self.sn2 = neuron.ParametricLIFNode(v_threshold=1.0, detach_reset=True, surrogate_function=surrogate.ATan())
        
            if self.identity:
                self.skip = nn.Identity()
                self.aac = nn.Identity()
            else:
                self.skip = nn.Sequential(
                    OrderedDict([
                        ('conv', layer.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False)),
                        ('bn', layer.BatchNorm2d(out_channels)),
                    ])
                )
                self.aac = AACBlock(in_channels, out_channels, stride=stride)

    def forward(self, x):
        if self.deploy:
            if self.dsnn:
                x, y = x
                out = self.reparam_conv2(self.sn1(self.reparam_conv1(x)))
                skip = self.reparam_skip(x)
                if y is not None:
                    y = self.aac(y) + out
                else:
                    y = out
                return self.sn2(out+skip), y
            else:
                return self.sn2(self.reparam_conv2(self.sn1(self.reparam_conv1(x))) + self.reparam_skip(x))
        else:
            x,y = x
            out = self.bn2(self.conv2(self.sn1(self.bn1(self.conv1(x)))))
            skip = self.skip(x)
            if y is not None:
                y = self.aac(y) + out
            elif self.training or self.dsnn:
                y = out
            return self.sn2(out+skip), y
        
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
        
    def switch_to_deploy(self):
        if self.deploy:
            return
        kernel1, bias1 = self._fuse_bn_tensor(self.conv1, self.bn1)
        self.reparam_conv1 = layer.Conv2d(self.in_channels, self.out_channels, kernel_size=3, stride=self.stride, groups=self.groups, padding=1, bias=True, step_mode=self.conv1.step_mode).to(self.conv1.weight.device)
        self.reparam_conv1.weight.data = kernel1
        self.reparam_conv1.bias.data = bias1

        kernel2, bias2 = self._fuse_bn_tensor(self.conv2, self.bn2)
        self.reparam_conv2 = layer.Conv2d(self.out_channels, self.out_channels, kernel_size=3, stride=1, groups=self.groups, padding=1, bias=True, step_mode=self.conv2.step_mode).to(self.conv2.weight.device)
        self.reparam_conv2.weight.data = kernel2
        self.reparam_conv2.bias.data = bias2

        if not self.identity:
            kernel3, bias3 = self._fuse_bn_tensor(self.skip.conv, self.skip.bn)
            self.reparam_skip = layer.Conv2d(self.in_channels, self.out_channels, kernel_size=1, stride=self.stride, bias=True, step_mode=self.skip.conv.step_mode).to(self.skip.conv.weight.device)
            self.reparam_skip.weight.data = kernel3
            self.reparam_skip.bias.data = bias3
        else:
            self.reparam_skip = nn.Identity().to(self.conv1.weight.device)

        if not self.dsnn:
            self.__delattr__('aac')
        elif isinstance(self.aac, AACBlock):
            self.aac.switch_to_deploy()

        self.__delattr__('conv1')
        self.__delattr__('bn1')
        self.__delattr__('conv2')
        self.__delattr__('bn2')

        self.deploy=True
        
class SEWResNetBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride, groups=1, cnf='AND', deploy=False, dsnn=False):
        super(SEWResNetBlock, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.stride = stride
        self.groups = groups
        self.deploy = deploy
        self.dsnn = dsnn
        self.identity = stride==1 and in_channels==out_channels

        if self.deploy:
            self.reparam_conv1 = layer.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, groups=groups, padding=1, bias=True)
            self.sn1 = neuron.ParametricLIFNode(v_threshold=1.0, detach_reset=True, surrogate_function=surrogate.ATan())
            self.reparam_conv2 = layer.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, groups=groups, padding=1, bias=True)
            self.sn2 = neuron.ParametricLIFNode(v_threshold=1.0, detach_reset=True, surrogate_function=surrogate.ATan())
            
            if self.identity:
                self.reparam_skip = nn.Identity()
                if self.dsnn:
                    self.aac = nn.Identity()
            else:
                self.reparam_skip = nn.Sequential(
                    OrderedDict([
                        ('conv', layer.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=True)),
                        ('sn', neuron.ParametricLIFNode(v_threshold=1.0, detach_reset=True, surrogate_function=surrogate.ATan()))
                    ])
                )
                if self.dsnn:
                    self.aac = AACBlock(in_channels, out_channels, stride=stride, deploy=True)
        else:
            self.conv1 = layer.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, groups=groups, padding=1, bias=False)
            self.bn1 = layer.BatchNorm2d(out_channels)
            self.sn1 = neuron.ParametricLIFNode(v_threshold=1.0, detach_reset=True, surrogate_function=surrogate.ATan())
            self.conv2 = layer.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, groups=groups, padding=1, bias=False)
            self.bn2 = layer.BatchNorm2d(out_channels)
            self.sn2 = neuron.ParametricLIFNode(v_threshold=1.0, detach_reset=True, surrogate_function=surrogate.ATan())
        
            if self.identity:
                self.skip = nn.Identity()
                self.aac = nn.Identity()
            else:
                self.skip = nn.Sequential(
                    OrderedDict([
                        ('conv', layer.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False)),
                        ('bn', layer.BatchNorm2d(out_channels)),
                        ('sn', neuron.ParametricLIFNode(v_threshold=1.0, detach_reset=True, surrogate_function=surrogate.ATan()))
                    ])
                )
                self.aac = AACBlock(in_channels, out_channels, stride=stride)

        self.cnf = ConnectingFunction(cnf)
        
    def forward(self, x):
        if self.deploy:
            if self.dsnn:
                x,y = x
                out = self.sn2(self.reparam_conv2(self.sn1(self.reparam_conv1(x))))
                skip = self.reparam_skip(x)
                if y is not None:
                    y = self.aac(y) + out
                else:
                    y = out
                return self.cnf(out,skip), y
            else:
                return self.cnf(self.sn2(self.reparam_conv2(self.sn1(self.reparam_conv1(x)))), self.reparam_skip(x))
        else:
            x,y = x
            out = self.sn2(self.bn2(self.conv2(self.sn1(self.bn1(self.conv1(x))))))
            skip = self.skip(x)
            if y is not None:
                y = self.aac(y) + out
            elif self.training or self.dsnn:
                y = out
            return self.cnf(out,skip), y
        
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
        
    def switch_to_deploy(self):
        if self.deploy:
            return
        kernel1, bias1 = self._fuse_bn_tensor(self.conv1, self.bn1)
        self.reparam_conv1 = layer.Conv2d(self.in_channels, self.out_channels, kernel_size=3, stride=self.stride, groups=self.groups, padding=1, bias=True, step_mode=self.conv1.step_mode).to(self.conv1.weight.device)
        self.reparam_conv1.weight.data = kernel1
        self.reparam_conv1.bias.data = bias1

        kernel2, bias2 = self._fuse_bn_tensor(self.conv2, self.bn2)
        self.reparam_conv2 = layer.Conv2d(self.out_channels, self.out_channels, kernel_size=3, stride=1, groups=self.groups, padding=1, bias=True, step_mode=self.conv2.step_mode).to(self.conv2.weight.device)
        self.reparam_conv2.weight.data = kernel2
        self.reparam_conv2.bias.data = bias2

        if not self.identity:
            kernel3, bias3 = self._fuse_bn_tensor(self.skip.conv, self.skip.bn)
            self.reparam_skip = nn.Sequential(
                OrderedDict([
                    ('conv', layer.Conv2d(self.in_channels, self.out_channels, kernel_size=1, stride=self.stride, bias=True, step_mode=self.skip.conv.step_mode).to(self.skip.conv.weight.device)),
                    ('sn', neuron.ParametricLIFNode(v_threshold=1.0, detach_reset=True, surrogate_function=surrogate.ATan(), step_mode=self.skip.sn.step_mode).to(self.skip.conv.weight.device))
                ])
            )
            self.reparam_skip.conv.weight.data = kernel3
            self.reparam_skip.conv.bias.data = bias3
            self.reparam_skip.sn.w.data = self.skip.sn.w.data
        else:
            self.reparam_skip = nn.Identity().to(self.conv1.weight.device)

        if not self.dsnn:
            self.__delattr__('aac')
        elif isinstance(self.aac, AACBlock):
            self.aac.switch_to_deploy()

        self.__delattr__('conv1')
        self.__delattr__('bn1')
        self.__delattr__('conv2')
        self.__delattr__('bn2')

        self.deploy=True
            
class S7BNet(nn.Module):
    def __init__(self, cfg_dict, num_classes, deploy=False, dsnn=False):
        super(S7BNet, self).__init__()
        self.deploy = deploy
        self.dsnn = dsnn
        if cfg_dict['block_type'] == 'spike':
            self.block = SpikeResNetBlock
        elif cfg_dict['block_type'] == 'sew':
            self.block = partial(SEWResNetBlock, cnf=cfg_dict['cnf'])

        in_channels=2
        layer_list = cfg_dict['layers']
        convs = nn.Sequential()
        for layer_dict in layer_list:
            convs.extend(self._build_layer(in_channels, layer_dict))
            in_channels = layer_dict['channels']
        self.convs = convs

        self.flatten = nn.Flatten(2)

        with torch.no_grad():
            x=torch.zeros(1,1,128,128)
            for m in self.convs.modules():
                if isinstance(m, DSTMaxPool2dBlock):
                    x = m(x)
            x = x[0] if isinstance(x, tuple) else x
            out_features = x.numel() * in_channels
        self.fc = layer.Linear(out_features, num_classes, bias=True)
        if not self.deploy or self.dsnn:
            self.aac = layer.Linear(out_features, num_classes, bias=True)

    def _build_layer(self, in_channels, layer_dict):
        channels = layer_dict.get('channels', in_channels)
        convs = nn.Sequential()
        if channels != in_channels:
            convs.append(DSTUpChannelBlock(in_channels, channels, deploy=self.deploy, dsnn=self.dsnn))
        for _ in range(layer_dict['num_blocks']):
            convs.append(self.block(channels, channels, stride=1, deploy=self.deploy, dsnn=self.dsnn))
        if 'k_pool' in layer_dict.keys():
            convs.append(DSTMaxPool2dBlock(layer_dict['k_pool'], dsnn=self.dsnn))
        return convs

    def forward(self, x):
        x = x.permute(1,0,2,3,4)
        if self.deploy and not self.dsnn:
            x = self.convs(x)
            x = self.flatten(x)
            x = self.fc(x.mean(0))
            return x
        x = self.convs((x,None))
        x,y = x
        x = self.flatten(x)
        x = self.fc(x.mean(0))
        if y is not None:
            y = self.flatten(y)
            y = self.aac(y.mean(0))
        return x,y
    
    def switch_to_deploy(self):
        if self.deploy:
            return
        for m in self.convs.modules():
            if isinstance(m, (SpikeResNetBlock, SEWResNetBlock, DSTUpChannelBlock)):
                m.switch_to_deploy()
        if not self.dsnn:
            self.__delattr__('aac')
        self.deploy=True

def get_7BNet(num_classes, block_type='sew',cnf=None,deploy=False, dsnn=False):
    cfg_dict = {
        'block_type': block_type,
        'cnf': cnf,
        'layers': [
            {'channels':32, 'num_blocks':1, 'k_pool':2},
            {'channels':32, 'num_blocks':1, 'k_pool':2},
            {'channels':32, 'num_blocks':1, 'k_pool':2},
            {'channels':32, 'num_blocks':1, 'k_pool':2},
            {'channels':32, 'num_blocks':1, 'k_pool':2},
            {'channels':32, 'num_blocks':1, 'k_pool':2},
            {'channels':32, 'num_blocks':1, 'k_pool':2},
        ]
    }
    return S7BNet(cfg_dict, num_classes, deploy=deploy, dsnn=dsnn)

def get_7BNet_wide(num_classes, block_type='sew',cnf=None,deploy=False, dsnn=False):
    cfg_dict = {
        'block_type': block_type,
        'cnf': cnf,
        'layers': [
            {'channels':64, 'num_blocks':1, 'k_pool':2},
            {'channels':64, 'num_blocks':1, 'k_pool':2},
            {'channels':64, 'num_blocks':1, 'k_pool':2},
            {'channels':64, 'num_blocks':1, 'k_pool':2},
            {'channels':128, 'num_blocks':1, 'k_pool':2},
            {'channels':128, 'num_blocks':1, 'k_pool':2},
            {'channels':128, 'num_blocks':1, 'k_pool':2},
        ]
    }
    return S7BNet(cfg_dict, num_classes, deploy=deploy, dsnn=dsnn)

model_dict = {
    '7BNet': get_7BNet,
    '7BNet_wide': get_7BNet_wide
}

def get_model_by_name(name):
    return model_dict[name]
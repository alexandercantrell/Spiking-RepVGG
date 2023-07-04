# --------------------------------------------------------
# RepVGG: Making VGG-style ConvNets Great Again (https://openaccess.thecvf.com/content/CVPR2021/papers/Ding_IDSpikingRepVGG_Making_VGG-Style_ConvNets_Great_Again_CVPR_2021_paper.pdf)
# Github source: https://github.com/DingXiaoH/RepVGG
# Licensed under The MIT License [see LICENSE for details]
# --------------------------------------------------------
import torch.nn as nn
import torch
from copy import deepcopy
import torch.utils.checkpoint as checkpoint
from spikingjelly.activation_based import layer
from models.cnf import ConnectingFunction

def conv_bn(in_channels, out_channels, kernel_size, stride, padding, groups=1):
    result = nn.Sequential()
    result.add_module('conv', layer.Conv2d(in_channels=in_channels, out_channels=out_channels,
                                                  kernel_size=kernel_size, stride=stride, padding=padding, groups=groups, bias=False))
    result.add_module('bn', layer.BatchNorm2d(num_features=out_channels))
    return result

class IDSpikingRepVGGBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size,
                 stride=1, padding=0, dilation=1, groups=1, padding_mode='zeros', deploy=False, cnf = None, spiking_neuron = None, **kwargs):
        super(IDSpikingRepVGGBlock, self).__init__()
        self.deploy = deploy
        self.groups = groups
        self.in_channels = in_channels
        self.is_downsample = not (out_channels==in_channels and stride==1)

        assert kernel_size == 3
        assert padding == 1

        padding_11 = padding - kernel_size // 2

        if deploy:
            self.rbr_reparam = layer.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride,
                                      padding=padding, dilation=dilation, groups=groups, bias=True, padding_mode=padding_mode)
            if self.is_downsample:
                self.identity = nn.Sequential()
                self.identity.add_module('conv', layer.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=stride,padding=padding_11, groups=groups))
                self.identity.add_module('sn', spiking_neuron(**deepcopy(kwargs)))
            else:
                self.identity = nn.Identity()
        else:
            self.rbr_dense = conv_bn(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding, groups=groups)
            self.rbr_1x1 = conv_bn(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=stride, padding=padding_11, groups=groups)
            if self.is_downsample:
                self.identity = nn.Sequential()
                self.identity.add_module('conv', conv_bn(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=stride, padding=padding_11, groups=groups))
                self.identity.add_module('sn', spiking_neuron(**deepcopy(kwargs)))
            else:
                self.identity = nn.Identity()

        self.sn = spiking_neuron(**deepcopy(kwargs))
        self.cnf = ConnectingFunction(cnf)


    def forward(self, inputs):
        if hasattr(self, 'rbr_reparam'):
            out = self.sn(self.rbr_reparam(inputs))
        else:
            out = self.sn(self.rbr_dense(inputs) + self.rbr_1x1(inputs))

        out = self.cnf(self.identity(inputs),out)

        return out



#   This func derives the equivalent kernel and bias in a DIFFERENTIABLE way.
#   You can get the equivalent kernel and bias at any time and do whatever you want,
    #   for example, apply some penalties or constraints during training, just like you do to the other models.
#   May be useful for quantization or pruning.
    def get_equivalent_kernel_bias(self):
        kernel3x3, bias3x3 = self._fuse_bn_tensor(self.rbr_dense)
        kernel1x1, bias1x1 = self._fuse_bn_tensor(self.rbr_1x1)
        return kernel3x3 + self._pad_1x1_to_3x3_tensor(kernel1x1), bias3x3 + bias1x1

    def _pad_1x1_to_3x3_tensor(self, kernel1x1):
        if kernel1x1 is None:
            return 0
        else:
            return torch.nn.functional.pad(kernel1x1, [1,1,1,1])

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
        self.rbr_reparam = layer.Conv2d(in_channels=self.rbr_dense.conv.in_channels, out_channels=self.rbr_dense.conv.out_channels,
                                     kernel_size=self.rbr_dense.conv.kernel_size, stride=self.rbr_dense.conv.stride,
                                     padding=self.rbr_dense.conv.padding, dilation=self.rbr_dense.conv.dilation, groups=self.rbr_dense.conv.groups, bias=True)
        self.rbr_reparam.weight.data = kernel
        self.rbr_reparam.bias.data = bias
        if self.is_downsample:
            sn = deepcopy(self.identity.sn)
            kernel, bias = self._fuse_bn_tensor(self.identity.conv)
            self.identity = nn.Sequential()
            self.identity.add_module('conv', layer.Conv2d(in_channels=self.identity.conv.in_channels, out_channels=self.identity.conv.out_channels,
                                    kernel_size=self.identity.conv.kernel_size, stride=self.identity.conv.stride,padding=self.identity.conv.padding, 
                                    dilation=self.identity.conv.dilation, groups=self.identity.conv.groups, bias=True))
            self.identity.add_module('sn', sn)
            self.identity.conv.weight.data = kernel
            self.identity.conv.bias.data = bias
        for para in self.parameters():
            para.detach_()
        self.__delattr__('rbr_dense')
        self.__delattr__('rbr_1x1')
        self.deploy = True



class IDSpikingRepVGG(nn.Module):

    def __init__(self, num_blocks, num_classes=1000, width_multiplier=None, override_groups_map=None, deploy=False, use_checkpoint=False, cnf=None, spiking_neuron=None, zero_init_residual=True, **kwargs):
        super(IDSpikingRepVGG, self).__init__()
        assert len(width_multiplier) == 4
        self.deploy = deploy
        self.override_groups_map = override_groups_map or dict()
        assert 0 not in self.override_groups_map
        self.use_checkpoint = use_checkpoint

        self.in_planes = min(64, int(64 * width_multiplier[0]))
        self.stage0 = IDSpikingRepVGGBlock(in_channels=3, out_channels=self.in_planes, kernel_size=3, stride=2, padding=1, deploy=self.deploy, spiking_neuron=spiking_neuron, **kwargs)
        self.cur_layer_idx = 1
        self.stage1 = self._make_stage(int(64 * width_multiplier[0]), num_blocks[0], stride=2, cnf=cnf, spiking_neuron=spiking_neuron, **kwargs)
        self.stage2 = self._make_stage(int(128 * width_multiplier[1]), num_blocks[1], stride=2, cnf=cnf, spiking_neuron=spiking_neuron, **kwargs)
        self.stage3 = self._make_stage(int(256 * width_multiplier[2]), num_blocks[2], stride=2, cnf=cnf, spiking_neuron=spiking_neuron, **kwargs)
        self.stage4 = self._make_stage(int(512 * width_multiplier[3]), num_blocks[3], stride=2, cnf=cnf, spiking_neuron=spiking_neuron, **kwargs)
        self.gap = layer.AdaptiveAvgPool2d((1,1))
        self.linear = layer.Linear(int(512 * width_multiplier[3]), num_classes)

        for m in self.modules():
            if isinstance(m, layer.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (layer.BatchNorm2d, layer.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, IDSpikingRepVGGBlock) and m.cnf is not None:
                    if m.rbr_1x1 is not None:
                        nn.init.constant_(m.rbr_1x1.bn.weight,0)
                        if cnf == 'AND':
                            nn.init.constant_(m.rbr_1x1.bn.bias,1)
                    if m.rbr_dense is not None:
                        nn.init.constant_(m.rbr_dense.bn.weight,0)
                        if cnf == 'AND':
                            nn.init.constant_(m.rbr_dense.bn.bias,1)

    def _make_stage(self, planes, num_blocks, stride, cnf, spiking_neuron, **kwargs):
        strides = [stride] + [1]*(num_blocks-1)
        blocks = []
        for stride in strides:
            cur_groups = self.override_groups_map.get(self.cur_layer_idx, 1)
            blocks.append(IDSpikingRepVGGBlock(in_channels=self.in_planes, out_channels=planes, kernel_size=3,
                                      stride=stride, padding=1, groups=cur_groups, deploy=self.deploy, cnf=cnf, spiking_neuron=spiking_neuron, **kwargs))
            self.in_planes = planes
            self.cur_layer_idx += 1
        return nn.ModuleList(blocks)

    def forward(self, x):
        out = self.stage0(x)
        for stage in (self.stage1, self.stage2, self.stage3, self.stage4):
            for block in stage:
                if self.use_checkpoint:
                    out = checkpoint.checkpoint(block, out)
                else:
                    out = block(out)
        out = self.gap(out)
        if self.gap.step_mode == 's':
            out = torch.flatten(out, 1)
        elif self.gap.step_mode == 'm':
            out = torch.flatten(out, 2)
        out = self.linear(out)
        return out


optional_groupwise_layers = [2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26]
g2_map = {l: 2 for l in optional_groupwise_layers}
g4_map = {l: 4 for l in optional_groupwise_layers}

def create_IDSpikingRepVGG_A0(num_classes=1000, deploy=False, use_checkpoint=False,cnf=None,spiking_neuron=None,**kwargs):
    return IDSpikingRepVGG(num_blocks=[2, 4, 14, 1], num_classes=num_classes,
                  width_multiplier=[0.75, 0.75, 0.75, 2.5], override_groups_map=None, deploy=deploy, use_checkpoint=use_checkpoint,cnf=cnf,spiking_neuron=spiking_neuron,**kwargs)

def create_IDSpikingRepVGG_A1(num_classes=1000, deploy=False, use_checkpoint=False,cnf=None,spiking_neuron=None,**kwargs):
    return IDSpikingRepVGG(num_blocks=[2, 4, 14, 1], num_classes=num_classes,
                  width_multiplier=[1, 1, 1, 2.5], override_groups_map=None, deploy=deploy, use_checkpoint=use_checkpoint,cnf=cnf,spiking_neuron=spiking_neuron,**kwargs)

def create_IDSpikingRepVGG_A2(num_classes=1000, deploy=False, use_checkpoint=False,cnf=None,spiking_neuron=None,**kwargs):
    return IDSpikingRepVGG(num_blocks=[2, 4, 14, 1], num_classes=num_classes,
                  width_multiplier=[1.5, 1.5, 1.5, 2.75], override_groups_map=None, deploy=deploy, use_checkpoint=use_checkpoint,cnf=cnf,spiking_neuron=spiking_neuron,**kwargs)

def create_IDSpikingRepVGG_B0(num_classes=1000, deploy=False, use_checkpoint=False,cnf=None,spiking_neuron=None,**kwargs):
    return IDSpikingRepVGG(num_blocks=[4, 6, 16, 1], num_classes=num_classes,
                  width_multiplier=[1, 1, 1, 2.5], override_groups_map=None, deploy=deploy, use_checkpoint=use_checkpoint,cnf=cnf,spiking_neuron=spiking_neuron,**kwargs)

def create_IDSpikingRepVGG_B1(num_classes=1000, deploy=False, use_checkpoint=False,cnf=None,spiking_neuron=None,**kwargs):
    return IDSpikingRepVGG(num_blocks=[4, 6, 16, 1], num_classes=num_classes,
                  width_multiplier=[2, 2, 2, 4], override_groups_map=None, deploy=deploy, use_checkpoint=use_checkpoint,cnf=cnf,spiking_neuron=spiking_neuron,**kwargs)

def create_IDSpikingRepVGG_B1g2(num_classes=1000, deploy=False, use_checkpoint=False,cnf=None,spiking_neuron=None,**kwargs):
    return IDSpikingRepVGG(num_blocks=[4, 6, 16, 1], num_classes=num_classes,
                  width_multiplier=[2, 2, 2, 4], override_groups_map=g2_map, deploy=deploy, use_checkpoint=use_checkpoint,cnf=cnf,spiking_neuron=spiking_neuron,**kwargs)

def create_IDSpikingRepVGG_B1g4(num_classes=1000, deploy=False, use_checkpoint=False,cnf=None,spiking_neuron=None,**kwargs):
    return IDSpikingRepVGG(num_blocks=[4, 6, 16, 1], num_classes=num_classes,
                  width_multiplier=[2, 2, 2, 4], override_groups_map=g4_map, deploy=deploy, use_checkpoint=use_checkpoint,cnf=cnf,spiking_neuron=spiking_neuron,**kwargs)


def create_IDSpikingRepVGG_B2(num_classes=1000, deploy=False, use_checkpoint=False,cnf=None,spiking_neuron=None,**kwargs):
    return IDSpikingRepVGG(num_blocks=[4, 6, 16, 1], num_classes=num_classes,
                  width_multiplier=[2.5, 2.5, 2.5, 5], override_groups_map=None, deploy=deploy, use_checkpoint=use_checkpoint,cnf=cnf,spiking_neuron=spiking_neuron,**kwargs)

def create_IDSpikingRepVGG_B2g2(num_classes=1000, deploy=False, use_checkpoint=False,cnf=None,spiking_neuron=None,**kwargs):
    return IDSpikingRepVGG(num_blocks=[4, 6, 16, 1], num_classes=num_classes,
                  width_multiplier=[2.5, 2.5, 2.5, 5], override_groups_map=g2_map, deploy=deploy, use_checkpoint=use_checkpoint,cnf=cnf,spiking_neuron=spiking_neuron,**kwargs)

def create_IDSpikingRepVGG_B2g4(num_classes=1000, deploy=False, use_checkpoint=False,cnf=None,spiking_neuron=None,**kwargs):
    return IDSpikingRepVGG(num_blocks=[4, 6, 16, 1], num_classes=num_classes,
                  width_multiplier=[2.5, 2.5, 2.5, 5], override_groups_map=g4_map, deploy=deploy, use_checkpoint=use_checkpoint,cnf=cnf,spiking_neuron=spiking_neuron,**kwargs)


def create_IDSpikingRepVGG_B3(num_classes=1000, deploy=False, use_checkpoint=False,cnf=None,spiking_neuron=None,**kwargs):
    return IDSpikingRepVGG(num_blocks=[4, 6, 16, 1], num_classes=num_classes,
                  width_multiplier=[3, 3, 3, 5], override_groups_map=None, deploy=deploy, use_checkpoint=use_checkpoint,cnf=cnf,spiking_neuron=spiking_neuron,**kwargs)

def create_IDSpikingRepVGG_B3g2(num_classes=1000, deploy=False, use_checkpoint=False,cnf=None,spiking_neuron=None,**kwargs):
    return IDSpikingRepVGG(num_blocks=[4, 6, 16, 1], num_classes=num_classes,
                  width_multiplier=[3, 3, 3, 5], override_groups_map=g2_map, deploy=deploy, use_checkpoint=use_checkpoint,cnf=cnf,spiking_neuron=spiking_neuron,**kwargs)

def create_IDSpikingRepVGG_B3g4(num_classes=1000, deploy=False, use_checkpoint=False,cnf=None,spiking_neuron=None,**kwargs):
    return IDSpikingRepVGG(num_blocks=[4, 6, 16, 1], num_classes=num_classes,
                  width_multiplier=[3, 3, 3, 5], override_groups_map=g4_map, deploy=deploy, use_checkpoint=use_checkpoint,cnf=cnf,spiking_neuron=spiking_neuron,**kwargs)

def create_IDSpikingRepVGG_D2se(num_classes=1000, deploy=False, use_checkpoint=False,cnf=None,spiking_neuron=None,**kwargs):
    return IDSpikingRepVGG(num_blocks=[8, 14, 24, 1], num_classes=num_classes,
                  width_multiplier=[2.5, 2.5, 2.5, 5], override_groups_map=None, deploy=deploy, use_checkpoint=use_checkpoint,cnf=cnf,spiking_neuron=spiking_neuron,**kwargs)


func_dict = {
'IDSpikingRepVGG-A0': create_IDSpikingRepVGG_A0,
'IDSpikingRepVGG-A1': create_IDSpikingRepVGG_A1,
'IDSpikingRepVGG-A2': create_IDSpikingRepVGG_A2,
'IDSpikingRepVGG-B0': create_IDSpikingRepVGG_B0,
'IDSpikingRepVGG-B1': create_IDSpikingRepVGG_B1,
'IDSpikingRepVGG-B1g2': create_IDSpikingRepVGG_B1g2,
'IDSpikingRepVGG-B1g4': create_IDSpikingRepVGG_B1g4,
'IDSpikingRepVGG-B2': create_IDSpikingRepVGG_B2,
'IDSpikingRepVGG-B2g2': create_IDSpikingRepVGG_B2g2,
'IDSpikingRepVGG-B2g4': create_IDSpikingRepVGG_B2g4,
'IDSpikingRepVGG-B3': create_IDSpikingRepVGG_B3,
'IDSpikingRepVGG-B3g2': create_IDSpikingRepVGG_B3g2,
'IDSpikingRepVGG-B3g4': create_IDSpikingRepVGG_B3g4,
'IDSpikingRepVGG-D2se': create_IDSpikingRepVGG_D2se,      #   Updated at April 25, 2021. This is not reported in the CVPR paper.
}
def get_IDSpikingRepVGG_func_by_name(name):
    return func_dict[name]


# --------------------------------------------------------
# StaticSpikingRepVGG: Making VGG-style ConvNets Great Again (https://openaccess.thecvf.com/content/CVPR2021/papers/Ding_StaticSpikingRepVGG_Making_VGG-Style_ConvNets_Great_Again_CVPR_2021_paper.pdf)
# Github source: https://github.com/DingXiaoH/RepVGG
# Licensed under The MIT License [see LICENSE for details]
# --------------------------------------------------------
import torch.nn as nn
import torch
from copy import deepcopy
import torch.utils.checkpoint as checkpoint
from spikingjelly.activation_based import layer
from utils import conv_bn
from models.connecting_function import ConnectingFunction

class StaticSpikingRepVGGBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size,
                 stride=1, padding=0, dilation=1, groups=1, padding_mode='zeros', deploy=False, use_se=False, cnf = None, spiking_neuron=None):
        super(StaticSpikingRepVGGBlock, self).__init__()
        self.deploy = deploy
        self.groups = groups
        self.in_channels = in_channels

        assert kernel_size == 3
        assert padding == 1

        padding_11 = padding - kernel_size // 2

        if use_se:
            raise NotImplementedError("se block not supported yet")
        else:
            self.se = nn.Identity()

        if deploy:
            self.rbr_reparam = layer.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride,
                                      padding=padding, dilation=dilation, groups=groups, bias=True, padding_mode=padding_mode)

        else:
            self.rbr_dense = conv_bn(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding, groups=groups)
            self.rbr_1x1 = conv_bn(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=stride, padding=padding_11, groups=groups)

        if (out_channels == in_channels and stride == 1):
            self.identity=nn.Identity()
            self.cnf = ConnectingFunction(cnf)
        else:
            self.identity = None
            self.cnf = None

        self.sn = spiking_neuron
        


    def forward(self, inputs):
        if hasattr(self, 'rbr_reparam'):
            out = self.sn.single_step_forward(self.se(self.rbr_reparam(inputs)))
        else:
            out = self.sn.single_step_forward(self.se(self.rbr_dense(inputs) + self.rbr_1x1(inputs)))

        if self.identity is not None:
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
        for para in self.parameters():
            para.detach_()
        self.__delattr__('rbr_dense')
        self.__delattr__('rbr_1x1')
        self.deploy = True



class StaticSpikingRepVGG(nn.Module):

    def __init__(self, num_blocks, num_classes=1000, width_multiplier=None, override_groups_map=None, deploy=False, use_se=False, use_checkpoint=False, cnf=None, spiking_neuron=None, **kwargs):
        super(StaticSpikingRepVGG, self).__init__()
        assert len(width_multiplier) == 4
        self.deploy = deploy
        self.override_groups_map = override_groups_map or dict()
        assert 0 not in self.override_groups_map
        self.use_se = use_se
        self.use_checkpoint = use_checkpoint

        self.in_planes = min(64, int(64 * width_multiplier[0]))
        self.sn0 = spiking_neuron(**deepcopy(kwargs))
        self.stage0 = StaticSpikingRepVGGBlock(in_channels=3, out_channels=self.in_planes, kernel_size=3, stride=2, padding=1, deploy=self.deploy, use_se=self.use_se,spiking_neuron=self.sn0)
        self.cur_layer_idx = 1
        self.stage1, self.sn1 = self._make_stage(int(64 * width_multiplier[0]), num_blocks[0], stride=2, cnf=cnf, spiking_neuron=spiking_neuron, **kwargs)
        self.stage2, self.sn2 = self._make_stage(int(128 * width_multiplier[1]), num_blocks[1], stride=2, cnf=cnf, spiking_neuron=spiking_neuron, **kwargs)
        self.stage3, self.sn3 = self._make_stage(int(256 * width_multiplier[2]), num_blocks[2], stride=2, cnf=cnf, spiking_neuron=spiking_neuron, **kwargs)
        self.stage4, self.sn4 = self._make_stage(int(512 * width_multiplier[3]), num_blocks[3], stride=2, cnf=cnf, spiking_neuron=spiking_neuron, **kwargs)
        self.gap = layer.AdaptiveAvgPool2d((1,1))
        self.linear = layer.Linear(int(512 * width_multiplier[3]), num_classes)

    def _make_stage(self, planes, num_blocks, stride, cnf, spiking_neuron, **kwargs):
        sn = spiking_neuron(**deepcopy(kwargs))
        strides = [stride] + [1]*(num_blocks-1)
        blocks = []
        for stride in strides:
            cur_groups = self.override_groups_map.get(self.cur_layer_idx, 1)
            blocks.append(StaticSpikingRepVGGBlock(in_channels=self.in_planes, out_channels=planes, kernel_size=3,
                                      stride=stride, padding=1, groups=cur_groups, deploy=self.deploy, use_se=self.use_se, cnf=cnf, spiking_neuron=sn))
            self.in_planes = planes
            self.cur_layer_idx += 1
        return (nn.ModuleList(blocks), sn)

    def forward(self, x):
        out = self.stage0(x)
        self.sn0.reset()
        stages = ((self.stage1,self.sn1),(self.stage2,self.sn2),(self.stage3,self.sn3),(self.stage4,self.sn4))
        #for stage in (self.stage1, self.stage2, self.stage3, self.stage4):
        for stage in stages:
            for block in stage[0]:
                if self.use_checkpoint:
                    out = checkpoint.checkpoint(block, out)
                else:
                    out = block(out)
            stage[1].reset()
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

def create_StaticSpikingRepVGG_A0(deploy=False, use_checkpoint=False,cnf=None,spiking_neuron=None,**kwargs):
    return StaticSpikingRepVGG(num_blocks=[2, 4, 14, 1], num_classes=1000,
                  width_multiplier=[0.75, 0.75, 0.75, 2.5], override_groups_map=None, deploy=deploy, use_checkpoint=use_checkpoint,cnf=cnf,spiking_neuron=spiking_neuron,**kwargs)

def create_StaticSpikingRepVGG_A1(deploy=False, use_checkpoint=False,cnf=None,spiking_neuron=None,**kwargs):
    return StaticSpikingRepVGG(num_blocks=[2, 4, 14, 1], num_classes=1000,
                  width_multiplier=[1, 1, 1, 2.5], override_groups_map=None, deploy=deploy, use_checkpoint=use_checkpoint,cnf=cnf,spiking_neuron=spiking_neuron,**kwargs)

def create_StaticSpikingRepVGG_A2(deploy=False, use_checkpoint=False,cnf=None,spiking_neuron=None,**kwargs):
    return StaticSpikingRepVGG(num_blocks=[2, 4, 14, 1], num_classes=1000,
                  width_multiplier=[1.5, 1.5, 1.5, 2.75], override_groups_map=None, deploy=deploy, use_checkpoint=use_checkpoint,cnf=cnf,spiking_neuron=spiking_neuron,**kwargs)

def create_StaticSpikingRepVGG_B0(deploy=False, use_checkpoint=False,cnf=None,spiking_neuron=None,**kwargs):
    return StaticSpikingRepVGG(num_blocks=[4, 6, 16, 1], num_classes=1000,
                  width_multiplier=[1, 1, 1, 2.5], override_groups_map=None, deploy=deploy, use_checkpoint=use_checkpoint,cnf=cnf,spiking_neuron=spiking_neuron,**kwargs)

def create_StaticSpikingRepVGG_B1(deploy=False, use_checkpoint=False,cnf=None,spiking_neuron=None,**kwargs):
    return StaticSpikingRepVGG(num_blocks=[4, 6, 16, 1], num_classes=1000,
                  width_multiplier=[2, 2, 2, 4], override_groups_map=None, deploy=deploy, use_checkpoint=use_checkpoint,cnf=cnf,spiking_neuron=spiking_neuron,**kwargs)

def create_StaticSpikingRepVGG_B1g2(deploy=False, use_checkpoint=False,cnf=None,spiking_neuron=None,**kwargs):
    return StaticSpikingRepVGG(num_blocks=[4, 6, 16, 1], num_classes=1000,
                  width_multiplier=[2, 2, 2, 4], override_groups_map=g2_map, deploy=deploy, use_checkpoint=use_checkpoint,cnf=cnf,spiking_neuron=spiking_neuron,**kwargs)

def create_StaticSpikingRepVGG_B1g4(deploy=False, use_checkpoint=False,cnf=None,spiking_neuron=None,**kwargs):
    return StaticSpikingRepVGG(num_blocks=[4, 6, 16, 1], num_classes=1000,
                  width_multiplier=[2, 2, 2, 4], override_groups_map=g4_map, deploy=deploy, use_checkpoint=use_checkpoint,cnf=cnf,spiking_neuron=spiking_neuron,**kwargs)


def create_StaticSpikingRepVGG_B2(deploy=False, use_checkpoint=False,cnf=None,spiking_neuron=None,**kwargs):
    return StaticSpikingRepVGG(num_blocks=[4, 6, 16, 1], num_classes=1000,
                  width_multiplier=[2.5, 2.5, 2.5, 5], override_groups_map=None, deploy=deploy, use_checkpoint=use_checkpoint,cnf=cnf,spiking_neuron=spiking_neuron,**kwargs)

def create_StaticSpikingRepVGG_B2g2(deploy=False, use_checkpoint=False,cnf=None,spiking_neuron=None,**kwargs):
    return StaticSpikingRepVGG(num_blocks=[4, 6, 16, 1], num_classes=1000,
                  width_multiplier=[2.5, 2.5, 2.5, 5], override_groups_map=g2_map, deploy=deploy, use_checkpoint=use_checkpoint,cnf=cnf,spiking_neuron=spiking_neuron,**kwargs)

def create_StaticSpikingRepVGG_B2g4(deploy=False, use_checkpoint=False,cnf=None,spiking_neuron=None,**kwargs):
    return StaticSpikingRepVGG(num_blocks=[4, 6, 16, 1], num_classes=1000,
                  width_multiplier=[2.5, 2.5, 2.5, 5], override_groups_map=g4_map, deploy=deploy, use_checkpoint=use_checkpoint,cnf=cnf,spiking_neuron=spiking_neuron,**kwargs)


def create_StaticSpikingRepVGG_B3(deploy=False, use_checkpoint=False,cnf=None,spiking_neuron=None,**kwargs):
    return StaticSpikingRepVGG(num_blocks=[4, 6, 16, 1], num_classes=1000,
                  width_multiplier=[3, 3, 3, 5], override_groups_map=None, deploy=deploy, use_checkpoint=use_checkpoint,cnf=cnf,spiking_neuron=spiking_neuron,**kwargs)

def create_StaticSpikingRepVGG_B3g2(deploy=False, use_checkpoint=False,cnf=None,spiking_neuron=None,**kwargs):
    return StaticSpikingRepVGG(num_blocks=[4, 6, 16, 1], num_classes=1000,
                  width_multiplier=[3, 3, 3, 5], override_groups_map=g2_map, deploy=deploy, use_checkpoint=use_checkpoint,cnf=cnf,spiking_neuron=spiking_neuron,**kwargs)

def create_StaticSpikingRepVGG_B3g4(deploy=False, use_checkpoint=False,cnf=None,spiking_neuron=None,**kwargs):
    return StaticSpikingRepVGG(num_blocks=[4, 6, 16, 1], num_classes=1000,
                  width_multiplier=[3, 3, 3, 5], override_groups_map=g4_map, deploy=deploy, use_checkpoint=use_checkpoint,cnf=cnf,spiking_neuron=spiking_neuron,**kwargs)

def create_StaticSpikingRepVGG_D2se(deploy=False, use_checkpoint=False,cnf=None,spiking_neuron=None,**kwargs):
    return StaticSpikingRepVGG(num_blocks=[8, 14, 24, 1], num_classes=1000,
                  width_multiplier=[2.5, 2.5, 2.5, 5], override_groups_map=None, deploy=deploy, use_se=True, use_checkpoint=use_checkpoint,cnf=cnf,spiking_neuron=spiking_neuron,**kwargs)


func_dict = {
'StaticSpikingRepVGG-A0': create_StaticSpikingRepVGG_A0,
'StaticSpikingRepVGG-A1': create_StaticSpikingRepVGG_A1,
'StaticSpikingRepVGG-A2': create_StaticSpikingRepVGG_A2,
'StaticSpikingRepVGG-B0': create_StaticSpikingRepVGG_B0,
'StaticSpikingRepVGG-B1': create_StaticSpikingRepVGG_B1,
'StaticSpikingRepVGG-B1g2': create_StaticSpikingRepVGG_B1g2,
'StaticSpikingRepVGG-B1g4': create_StaticSpikingRepVGG_B1g4,
'StaticSpikingRepVGG-B2': create_StaticSpikingRepVGG_B2,
'StaticSpikingRepVGG-B2g2': create_StaticSpikingRepVGG_B2g2,
'StaticSpikingRepVGG-B2g4': create_StaticSpikingRepVGG_B2g4,
'StaticSpikingRepVGG-B3': create_StaticSpikingRepVGG_B3,
'StaticSpikingRepVGG-B3g2': create_StaticSpikingRepVGG_B3g2,
'StaticSpikingRepVGG-B3g4': create_StaticSpikingRepVGG_B3g4,
'StaticSpikingRepVGG-D2se': create_StaticSpikingRepVGG_D2se,      #   Updated at April 25, 2021. This is not reported in the CVPR paper.
}
def get_StaticSpikingRepVGG_func_by_name(name):
    return func_dict[name]


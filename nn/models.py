from copy import deepcopy
import torch.nn as nn
import torch
from spikingjelly.activation_based import layer
from layers import SpikingRepVGGBlock, get_block_by_name

class SpikingRepVGG(nn.Module):
    def __init__(self, block: SpikingRepVGGBlock, num_blocks, num_classes=1000, width_multipliers=None,
                 override_groups_map=None, deploy=False, spiking_neuron:callable=None, reuse_neurons=False,
                 zero_init_residual=True, **kwargs):
        super(SpikingRepVGG, self).__init__()
        assert (len(width_multipliers) == 4)
        self.deploy = deploy
        self.override_groups_map = override_groups_map or dict()
        assert 0 not in self.override_groups_map
        self.in_planes = min(64, int(64 * width_multipliers[0]))
        self.stage0 = block(
            in_channels=3, out_channels=self.in_planes, kernel_size=3, stride=2, 
            padding=1, deploy=self.deploy, spiking_neuron=spiking_neuron, **kwargs)
        self.cur_layer_idx = 1
        self.stage1 = self._make_stage(block, int(64 * width_multipliers[0]), num_blocks[0], stride=2, spiking_neuron=spiking_neuron, reuse_neuron=reuse_neurons, **kwargs)
        self.stage2 = self._make_stage(block, int(128 * width_multipliers[1]), num_blocks[1], stride=2, spiking_neuron=spiking_neuron, reuse_neuron=reuse_neurons, **kwargs)
        self.stage3 = self._make_stage(block, int(256 * width_multipliers[2]), num_blocks[2], stride=2, spiking_neuron=spiking_neuron, reuse_neuron=reuse_neurons, **kwargs)
        self.stage4 = self._make_stage(block, int(512 * width_multipliers[3]), num_blocks[3], stride=2, spiking_neuron=spiking_neuron, reuse_neuron=reuse_neurons, **kwargs)
        self.gap = layer.AdaptiveAvgPool2d((1, 1))
        self.linear = layer.Linear(int(512 * width_multipliers[3]), num_classes)

        for m in self.modules():
            if isinstance(m, layer.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (layer.BatchNorm2d, layer.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        if zero_init_residual and not deploy:
            for m in self.modules():
                if isinstance(m, block) and m.rbr_identity is not None:
                    if hasattr(m, 'rbr_dense'):
                        nn.init.constant_(m.rbr_dense.bn.weight, 0)
                    if hasattr(m, 'bn'):
                        nn.init.constant_(m.bn.weight, 0)
                    elif hasattr(m, 'rbr_1x1'):
                        nn.init.constant_(m.rbr_1x1.bn.weight, 0)

    def _make_stage(self, block, planes, num_blocks, stride, spiking_neuron:callable=None, reuse_neuron=False, **kwargs):
        if reuse_neuron:
            spiking_neuron = spiking_neuron(**deepcopy(kwargs))
        strides = [stride] + [1] * (num_blocks - 1)
        blocks = []
        for stride in strides:
            cur_groups = self.override_groups_map.get(self.cur_layer_idx, 1)
            blocks.append(block(
                in_channels=self.in_planes, out_channels=planes, kernel_size=3, stride=stride,
                padding=1, groups=cur_groups, deploy=self.deploy, spiking_neuron=spiking_neuron, **kwargs))
            self.in_planes = planes
            self.cur_layer_idx += 1
        return nn.Sequential(*blocks)
    
    def forward(self, x):
        out = self.stage0(x)
        out = self.stage1(out)
        out = self.stage2(out)
        out = self.stage3(out)
        out = self.stage4(out)
        out = self.gap(out)
        if self.gap.step_mode == 's':
            out = torch.flatten(out, 1)
        elif self.gap.step_mode == 'm':
            out = torch.flatten(out, 2)
        out = self.linear(out)
        return out

class SEWRepVGG(nn.Module):
    def __init__(self, block: SpikingRepVGGBlock, num_blocks, num_classes=1000, width_multipliers=None,
                    override_groups_map=None, deploy=False, cnf:str=None, 
                    spiking_neuron:callable=None, reuse_neurons=False, 
                    zero_init_residual=True, **kwargs):
        super(SEWRepVGG, self).__init__()
        assert (len(width_multipliers) == 4)
        self.deploy = deploy
        self.override_groups_map = override_groups_map or dict()
        assert 0 not in self.override_groups_map
        self.in_planes = min(64, int(64 * width_multipliers[0]))
        self.stage0 = block(
            in_channels=3, out_channels=self.in_planes, kernel_size=3, stride=2, 
            padding=1, deploy=self.deploy, cnf=cnf, spiking_neuron=spiking_neuron, **kwargs)
        self.cur_layer_idx = 1
        self.stage1 = self._make_stage(block, int(64 * width_multipliers[0]), num_blocks[0], stride=2, cnf=cnf, spiking_neuron=spiking_neuron, reuse_neuron=reuse_neurons, **kwargs)
        self.stage2 = self._make_stage(block, int(128 * width_multipliers[1]), num_blocks[1], stride=2, cnf=cnf, spiking_neuron=spiking_neuron, reuse_neuron=reuse_neurons, **kwargs)
        self.stage3 = self._make_stage(block, int(256 * width_multipliers[2]), num_blocks[2], stride=2, cnf=cnf, spiking_neuron=spiking_neuron, reuse_neuron=reuse_neurons, **kwargs)
        self.stage4 = self._make_stage(block, int(512 * width_multipliers[3]), num_blocks[3], stride=2, cnf=cnf, spiking_neuron=spiking_neuron, reuse_neuron=reuse_neurons, **kwargs)
        self.gap = layer.AdaptiveAvgPool2d((1, 1))
        self.linear = layer.Linear(int(512 * width_multipliers[3]), num_classes)

        for m in self.modules():
            if isinstance(m, layer.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (layer.BatchNorm2d, layer.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        if zero_init_residual and not deploy:
            for m in self.modules():
                if isinstance(m, block) and m.rbr_identity is not None:
                    if hasattr(m, 'rbr_dense'):
                        nn.init.constant_(m.rbr_dense.bn.weight, 0)
                        if cnf == 'AND':
                            nn.init.constant_(m.rbr_dense.bn.bias,1)
                    if hasattr(m, 'bn'):
                        nn.init.constant_(m.bn.weight, 0)
                        if cnf == 'AND':
                            nn.init.constant_(m.bn.bias,1)
                    elif hasattr(m, 'rbr_1x1'):
                        nn.init.constant_(m.rbr_1x1.bn.weight, 0)
                        if cnf == 'AND':
                            nn.init.constant_(m.rbr_1x1.bn.bias,1)

    def _make_stage(self, block, planes, num_blocks, stride, cnf:str=None, spiking_neuron:callable=None, reuse_neuron=False, **kwargs):
        if reuse_neuron:
            spiking_neuron = spiking_neuron(**deepcopy(kwargs))
        strides = [stride] + [1] * (num_blocks - 1)
        blocks = []
        for stride in strides:
            cur_groups = self.override_groups_map.get(self.cur_layer_idx, 1)
            blocks.append(block(
                in_channels=self.in_planes, out_channels=planes, kernel_size=3, stride=stride,
                padding=1, groups=cur_groups, deploy=self.deploy, cnf=cnf, spiking_neuron=spiking_neuron, **kwargs))
            self.in_planes = planes
            self.cur_layer_idx += 1
        return nn.Sequential(*blocks)
    
    def forward(self, x):
        out = self.stage0(x)
        out = self.stage1(out)
        out = self.stage2(out)
        out = self.stage3(out)
        out = self.stage4(out)
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

def create_SpikingRepVGG_A0(block, num_classes=1000, deploy=False, spiking_neuron:callable=None, reuse_neurons=False, zero_init_residual=True, **kwargs):
    return SpikingRepVGG(get_block_by_name(block), [2, 4, 14, 1], num_classes=num_classes, width_multipliers=[0.75, 0.75, 0.75, 2.5], override_groups_map=None, deploy=deploy, spiking_neuron=spiking_neuron, reuse_neurons=reuse_neurons, zero_init_residual=zero_init_residual, **kwargs)

def create_SpikingRepVGG_A1(block, num_classes=1000, deploy=False, spiking_neuron:callable=None, reuse_neurons=False, zero_init_residual=True, **kwargs):
    return SpikingRepVGG(get_block_by_name(block), [2, 4, 14, 1], num_classes=num_classes, width_multipliers=[1, 1, 1, 2.5], override_groups_map=None, deploy=deploy, spiking_neuron=spiking_neuron, reuse_neurons=reuse_neurons, zero_init_residual=zero_init_residual, **kwargs)

def create_SpikingRepVGG_A2(block, num_classes=1000, deploy=False, spiking_neuron:callable=None, reuse_neurons=False, zero_init_residual=True, **kwargs):
    return SpikingRepVGG(get_block_by_name(block), [2, 4, 14, 1], num_classes=num_classes, width_multipliers=[1.5, 1.5, 1.5, 2.75], override_groups_map=None, deploy=deploy, spiking_neuron=spiking_neuron, reuse_neurons=reuse_neurons, zero_init_residual=zero_init_residual, **kwargs)

def create_SpikingRepVGG_B0(block, num_classes=1000, deploy=False, spiking_neuron:callable=None, reuse_neurons=False, zero_init_residual=True, **kwargs):
    return SpikingRepVGG(get_block_by_name(block), [4, 6, 16, 1], num_classes=num_classes, width_multipliers=[1, 1, 1, 2.5], override_groups_map=None, deploy=deploy, spiking_neuron=spiking_neuron, reuse_neurons=reuse_neurons, zero_init_residual=zero_init_residual, **kwargs)

def create_SpikingRepVGG_B1(block, num_classes=1000, deploy=False, spiking_neuron:callable=None, reuse_neurons=False, zero_init_residual=True, **kwargs):
    return SpikingRepVGG(get_block_by_name(block), [4, 6, 16, 1], num_classes=num_classes, width_multipliers=[2, 2, 2, 4], override_groups_map=None, deploy=deploy, spiking_neuron=spiking_neuron, reuse_neurons=reuse_neurons, zero_init_residual=zero_init_residual, **kwargs)

def create_SpikingRepVGG_B1g2(block, num_classes=1000, deploy=False, spiking_neuron:callable=None, reuse_neurons=False, zero_init_residual=True, **kwargs):
    return SpikingRepVGG(get_block_by_name(block), [4, 6, 16, 1], num_classes=num_classes, width_multipliers=[2, 2, 2, 4], override_groups_map=g2_map, deploy=deploy, spiking_neuron=spiking_neuron, reuse_neurons=reuse_neurons, zero_init_residual=zero_init_residual, **kwargs)

def create_SpikingRepVGG_B1g4(block, num_classes=1000, deploy=False, spiking_neuron:callable=None, reuse_neurons=False, zero_init_residual=True, **kwargs):
    return SpikingRepVGG(get_block_by_name(block), [4, 6, 16, 1], num_classes=num_classes, width_multipliers=[2, 2, 2, 4], override_groups_map=g4_map, deploy=deploy, spiking_neuron=spiking_neuron, reuse_neurons=reuse_neurons, zero_init_residual=zero_init_residual, **kwargs)

def create_SpikingRepVGG_B2(block, num_classes=1000, deploy=False, spiking_neuron:callable=None, reuse_neurons=False, zero_init_residual=True, **kwargs):
    return SpikingRepVGG(get_block_by_name(block), [4, 6, 16, 1], num_classes=num_classes, width_multipliers=[2.5, 2.5, 2.5, 5], override_groups_map=None, deploy=deploy, spiking_neuron=spiking_neuron, reuse_neurons=reuse_neurons, zero_init_residual=zero_init_residual, **kwargs)

def create_SpikingRepVGG_B2g2(block, num_classes=1000, deploy=False, spiking_neuron:callable=None, reuse_neurons=False, zero_init_residual=True, **kwargs):
    return SpikingRepVGG(get_block_by_name(block), [4, 6, 16, 1], num_classes=num_classes, width_multipliers=[2.5, 2.5, 2.5, 5], override_groups_map=g2_map, deploy=deploy, spiking_neuron=spiking_neuron, reuse_neurons=reuse_neurons, zero_init_residual=zero_init_residual, **kwargs)

def create_SpikingRepVGG_B2g4(block, num_classes=1000, deploy=False, spiking_neuron:callable=None, reuse_neurons=False, zero_init_residual=True, **kwargs):
    return SpikingRepVGG(get_block_by_name(block), [4, 6, 16, 1], num_classes=num_classes, width_multipliers=[2.5, 2.5, 2.5, 5], override_groups_map=g4_map, deploy=deploy, spiking_neuron=spiking_neuron, reuse_neurons=reuse_neurons, zero_init_residual=zero_init_residual, **kwargs)

def create_SpikingRepVGG_B3(block, num_classes=1000, deploy=False, spiking_neuron:callable=None, reuse_neurons=False, zero_init_residual=True, **kwargs):
    return SpikingRepVGG(get_block_by_name(block), [4, 6, 16, 1], num_classes=num_classes, width_multipliers=[3, 3, 3, 5], override_groups_map=None, deploy=deploy, spiking_neuron=spiking_neuron, reuse_neurons=reuse_neurons, zero_init_residual=zero_init_residual, **kwargs)

def create_SpikingRepVGG_B3g2(block, num_classes=1000, deploy=False, spiking_neuron:callable=None, reuse_neurons=False, zero_init_residual=True, **kwargs):
    return SpikingRepVGG(get_block_by_name(block), [4, 6, 16, 1], num_classes=num_classes, width_multipliers=[3, 3, 3, 5], override_groups_map=g2_map, deploy=deploy, spiking_neuron=spiking_neuron, reuse_neurons=reuse_neurons, zero_init_residual=zero_init_residual, **kwargs)

def create_SpikingRepVGG_B3g4(block, num_classes=1000, deploy=False, spiking_neuron:callable=None, reuse_neurons=False, zero_init_residual=True, **kwargs):
    return SpikingRepVGG(get_block_by_name(block), [4, 6, 16, 1], num_classes=num_classes, width_multipliers=[3, 3, 3, 5], override_groups_map=g4_map, deploy=deploy, spiking_neuron=spiking_neuron, reuse_neurons=reuse_neurons, zero_init_residual=zero_init_residual, **kwargs)


def create_SEWRepVGG_A0(block, num_classes=1000, deploy=False, cnf=None, spiking_neuron:callable=None, reuse_neurons=False, zero_init_residual=True, **kwargs):
    return SEWRepVGG(get_block_by_name(block), [2, 4, 14, 1], num_classes=num_classes, width_multipliers=[0.75, 0.75, 0.75, 2.5], override_groups_map=None, deploy=deploy, cnf=cnf, spiking_neuron=spiking_neuron, reuse_neurons=reuse_neurons, zero_init_residual=zero_init_residual,**kwargs)

def create_SEWRepVGG_A1(block, num_classes=1000, deploy=False, cnf=None, spiking_neuron:callable=None, reuse_neurons=False, zero_init_residual=True, **kwargs):    
    return SEWRepVGG(get_block_by_name(block), [2, 4, 14, 1], num_classes=num_classes, width_multipliers=[1, 1, 1, 2.5], override_groups_map=None, deploy=deploy, cnf=cnf, spiking_neuron=spiking_neuron, reuse_neurons=reuse_neurons, zero_init_residual=zero_init_residual,**kwargs)

def create_SEWRepVGG_A2(block, num_classes=1000, deploy=False, cnf=None, spiking_neuron:callable=None, reuse_neurons=False, zero_init_residual=True, **kwargs):
    return SEWRepVGG(get_block_by_name(block), [2, 4, 14, 1], num_classes=num_classes, width_multipliers=[1.5, 1.5, 1.5, 2.5], override_groups_map=None, deploy=deploy, cnf=cnf, spiking_neuron=spiking_neuron, reuse_neurons=reuse_neurons, zero_init_residual=zero_init_residual,**kwargs)

def create_SEWRepVGG_B0(block, num_classes=1000, deploy=False, cnf=None, spiking_neuron:callable=None, reuse_neurons=False, zero_init_residual=True, **kwargs):
    return SEWRepVGG(get_block_by_name(block), [4, 6, 16, 1], num_classes=num_classes, width_multipliers=[1, 1, 1, 2.5], override_groups_map=None, deploy=deploy, cnf=cnf, spiking_neuron=spiking_neuron, reuse_neurons=reuse_neurons, zero_init_residual=zero_init_residual,**kwargs)

def create_SEWRepVGG_B1(block, num_classes=1000, deploy=False, cnf=None, spiking_neuron:callable=None, reuse_neurons=False, zero_init_residual=True, **kwargs):
    return SEWRepVGG(get_block_by_name(block), [4, 6, 16, 1], num_classes=num_classes, width_multipliers=[1.5, 1.5, 1.5, 2.5], override_groups_map=None, deploy=deploy, cnf=cnf, spiking_neuron=spiking_neuron, reuse_neurons=reuse_neurons, zero_init_residual=zero_init_residual,**kwargs)

def create_SEWRepVGG_B1g2(block, num_classes=1000, deploy=False, cnf=None, spiking_neuron:callable=None, reuse_neurons=False, zero_init_residual=True, **kwargs):
    return SEWRepVGG(get_block_by_name(block), [4, 6, 16, 1], num_classes=num_classes, width_multipliers=[1.5, 1.5, 1.5, 2.5], override_groups_map=g2_map, deploy=deploy, cnf=cnf, spiking_neuron=spiking_neuron, reuse_neurons=reuse_neurons, zero_init_residual=zero_init_residual,**kwargs)

def create_SEWRepVGG_B1g4(block, num_classes=1000, deploy=False, cnf=None, spiking_neuron:callable=None, reuse_neurons=False, zero_init_residual=True, **kwargs):
    return SEWRepVGG(get_block_by_name(block), [4, 6, 16, 1], num_classes=num_classes, width_multipliers=[1.5, 1.5, 1.5, 2.5], override_groups_map=g4_map, deploy=deploy, cnf=cnf, spiking_neuron=spiking_neuron, reuse_neurons=reuse_neurons, zero_init_residual=zero_init_residual,**kwargs)

def create_SEWRepVGG_B2(block, num_classes=1000, deploy=False, cnf=None, spiking_neuron:callable=None, reuse_neurons=False, zero_init_residual=True, **kwargs):
    return SEWRepVGG(get_block_by_name(block), [4, 6, 16, 1], num_classes=num_classes, width_multipliers=[2, 2, 2, 2.5], override_groups_map=None, deploy=deploy, cnf=cnf, spiking_neuron=spiking_neuron, reuse_neurons=reuse_neurons, zero_init_residual=zero_init_residual,**kwargs)

def create_SEWRepVGG_B2g2(block, num_classes=1000, deploy=False, cnf=None, spiking_neuron:callable=None, reuse_neurons=False, zero_init_residual=True,**kwargs):
    return SEWRepVGG(get_block_by_name(block), [4, 6, 16, 1], num_classes=num_classes, width_multipliers=[2, 2, 2, 2.5], override_groups_map=g2_map, deploy=deploy, cnf=cnf, spiking_neuron=spiking_neuron, reuse_neurons=reuse_neurons, zero_init_residual=zero_init_residual, **kwargs)

def create_SEWRepVGG_B2g4(block, num_classes=1000, deploy=False, cnf=None, spiking_neuron:callable=None, reuse_neurons=False, zero_init_residual=True,**kwargs):
    return SEWRepVGG(get_block_by_name(block), [4, 6, 16, 1], num_classes=num_classes, width_multipliers=[2, 2, 2, 2.5], override_groups_map=g4_map, deploy=deploy, cnf=cnf, spiking_neuron=spiking_neuron, reuse_neurons=reuse_neurons, zero_init_residual=zero_init_residual, **kwargs)

def create_SEWRepVGG_B3(block, num_classes=1000, deploy=False, cnf=None, spiking_neuron:callable=None, reuse_neurons=False, zero_init_residual=True,**kwargs):
    return SEWRepVGG(get_block_by_name(block), [4, 6, 16, 1], num_classes=num_classes, width_multipliers=[3, 3, 3, 2.75], override_groups_map=None, deploy=deploy, cnf=cnf, spiking_neuron=spiking_neuron, reuse_neurons=reuse_neurons, zero_init_residual=zero_init_residual, **kwargs)

def create_SEWRepVGG_B3g2(block, num_classes=1000, deploy=False, cnf=None, spiking_neuron:callable=None, reuse_neurons=False, zero_init_residual=True,**kwargs):
    return SEWRepVGG(get_block_by_name(block), [4, 6, 16, 1], num_classes=num_classes, width_multipliers=[3, 3, 3, 2.75], override_groups_map=g2_map, deploy=deploy, cnf=cnf, spiking_neuron=spiking_neuron, reuse_neurons=reuse_neurons, zero_init_residual=zero_init_residual, **kwargs)

def create_SEWRepVGG_B3g4(block, num_classes=1000, deploy=False, cnf=None, spiking_neuron:callable=None, reuse_neurons=False, zero_init_residual=True,**kwargs):
    return SEWRepVGG(get_block_by_name(block), [4, 6, 16, 1], num_classes=num_classes, width_multipliers=[3, 3, 3, 2.75], override_groups_map=g4_map, deploy=deploy, cnf=cnf, spiking_neuron=spiking_neuron, reuse_neurons=reuse_neurons, zero_init_residual=zero_init_residual, **kwargs)

func_dict = {
    'SpikingRepVGG-A0': create_SpikingRepVGG_A0,
    'SpikingRepVGG-A1': create_SpikingRepVGG_A1,
    'SpikingRepVGG-A2': create_SpikingRepVGG_A2,
    'SpikingRepVGG-B0': create_SpikingRepVGG_B0,
    'SpikingRepVGG-B1': create_SpikingRepVGG_B1,
    'SpikingRepVGG-B1g2': create_SpikingRepVGG_B1g2,
    'SpikingRepVGG-B1g4': create_SpikingRepVGG_B1g4,
    'SpikingRepVGG-B2': create_SpikingRepVGG_B2,
    'SpikingRepVGG-B2g2': create_SpikingRepVGG_B2g2,
    'SpikingRepVGG-B2g4': create_SpikingRepVGG_B2g4,
    'SpikingRepVGG-B3': create_SpikingRepVGG_B3,
    'SpikingRepVGG-B3g2': create_SpikingRepVGG_B3g2,
    'SpikingRepVGG-B3g4': create_SpikingRepVGG_B3g4,
    'SEWRepVGG-A0': create_SEWRepVGG_A0,
    'SEWRepVGG-A1': create_SEWRepVGG_A1,
    'SEWRepVGG-A2': create_SEWRepVGG_A2,
    'SEWRepVGG-B0': create_SEWRepVGG_B0,
    'SEWRepVGG-B1': create_SEWRepVGG_B1,
    'SEWRepVGG-B1g2': create_SEWRepVGG_B1g2,
    'SEWRepVGG-B1g4': create_SEWRepVGG_B1g4,
    'SEWRepVGG-B2': create_SEWRepVGG_B2,
    'SEWRepVGG-B2g2': create_SEWRepVGG_B2g2,
    'SEWRepVGG-B2g4': create_SEWRepVGG_B2g4,
    'SEWRepVGG-B3': create_SEWRepVGG_B3,
    'SEWRepVGG-B3g2': create_SEWRepVGG_B3g2,
    'SEWRepVGG-B3g4': create_SEWRepVGG_B3g4,
}

def get_model_func_by_name(name):
    return func_dict[name]
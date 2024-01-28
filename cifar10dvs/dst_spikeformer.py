import torch
import torch.nn as nn
import torch.nn.functional as F
from spikingjelly.activation_based import layer, neuron, surrogate
from connecting_neuron import ConnLIFNode
from collections import OrderedDict

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

class Rep3x3(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, groups=1, deploy=False):
        super(Rep3x3, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.stride = stride
        self.groups = groups
        self.identity = (in_channels == out_channels and stride == 1)
        self.deploy = deploy

        if deploy:
            self.reparm = layer.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, groups=groups, padding=1, bias=True)
            self.sn = neuron.LIFNode(tau=2.0, detach_reset=True, surrogate_function=surrogate.ATan())
        else:
            self.conv3x3 = layer.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, groups=groups, padding=1, bias=False)
            self.bn3x3 = layer.BatchNorm2d(out_channels)
            self.conv1x1 = layer.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, groups=groups, bias=False)
            self.bn = layer.BatchNorm2d(out_channels)
            if self.identity:
                self.aac = nn.Identity()
                self.sn = ConnLIFNode(tau=2.0, detach_reset=True, surrogate_function=surrogate.ATan())
            else:
                self.aac = convrelupxp(in_channels, out_channels, stride=stride)
                self.sn = neuron.LIFNode(tau=2.0, detach_reset=True, surrogate_function=surrogate.ATan())
                

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
        self.sn = neuron.LIFNode(tau=2.0, detach_reset=True, surrogate_function=surrogate.ATan(), step_mode=self.conv3x3.step_mode).to(self.conv3x3.weight.device)
        #for para in self.parameters(): #commented out for syops param count
        #    para.detach_()
        self.__delattr__('conv3x3')
        self.__delattr__('conv1x1')
        self.__delattr__('bn3x3')
        self.__delattr__('bn')
        self.__delattr__('aac')
        self.deploy=True

class Rep1x1(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, groups=1, deploy=False):
        super(Rep1x1, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.stride = stride
        self.groups = groups
        self.identity = (in_channels == out_channels and stride == 1)
        self.deploy = deploy

        if deploy:
            self.reparm = layer.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, groups=groups, bias=True)
            self.sn = neuron.LIFNode(tau=2.0, detach_reset=True, surrogate_function=surrogate.ATan())
        else:
            self.conv1x1 = layer.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, groups=groups, bias=False)
            self.bn = layer.BatchNorm2d(out_channels)
            if self.identity:
                self.aac = nn.Identity()
                self.sn = ConnLIFNode(tau=2.0, detach_reset=True, surrogate_function=surrogate.ATan())
            else:
                self.aac = convrelupxp(in_channels, out_channels, stride=stride)
                self.sn = neuron.LIFNode(tau=2.0, detach_reset=True, surrogate_function=surrogate.ATan())
                

    def forward(self, x):
        if self.deploy:
            return self.sn(self.reparam(x))
        else:
            x, y = x
            out = self.bn(self.conv1x1(x))
            if y is not None:
                y = self.aac(y) + out
            elif self.training:
                y = out
            if self.identity:
                return self.sn(out, x), y
            else:
                return self.sn(out), y
            
    def get_equivalent_kernel_bias(self):
        kernel, bias = self._fuse_bn_tensor(self.conv1x1, self.bn)
        if self.identity:
            identity_value = self.sn.y_multiplier
            input_dim = self.in_channels // self.groups
            id_tensor = torch.zeros((self.in_channels, input_dim, 1, 1), dtype=kernel.dtype, device=kernel.device)
            for i in range(self.in_channels):
                id_tensor[i, i % input_dim, 0, 0] = identity_value
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
    
    def switch_to_deploy(self):
        if hasattr(self, 'reparam'):
            return
        kernel, bias = self.get_equivalent_kernel_bias()
        self.reparam = layer.Conv2d(in_channels=self.in_channels, out_channels=self.out_channels,
                                    kernel_size=1, stride=self.stride,
                                    groups=self.groups, bias=True, step_mode=self.conv1x1.step_mode).to(self.conv1x1.weight.device)
        self.reparam.weight.data = kernel
        self.reparam.bias.data = bias
        self.sn = neuron.LIFNode(tau=2.0, detach_reset=True, surrogate_function=surrogate.ATan(), step_mode=self.conv1x1.step_mode).to(self.conv1x1.weight.device)
        #for para in self.parameters(): #commented out for syops param count
        #    para.detach_()
        self.__delattr__('conv1x1')
        self.__delattr__('bn')
        self.__delattr__('aac')
        self.deploy=True

class RepMLP(nn.Module):
    def __init__(self, in_channels, out_channels=None, num_blocks=2, kernel=1, deploy=False):
        super(RepMLP, self).__init__()
        out_channels = out_channels or in_channels
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel = kernel
        self.deploy = deploy
        block = Rep1x1 if kernel == 1 else Rep3x3
        blocks = nn.Sequential()
        blocks.append(block(in_channels, out_channels, deploy=deploy))
        for _ in range(num_blocks - 1):
            blocks.append(block(out_channels, out_channels, deploy=deploy))
        self.blocks = blocks

    def forward(self, x):
        return self.blocks(x)
    
    def switch_to_deploy(self):
        for block in self.blocks:
            block.switch_to_deploy()
        self.deploy=True

class RepSSA(nn.Module):
    def __init__(self, dim, num_heads=8, deploy=False):
        super(RepSSA, self).__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.scale = 0.125
        self.deploy = deploy

        if self.deploy:
            self.q_reparam = layer.Conv1d(dim, dim, kernel_size=1, stride=1, bias=True)
            self.k_reparam = layer.Conv1d(dim, dim, kernel_size=1, stride=1, bias=True)
            self.v_reparam = layer.Conv1d(dim, dim, kernel_size=1, stride=1, bias=True)
            self.proj_reparam = layer.Conv1d(dim, dim, kernel_size=1, stride=1, bias=True)
        else:
            self.q_conv = layer.Conv1d(dim, dim, kernel_size=1, stride=1, bias=False)
            self.q_bn = layer.BatchNorm1d(dim)
            self.k_conv = layer.Conv1d(dim, dim, kernel_size=1, stride=1, bias=False)
            self.k_bn = layer.BatchNorm1d(dim)
            self.v_conv = layer.Conv1d(dim, dim, kernel_size=1, stride=1, bias=False)
            self.v_bn = layer.BatchNorm1d(dim)
            self.proj_conv = layer.Conv1d(dim, dim, kernel_size=1, stride=1, bias=False)
            self.proj_bn = layer.BatchNorm1d(dim)

        self.q_sn = neuron.LIFNode(tau=2.0, detach_reset=True, surrogate_function=surrogate.ATan())
        self.k_sn = neuron.LIFNode(tau=2.0, detach_reset=True, surrogate_function=surrogate.ATan())
        self.v_sn = neuron.LIFNode(tau=2.0, detach_reset=True, surrogate_function=surrogate.ATan())
        self.attn_sn = neuron.LIFNode(tau=2.0, detach_reset=True, surrogate_function=surrogate.ATan())
        self.proj_sn = ConnLIFNode(tau=2.0, detach_reset=True, surrogate_function=surrogate.ATan())

    def forward(self, x):
        if self.deploy:
            T, B, C, H, W = x.shape
            qkv_x = x.flatten(3)
            N = H * W

            q = self.q_sn(self.q_reparam(qkv_x))
            q = q.transpose(-1,-2).reshape(T,B, N, self.num_heads, C//self.num_heads).permute(0,1,3,2,4).contiguous()

            k = self.k_sn(self.k_reparam(qkv_x))
            k = k.transpose(-1,-2).reshape(T,B, N, self.num_heads, C//self.num_heads).permute(0,1,3,2,4).contiguous()

            v = self.v_sn(self.v_reparam(qkv_x))
            v = v.transpose(-1,-2).reshape(T,B, N, self.num_heads, C//self.num_heads).permute(0,1,3,2,4).contiguous()

            out = k.transpose(-2,-1) @ v
            out = (q @ out) * self.scale

            out = out.transpose(3,4).reshape(T,B,C,N).contiguous()
            out = self.attn_sn(out)
            out = self.proj_sn(self.proj_reparam(out).reshape(T,B,C,H,W),x)
            return out
        else:
            x, y = x
            T, B, C, H, W = x.shape
            qkv_x = x.flatten(3)
            N = H * W

            q = self.q_sn(self.q_bn(self.q_conv(qkv_x)))
            q = q.transpose(-1,-2).reshape(T,B, N, self.num_heads, C//self.num_heads).permute(0,1,3,2,4).contiguous()

            k = self.k_sn(self.k_bn(self.k_conv(qkv_x)))
            k = k.transpose(-1,-2).reshape(T,B, N, self.num_heads, C//self.num_heads).permute(0,1,3,2,4).contiguous()

            v = self.v_sn(self.v_bn(self.v_conv(qkv_x)))
            v = v.transpose(-1,-2).reshape(T,B, N, self.num_heads, C//self.num_heads).permute(0,1,3,2,4).contiguous()

            out = k.transpose(-2,-1) @ v
            out = (q @ out) * self.scale

            out = out.transpose(3,4).reshape(T,B,C,N).contiguous()
            out = self.attn_sn(out)
            out = self.proj_bn(self.proj_conv(out)).reshape(T,B,C,H,W)
            if y is not None:
                y = y + out
            elif self.training:
                y = out
            out = self.proj_sn(out,x)
            return out, y
    
    def _fuse_bn_tensor(self, conv, bn):
        kernel = conv.weight
        running_mean = bn.running_mean
        running_var = bn.running_var
        gamma = bn.weight
        beta = bn.bias
        eps = bn.eps
        std = torch.sqrt(running_var + eps)
        t = (gamma / std).reshape(-1, 1, 1)
        return kernel * t, beta - running_mean * gamma / std
    
    def switch_to_deploy(self):
        if hasattr(self, 'proj_reparam'):
            return
        
        q_kernel, q_bias = self._fuse_bn_tensor(self.q_conv, self.q_bn)
        self.q_reparam = layer.Conv1d(in_channels=self.dim, out_channels=self.dim,
                                  kernel_size=1, stride=1,
                                  bias=True, step_mode=self.q_conv.step_mode).to(self.q_conv.weight.device)
        self.q_reparam.weight.data = q_kernel
        self.q_reparam.bias.data = q_bias

        k_kernel, k_bias = self._fuse_bn_tensor(self.k_conv, self.k_bn)
        self.k_reparam = layer.Conv1d(in_channels=self.dim, out_channels=self.dim,
                                    kernel_size=1, stride=1,
                                    bias=True, step_mode=self.k_conv.step_mode).to(self.k_conv.weight.device)
        self.k_reparam.weight.data = k_kernel
        self.k_reparam.bias.data = k_bias

        v_kernel, v_bias = self._fuse_bn_tensor(self.v_conv, self.v_bn)
        self.v_reparam = layer.Conv1d(in_channels=self.dim, out_channels=self.dim,
                                    kernel_size=1, stride=1,
                                    bias=True, step_mode=self.v_conv.step_mode).to(self.v_conv.weight.device)
        self.v_reparam.weight.data = v_kernel
        self.v_reparam.bias.data = v_bias

        proj_kernel, proj_bias = self._fuse_bn_tensor(self.proj_conv, self.proj_bn)
        self.proj_reparam = layer.Conv1d(in_channels=self.dim, out_channels=self.dim,
                                        kernel_size=1, stride=1,
                                        bias=True, step_mode=self.proj_conv.step_mode).to(self.proj_conv.weight.device)
        self.proj_reparam.weight.data = proj_kernel
        self.proj_reparam.bias.data = proj_bias

        self.q_sn = neuron.LIFNode(tau=2.0, detach_reset=True, surrogate_function=surrogate.ATan(), step_mode=self.q_conv.step_mode).to(self.q_conv.weight.device)
        self.k_sn = neuron.LIFNode(tau=2.0, detach_reset=True, surrogate_function=surrogate.ATan(), step_mode=self.k_conv.step_mode).to(self.k_conv.weight.device)
        self.v_sn = neuron.LIFNode(tau=2.0, detach_reset=True, surrogate_function=surrogate.ATan(), step_mode=self.v_conv.step_mode).to(self.v_conv.weight.device)
        self.attn_sn = neuron.LIFNode(tau=2.0, detach_reset=True, surrogate_function=surrogate.ATan(), step_mode=self.proj_conv.step_mode).to(self.proj_conv.weight.device)
        self.proj_sn = ConnLIFNode(tau=2.0, detach_reset=True, surrogate_function=surrogate.ATan(), step_mode=self.proj_conv.step_mode).to(self.proj_conv.weight.device)
        #for para in self.parameters(): #commented out for syops param count
        #    para.detach_()
        self.__delattr__('q_conv')
        self.__delattr__('q_bn')
        self.__delattr__('k_conv')
        self.__delattr__('k_bn')
        self.__delattr__('v_conv')
        self.__delattr__('v_bn')
        self.__delattr__('proj_conv')
        self.__delattr__('proj_bn')
        self.deploy=True

class RepEncoderBlock(nn.Module):
    def __init__(self, dim, num_heads, mlp_depth, mlp_kernel=1, deploy=False):
        super(RepEncoderBlock, self).__init__()
        self.deploy = deploy

        self.attn = RepSSA(dim, num_heads, deploy)
        self.mlp = RepMLP(dim, num_blocks=mlp_depth, deploy=deploy, kernel=mlp_kernel)

    def forward(self, x):
        return self.mlp(self.attn(x))
    
    def switch_to_deploy(self):
        self.attn.switch_to_deploy()
        self.mlp.switch_to_deploy()
        self.deploy=True
        
class RepSCSBlock(nn.Module):
    def __init__(self, in_channels, out_channels, mlp_depth, mlp_kernel, deploy=False):
        super(RepSCSBlock, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.mlp_depth = mlp_depth
        self.mlp_kernel = mlp_kernel
        self.deploy = deploy

        if self.deploy:
            self.proj_reparm = layer.Conv2d(in_channels, out_channels, kernel_size=2, stride=2, bias=True)
        else:
            self.proj_conv = layer.Conv2d(in_channels, out_channels, kernel_size=2, stride=2, bias=False)
            self.proj_bn = layer.BatchNorm2d(out_channels)
            self.aac = convrelupxp(in_channels, out_channels, stride=2)
        self.proj_sn = neuron.LIFNode(tau=2.0, detach_reset=True, surrogate_function=surrogate.ATan())

        self.mlp = RepMLP(out_channels, out_channels, num_blocks=mlp_depth, kernel=mlp_kernel, deploy=deploy)

    def forward(self, x):
        if self.deploy:
            return self.mlp(self.proj_sn(self.proj_reparm(x)))
        else:
            x,y = x
            out = self.proj_bn(self.proj_conv(x))
            if y is not None:
                y = self.aac(y) + out
            elif self.training:
                y = out
            out = self.proj_sn(out)
            out = self.mlp((out,y))
            return out
        
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
        if hasattr(self, 'proj_reparm'):
            return
        proj_kernel, proj_bias = self._fuse_bn_tensor(self.proj_conv, self.proj_bn)
        self.proj_reparm = layer.Conv2d(in_channels=self.in_channels, out_channels=self.out_channels,
                                      kernel_size=2, stride=2,
                                      bias=True, step_mode=self.proj_conv.step_mode).to(self.proj_conv.weight.device)
        self.proj_reparm.weight.data = proj_kernel
        self.proj_reparm.bias.data = proj_bias
        self.proj_sn = neuron.LIFNode(tau=2.0, detach_reset=True, surrogate_function=surrogate.ATan(), step_mode=self.proj_conv.step_mode).to(self.proj_conv.weight.device)
        self.mlp.switch_to_deploy()
        #for para in self.parameters(): #commented out for syops param count
        #    para.detach_()
        self.__delattr__('proj_conv')
        self.__delattr__('proj_bn')
        self.__delattr__('aac')
        self.deploy=True

class RepSCS(nn.Module):
    def __init__(self, img_size=(128,128), patch_size=(4,4), in_channels=2, embed_dims=256, depths=[2,2,2,2], deploy=False):
        super(RepSCS, self).__init__()
        self.img_size = img_size
        self.patch_size = patch_size if isinstance(patch_size, tuple) else (patch_size, patch_size)
        self.C = in_channels
        self.H, self.W = self.img_size[0]//self.patch_size[0], self.img_size[1]//self.patch_size[1]
        self.num_patches = self.H * self.W
        self.embed_dims = embed_dims
        self.deploy = deploy

        blocks = nn.Sequential()
        blocks.append(RepSCSBlock(in_channels,embed_dims//8,mlp_kernel=3,mlp_depth=depths[0],deploy=self.deploy))
        blocks.append(RepSCSBlock(embed_dims//8,embed_dims//4,mlp_kernel=3,mlp_depth=depths[1],deploy=self.deploy))
        blocks.append(RepSCSBlock(embed_dims//4,embed_dims//2,mlp_kernel=3,mlp_depth=depths[2],deploy=self.deploy))
        blocks.append(RepSCSBlock(embed_dims//2,embed_dims,mlp_kernel=3,mlp_depth=depths[3],deploy=self.deploy))
        blocks.append(Rep3x3(embed_dims,embed_dims,stride=1,deploy=self.deploy))

        self.blocks = blocks
    
    def forward(self, x):
        return self.blocks(x)
    
    def switch_to_deploy(self):
        for block in self.blocks:
            block.switch_to_deploy()
        self.deploy=True

class RepSPSBlock(nn.Module):
    def __init__(self, in_channels, out_channels, deploy=False):
        super(RepSPSBlock, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.deploy = deploy

        if self.deploy:
            self.proj_reparm = layer.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, bias=True)
            self.mp = layer.MaxPool2d(kernel_size=2, stride=2)
        else:
            self.proj_conv = layer.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, bias=False)
            self.proj_bn = layer.BatchNorm2d(out_channels)
            self.mp = layer.MaxPool2d(kernel_size=2, stride=2)
        self.proj_sn = neuron.LIFNode(tau=2.0, detach_reset=True, surrogate_function=surrogate.ATan())

    def forward(self, x):
        if self.deploy:
            return self.mp(self.proj_sn(self.proj_reparm(x)))
        else:
            x,y = x
            out = self.proj_bn(self.proj_conv(x))
            if y is not None:
                y = self.mp(y + out)
            elif self.training:
                y = self.mp(out)
            out = self.proj_sn(out)
            out = self.mp((out))
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
        if hasattr(self, 'proj_reparm'):
            return
        proj_kernel, proj_bias = self._fuse_bn_tensor(self.proj_conv, self.proj_bn)
        self.proj_reparm = layer.Conv2d(in_channels=self.in_channels, out_channels=self.out_channels,
                                      kernel_size=3, stride=1,
                                      bias=True, step_mode=self.proj_conv.step_mode).to(self.proj_conv.weight.device)
        self.proj_reparm.weight.data = proj_kernel
        self.proj_reparm.bias.data = proj_bias
        self.proj_sn = neuron.LIFNode(tau=2.0, detach_reset=True, surrogate_function=surrogate.ATan(), step_mode=self.proj_conv.step_mode).to(self.proj_conv.weight.device)
        #for para in self.parameters(): #commented out for syops param count
        #    para.detach_()
        self.__delattr__('proj_conv')
        self.__delattr__('proj_bn')
        self.deploy=True

class RepSPS(nn.Module):
    def __init__(self, img_size=(128,128), patch_size=(4,4), in_channels=2, embed_dims=256, deploy=False):
        super(RepSPS, self).__init__()
        self.img_size = img_size
        self.patch_size = patch_size if isinstance(patch_size, tuple) else (patch_size, patch_size)
        self.C = in_channels
        self.H, self.W = self.img_size[0]//self.patch_size[0], self.img_size[1]//self.patch_size[1]
        self.num_patches = self.H * self.W
        self.embed_dims = embed_dims
        self.deploy = deploy

        blocks = nn.Sequential()
        blocks.append(RepSPSBlock(in_channels,embed_dims//8,deploy=self.deploy))
        blocks.append(RepSPSBlock(embed_dims//8,embed_dims//4,deploy=self.deploy))
        blocks.append(RepSPSBlock(embed_dims//4,embed_dims//2,deploy=self.deploy))
        blocks.append(RepSPSBlock(embed_dims//2,embed_dims,deploy=self.deploy))
        blocks.append(Rep3x3(embed_dims,embed_dims,stride=1,deploy=self.deploy))

        self.blocks = blocks
    
    def forward(self, x):
        return self.blocks(x)
    
    def switch_to_deploy(self):
        for block in self.blocks:
            block.switch_to_deploy()
        self.deploy=True

class RepSpikeFormer(nn.Module):
    def __init__(self, img_size=(128,128), patch_size=4, in_channels=2, num_classes=10,
                embed_dims=256, num_heads=8, scs_depths=[2,2,2,2], mlp_depths=[2,2], mlp_kernel=1, deploy=False, use_scs=True):
        super(RepSpikeFormer, self).__init__()
        self.num_classes = num_classes
        self.deploy = deploy
        self.use_scs = use_scs
        self.patch_embed = RepSCS(img_size, patch_size, in_channels, embed_dims, scs_depths, deploy) if use_scs else RepSPS(img_size, patch_size, in_channels, embed_dims, deploy)
        self.attn_blocks = nn.Sequential(*[
            RepEncoderBlock(embed_dims, num_heads, mlp_depth, mlp_kernel=mlp_kernel, deploy=deploy) for mlp_depth in mlp_depths
        ])
        self.fc = layer.Linear(embed_dims, num_classes, bias=True) if num_classes > 0 else nn.Identity()
        if not self.deploy:
            self.aac = layer.Linear(embed_dims, num_classes, bias=True) if num_classes > 0 else nn.Identity()
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, layer.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, layer.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, layer.BatchNorm2d):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)

    def forward(self, x):
        x = x.permute(1,0,2,3,4)
        if self.deploy:
            x = self.patch_embed(x)
            x = self.attn_blocks(x)
            x = x.flatten(3).mean(3).mean(0)
            x = self.fc(x)
            return x
        else:
            (x,y) = self.patch_embed((x,None))
            (x,y) = self.attn_blocks((x,y))
            x = x.flatten(3).mean(3).mean(0)
            x = self.fc(x)
            if y is not None:
                y = y.flatten(3).mean(3).mean(0)
                y = self.aac(y)
            return x, y
        
    def switch_to_deploy(self):
        self.patch_embed.switch_to_deploy()
        for block in self.attn_blocks:
            block.switch_to_deploy()
        self.deploy=True

def RepSpikeFormerA0(num_classes=10, deploy=False):
    return RepSpikeFormer(
        img_size=(128,128),
        patch_size=(16,16),
        in_channels=2,
        num_classes=num_classes, 
        embed_dims=256,
        num_heads=16,
        scs_depths=[1,1,1,1],
        mlp_depths=[2,2],
        mlp_kernel=1
        )

def RepSpikeFormerA0_SPS(num_classes=10, deploy=False):
    return RepSpikeFormer(
        img_size=(128,128),
        patch_size=(16,16),
        in_channels=2,
        num_classes=num_classes, 
        embed_dims=256,
        num_heads=16,
        mlp_depths=[2,2],
        mlp_kernel=1,
        use_scs=False
        )

#TODO: implement for dvsgesture for B series
def RepSpikeFormerB0(num_classes=10, deploy=False):
    return RepSpikeFormer(
        img_size=(128,128),
        patch_size=(16,16),
        in_channels=2,
        num_classes=num_classes, 
        embed_dims=256,
        num_heads=16,
        scs_depths=[1,1,1,1],
        mlp_depths=[2,2],
        mlp_kernel=1
    )

def RepSpikeFormerB0_SPS(num_classes=10, deploy=False):
    return RepSpikeFormer(
        img_size=(128,128),
        patch_size=(16,16),
        in_channels=2,
        num_classes=num_classes, 
        embed_dims=256,
        num_heads=16,
        mlp_depths=[2,2],
        mlp_kernel=1,
        use_scs=False
    )

model_dict = {
    'RepSpikeFormerA0': RepSpikeFormerA0,
    'RepSpikeFormerA0_SPS': RepSpikeFormerA0_SPS,
    'RepSpikeFormerB0': RepSpikeFormerB0,
    'RepSpikeFormerB0_SPS': RepSpikeFormerB0_SPS
}

def get_model_by_name(model_name):
    return model_dict[model_name]
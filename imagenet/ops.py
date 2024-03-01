import numpy as np
import torch.nn as nn

from spikingjelly.activation_based import neuron, layer
from batchnorm_neuron import BNIFNode

def spike_rate(input):
    unique = input.unique()
    if len(unique) <= 2 and input.max() <= 1 and input.min() >= 0: 
        spike = True
        spike_rate = (input.sum() / input.numel()).item()
    else: 
        spike = False
        spike_rate = 1

    return spike, spike_rate

def empty_syops_counter_hook(module, input, output):
    module.__syops__ += np.array([0.0, 0.0, 0.0, 0.0])

def relu_syops_counter_hook(module, input, output):
    active_element_count = output.numel()
    module.__syops__[0] += int(active_element_count)

    spike, rate = spike_rate(input)
    if spike:
        module.__syops__[1] += int(active_element_count) * rate
    else:
        module.__syops__[2] += int(active_element_count)

    module.__syops__[3] += rate * 100

def if_syops_counter_hook(module, input, output):
    if isinstance(input, tuple):
        input = input[0]
    if isinstance(output, tuple):
        output = output[0]
    spike, rate = spike_rate(output)
    active_element_count = input.numel()
    module.__syops__[0] += int(active_element_count)
    module.__syops__[1] += int(active_element_count)
    module.__syops__[3] += rate * 100

def lif_syops_counter_hook(module, input, output):
    if isinstance(input, tuple):
        input = input[0]
    if isinstance(output, tuple):
        output = output[0]
    spike, rate = spike_rate(output)
    active_element_count = input.numel()
    module.__syops__[0] += int(active_element_count)
    module.__syops__[1] += int(active_element_count)
    module.__syops__[3] += rate * 100

def linear_sysops_counter_hook(module, input, output):
    if isinstance(input, tuple):
        input = input[0]
    if isinstance(output, tuple):
        output = output[0]
    spike, rate = spike_rate(input)
    output_last_dim = output.shape[-1]
    bias_syops = output_last_dim if module.bias is not None else 0
    module.__syops__[0] += int(input.numel() * output_last_dim + bias_syops)
    if spike:
        module.__syops__[1] += int(input.numel() * output_last_dim + bias_syops) * rate
    else:
        module.__syops__[2] += int(input.numel() * output_last_dim + bias_syops)

    module.__syops__[3] += rate * 100

def pool_syops_counter_hook(module, input, output):
    if isinstance(input, tuple):
        input = input[0]
    if isinstance(output, tuple):
        output = output[0]
    spike, rate = spike_rate(input)
    module.__syops__[0]+= int(input.numel())
    if spike:
        module.__syops__[1] += int(input.numel()) * rate
    else:
        module.__syops__[2] += int(input.numel())
    module.__syops__[3] += rate * 100

def bn_syops_counter_hook(module, input, output):
    if isinstance(input, tuple):
        input = input[0]
    if isinstance(output, tuple):
        output = output[0]
    spike, rate = spike_rate(input)
    batch_syops = input.numel()
    if module.affine:
        batch_syops *= 2
    module.__syops__[0] += int(batch_syops)

    if spike:
        module.__syops__[1] += int(batch_syops) * rate
    else:
        module.__syops__[2] += int(batch_syops)

    module.__syops__[3] += rate * 100

def conv_syops_counter_hook(module, input, output):
    if isinstance(input, tuple):
        input = input[0]
    if isinstance(output, tuple):
        output = output[0]
    spike, rate = spike_rate(input)
    
    if not hasattr(module,'step_mode') or module.step_mode == 's': #check for if torch.nn conv or if step mode is set to 's'
        batch_size = input.shape[0]
    elif module.step_mode == 'm':
        batch_size = input.shape[0] * input.shape[1] #T * batch_size
    
    output_dims = list(output.shape[-2:])
    kernel_dims = list(module.kernel_size)
    in_channels = module.in_channels
    out_channels = module.out_channels
    groups = module.groups

    filters_per_channel = out_channels // groups
    conv_per_position_syops = np.prod(kernel_dims) * in_channels * filters_per_channel

    active_elements_count = batch_size * np.prod(output_dims)

    overall_conv_syops = conv_per_position_syops * active_elements_count

    bias_sysops = 0

    if module.bias is not None:
        bias_sysops = out_channels * active_elements_count
    
    overall_sysops = overall_conv_syops + bias_sysops
    
    module.__syops__[0] += int(overall_sysops)
    if spike:
        module.__syops__[1] += int(overall_sysops) * rate
    else:
        module.__syops__[2] += int(overall_sysops)
    module.__syops__[3] += rate * 100

MODULES_MAPPING = {
    #torch.nn convs
    nn.Conv1d: conv_syops_counter_hook,
    nn.Conv2d: conv_syops_counter_hook,
    nn.Conv3d: conv_syops_counter_hook,

    #spikingjelly convs
    layer.Conv1d: conv_syops_counter_hook,
    layer.Conv2d: conv_syops_counter_hook,
    layer.Conv3d: conv_syops_counter_hook,

    #torch.nn deconvolutions
    nn.ConvTranspose1d: conv_syops_counter_hook,
    nn.ConvTranspose2d: conv_syops_counter_hook,
    nn.ConvTranspose3d: conv_syops_counter_hook,

    #spikingjelly deconvolutions
    layer.ConvTranspose1d: conv_syops_counter_hook,
    layer.ConvTranspose2d: conv_syops_counter_hook,
    layer.ConvTranspose3d: conv_syops_counter_hook,

    #activations
    nn.ReLU: relu_syops_counter_hook,
    nn.PReLU: relu_syops_counter_hook,
    nn.ELU: relu_syops_counter_hook,
    nn.LeakyReLU: relu_syops_counter_hook,
    nn.ReLU6: relu_syops_counter_hook,

    #torch.nn poolings
    nn.MaxPool1d: pool_syops_counter_hook,
    nn.MaxPool2d: pool_syops_counter_hook,
    nn.MaxPool3d: pool_syops_counter_hook,
    nn.AvgPool1d: pool_syops_counter_hook,
    nn.AvgPool2d: pool_syops_counter_hook,
    nn.AvgPool3d: pool_syops_counter_hook,
    nn.AdaptiveMaxPool1d: pool_syops_counter_hook,
    nn.AdaptiveMaxPool2d: pool_syops_counter_hook,
    nn.AdaptiveMaxPool3d: pool_syops_counter_hook,
    nn.AdaptiveAvgPool1d: pool_syops_counter_hook,
    nn.AdaptiveAvgPool2d: pool_syops_counter_hook,
    nn.AdaptiveAvgPool3d: pool_syops_counter_hook,

    #spikingjelly poolings
    layer.MaxPool1d: pool_syops_counter_hook,
    layer.MaxPool2d: pool_syops_counter_hook,
    layer.MaxPool3d: pool_syops_counter_hook,
    layer.AvgPool1d: pool_syops_counter_hook,
    layer.AvgPool2d: pool_syops_counter_hook,
    layer.AvgPool3d: pool_syops_counter_hook,
    layer.AdaptiveAvgPool1d: pool_syops_counter_hook,
    layer.AdaptiveAvgPool2d: pool_syops_counter_hook,
    layer.AdaptiveAvgPool3d: pool_syops_counter_hook,

    #torch.nn batchnorm
    nn.BatchNorm1d: bn_syops_counter_hook,
    nn.BatchNorm2d: bn_syops_counter_hook,
    nn.BatchNorm3d: bn_syops_counter_hook,

    #spikingjelly batchnorm
    layer.BatchNorm1d: bn_syops_counter_hook,
    layer.BatchNorm2d: bn_syops_counter_hook,
    layer.BatchNorm3d: bn_syops_counter_hook,

    #torch.nn instance norm
    nn.InstanceNorm1d: bn_syops_counter_hook,
    nn.InstanceNorm2d: bn_syops_counter_hook,
    nn.InstanceNorm3d: bn_syops_counter_hook,

    #torch.nn group norm
    nn.GroupNorm: bn_syops_counter_hook,

    #spikingjelly group norm
    layer.GroupNorm: bn_syops_counter_hook,

    #torch.nn linear
    nn.Linear: linear_sysops_counter_hook,

    #spikingjelly linear
    layer.Linear: linear_sysops_counter_hook,

    #neurons
    neuron.IFNode: if_syops_counter_hook,
    neuron.LIFNode: lif_syops_counter_hook,
    neuron.ParametricLIFNode: lif_syops_counter_hook,
    BNIFNode: lif_syops_counter_hook,

}

if hasattr(nn, 'GELU'):
    MODULES_MAPPING[nn.GELU] = relu_syops_counter_hook
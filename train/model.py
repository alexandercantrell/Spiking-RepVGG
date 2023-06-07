import warnings

import numpy as np

import torch
import torch.nn as nn

from spikingjelly.activation_based import surrogate, neuron, functional

from train.blurpool import BlurPoolConv2d
from models.surrogate import FastATan
from models.spiking_repvgg import get_SpikingRepVGG_func_by_name
from models.hybrid_spiking_repvgg import get_HybridSpikingRepVGG_func_by_name
from models.static_spiking_repvgg import get_StaticSpikingRepVGG_func_by_name

def build_model(args):
    
    #set surrogate method
    surrogate_function = surrogate.ATan(alpha=args.surrogate_alpha)
    if args.fast_surrogate:
        warnings.warn(f"Using surrogate function FastATan which is experimental.")
        surrogate_function = FastATan(alpha=args.surrogate_alpha/2.0)
    
    #create model
    if 'StaticSpikingRepVGG' in args.arch:
        model = get_StaticSpikingRepVGG_func_by_name(args.arch)(num_classes=args.num_classes,deploy=False,use_checkpoint=args.use_checkpoint,
                        cnf=args.cnf,spiking_neuron=neuron.IFNode,surrogate_function=surrogate_function,detach_reset=True)
    elif 'HybridSpikingRepVGG' in args.arch:
        model = get_HybridSpikingRepVGG_func_by_name(args.arch)(num_classes=args.num_classes,deploy=False,use_checkpoint=args.use_checkpoint,
                        cnf=args.cnf,spiking_neuron=neuron.IFNode,surrogate_function=surrogate_function,detach_reset=True)
    elif 'SpikingRepVGG' in args.arch:
        model = get_SpikingRepVGG_func_by_name(args.arch)(num_classes=args.num_classes,deploy=False,use_checkpoint=args.use_checkpoint,
                        cnf=args.cnf,spiking_neuron=neuron.IFNode,surrogate_function=surrogate_function,detach_reset=True)
    else:
        raise ValueError(f"Model architecture {args.arch} does not exist!")
    
    #set model step mode
    if args.T > 0:
        functional.set_step_mode(model,'m')
    else:
        functional.set_step_mode(model,'s')

    #set neuron backend
    if args.cupy:
        functional.set_backend(model,'cupy',neuron.IFNode)
        warnings.warn("Setting the neuron backend to 'cupy'. This commonly conflicts with automatic mixed precision (AMP) training if enabled.")
        if not args.disable_amp:
            raise ValueError("Neuron backend 'cupy' conflicts with automatic mixed precision (AMP) training. "
                        "Either remove the --cupy flag from your execution, or disable AMP with --disable-amp.")
    
    #blurpool #TODO: test
    def apply_blurpool(mod: nn.Module):
        for (name, child) in mod.named_children():
            if isinstance(child, nn.Conv2d) and (np.max(child.stride) > 1 and child.in_channels >= 16): 
                setattr(mod, name, BlurPoolConv2d(child))
            else: apply_blurpool(child)
    if args.use_blurpool: apply_blurpool(model)

    #set channels last #TODO: test
    if args.channels_last:
        model = model.to(memory_format=torch.channels_last)
    
    #send to device
    model = model.to(torch.device(args.device))

    return model
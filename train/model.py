import warnings

import torch

from fastargs.decorators import param
from fastargs import Param, Section

from spikingjelly.activation_based import surrogate, neuron, functional

from models.surrogate import FastATan
from models.spiking_repvgg import get_SpikingRepVGG_func_by_name
from models.hybrid_spiking_repvgg import get_HybridSpikingRepVGG_func_by_name
from models.static_spiking_repvgg import get_StaticSpikingRepVGG_func_by_name
from train.data import get_num_classes
from train.utils import get_device_name, is_dist_avail_and_initialized

Section('model', 'model details').params(
    arch=Param(str, 'model arch', required=True),
    fast_atan=Param(bool,'whether or not to use the FastATan surrogate',is_flag=True),
    atan_alpha=Param(float,'atan alpha',default=2.0),
    cnf = Param(str,'cnf',default='FAST_XOR'),
    use_cupy = Param(bool,'use cupy',is_flag=True),
    use_checkpoints=Param(bool,'use gradient checkpointing',is_flag=True),
    sync_bn=Param(bool,'',is_flag=True),
    resume=Param(str,'',default=None),
    zero_init=Param(bool,'',is_flag=True)
)

@param('model.arch')
@param('model.fast_atan')
@param('model.atan_alpha')
@param('model.cnf')
@param('data.T')
@param('model.use_cupy')
@param('model.use_checkpoints')
@param('model.sync_bn')
@param('model.zero_init') #TODO:implement
def build_model(arch,fast_atan,atan_alpha,cnf,T,use_cupy,use_checkpoints,sync_bn,zero_init):
    num_classes = get_num_classes()
    surrogate_function = surrogate.ATan(alpha=atan_alpha)
    if fast_atan:
        warnings.warn(f"Using surrogate function FastATan which is experimental.")
        surrogate_function = FastATan(alpha=atan_alpha/2.0)

    if 'StaticSpikingRepVGG' in arch:
        model = get_StaticSpikingRepVGG_func_by_name(arch)(num_classes=num_classes,deploy=False,use_checkpoint=use_checkpoints,
                        cnf=cnf,spiking_neuron=neuron.IFNode,surrogate_function=surrogate_function,detach_reset=True)
    elif 'HybridSpikingRepVGG' in arch:
        model = get_HybridSpikingRepVGG_func_by_name(arch)(num_classes=num_classes,deploy=False,use_checkpoint=use_checkpoints,
                        cnf=cnf,spiking_neuron=neuron.IFNode,surrogate_function=surrogate_function,detach_reset=True)
    elif 'SpikingRepVGG' in arch:
        model = get_SpikingRepVGG_func_by_name(arch)(num_classes=num_classes,deploy=False,use_checkpoint=use_checkpoints,
                        cnf=cnf,spiking_neuron=neuron.IFNode,surrogate_function=surrogate_function,detach_reset=True)
    else:
        raise ValueError(f"Model architecture {arch} does not exist!")
    if T>0:
        functional.set_step_mode(model,'m')
    else:
        functional.set_step_mode(model,'s')

    if use_cupy:
        functional.set_backend(model,'cupy',instance=neuron.IFNode)

    model = model.to(memory_format=torch.channels_last)
    model = model.to(torch.device(get_device_name()))

    model_without_ddp=model
    if is_dist_avail_and_initialized():
        if sync_bn: model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[torch.device(get_device_name())])
    return model, model_without_ddp
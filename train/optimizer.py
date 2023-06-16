from typing import List, Optional, Tuple

import torch
import torch.optim as optim

from fastargs.decorators import param
from fastargs import Param, Section
from fastargs.validation import And, OneOf

Section('optim', 'optim details').params(
    optimizer = Param(And(str,OneOf(['sgd','sgd_nesterov','rmsprop','adam','adamw'])),'',default='sgd'),
    bn_decay = Param(float,'',default=0.0),
    weight_decay = Param(float,'',default=4e-5),
)

Section('optim').enable_if(lambda cfg:cfg['optim.optimizer']in['sgd','sgd_nesterov','rmsprop']).params(
    momentum = Param(float,'',default=0.9)
)

@param('optim.optimizer')
@param('lr.lr')
@param('optim.bn_decay')
@param('optim.weight_decay')
@param('optim.momentum') 
def build_optimizer(model,opt,lr,bn_decay,weight_decay,momentum=None):
    # Only do weight decay on non-batchnorm parameters
    all_params = list(model.named_parameters())
    bn_params = [v for k, v in all_params if ('bn' in k)]
    other_params = [v for k, v in all_params if not ('bn' in k)]
    param_groups = [{
        'params': bn_params,
        'weight_decay': bn_decay
    }, {
        'params': other_params,
        'weight_decay': weight_decay
    }]
    opt = opt.lower()
    if opt.startswith("sgd"):
        optimizer = optim.SGD(param_groups,lr=lr,momentum=momentum,nesterov="nesterov" in opt)
    elif opt == "rmsprop":
        optimizer = optim.RMSprop(param_groups,lr = lr,momentum=momentum,eps=0.0316,alpha=0.9)
    elif opt == "adam":
        optimizer = optim.Adam(param_groups,lr=lr)
    elif opt == 'adamw':
        optimizer = optim.AdamW(param_groups,lr=lr)
    else:
        raise RuntimeError(f"Invalid optimizer {opt}. Only sgd, rmsprop, adam, and adamw are supported.")
    return optimizer
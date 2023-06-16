import torch.optim as optim

from fastargs.decorators import param
from fastargs import Param, Section
from fastargs.validation import And, OneOf


Section('lr','lr').params(
    lr=Param(float,'',default=0.1),
    scheduler=Param(And(str,OneOf(['step','cosa','exp'])),'',default='cosa'),
    warmup_epochs=Param(int,'',default=5)
)
Section('lr').enable_if(lambda cfg: cfg['lr.scheduler'] in ['step','exp']).params(
    gamma=Param(float,'',default=0.1)
)
Section('lr').enable_if(lambda cfg: cfg['lr.scheduler'] == 'step').params(
    step_size=Param(int,'',default=30)
)
Section('lr').enable_if(lambda cfg: cfg['lr.scheduler'] == 'cosa').params(
    eta_min=Param(float,'',default=0.0)
)
Section('lr').enable_if(lambda cfg: cfg['lr.warmup_epochs']>0).params(
    warmup_method=Param(And(str,OneOf(['linear','constant'])),'',default='linear'),
    warmup_decay=Param(float,'',default=0.01)
)

@param('lr.scheduler')
@param('train.epochs')
@param('lr.warmup_epochs')
@param('lr.step_size')
@param('lr.gamma')
@param('lr.eta_min')
@param('lr.warmup_method')
@param('lr.warmup_decay')
def build_scheduler(optimizer,scheduler,epochs,warmup_epochs,step_size=None,gamma=None,eta_min=None,warmup_method=None,warmup_decay=None):
    scheduler = scheduler.lower()
    if scheduler == "step":
        main_lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)
    elif scheduler == "cosa":
        main_lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=epochs - warmup_epochs, eta_min=eta_min
        )
    elif scheduler == "exp":
        main_lr_scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=gamma)
    else:
        raise RuntimeError(
            f"Invalid lr scheduler '{scheduler}'. Only StepLR, CosineAnnealingLR and ExponentialLR "
            "are supported."
        )
    if warmup_epochs > 0:
        warmup_method = warmup_method.lower()
        if warmup_method == "linear":
            warmup_lr_scheduler = optim.lr_scheduler.LinearLR(
                optimizer, start_factor=warmup_decay, total_iters=warmup_epochs
            )
        elif warmup_method == "constant":
            warmup_lr_scheduler = optim.lr_scheduler.ConstantLR(
                optimizer, factor=warmup_decay, total_iters=warmup_epochs
            )
        else:
            raise RuntimeError(
                f"Invalid warmup lr method '{warmup_method}'. Only linear and constant are supported."
            )
        lr_scheduler = optim.lr_scheduler.SequentialLR(
            optimizer, schedulers=[warmup_lr_scheduler, main_lr_scheduler], milestones=[warmup_epochs]
        )
    else:
        lr_scheduler = main_lr_scheduler

    return lr_scheduler
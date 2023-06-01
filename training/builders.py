import os
import time

import torch
import torch.optim as optim
import torch.utils.data as data
from torch.utils.data.dataloader import default_collate

import torchvision
from torchvision import datasets
from torchvision.transforms.functional import InterpolationMode

from spikingjelly.activation_based import surrogate, neuron, functional

from models.fast_surrogate import FastATan
from models.spiking_repvgg import get_SpikingRepVGG_func_by_name
from models.hybrid_spiking_repvgg import get_HybridSpikingRepVGG_func_by_name
from models.static_spiking_repvgg import get_StaticSpikingRepVGG_func_by_name
from typing import List, Optional, Tuple

from training.utils import is_main_process, get_cache_path
from training.presets import ClassificationPresetTrain, ClassificationPresetEval, IMAGENET_MEAN, IMAGENET_STD, CIFAR10_MEAN, CIFAR10_STD, CIFAR100_MEAN, CIFAR100_STD
from training.transforms import RandomMixup, RandomCutmix
from training.sampler import RASampler

def build_model(args, logger):
    logger.info(f"Creating model: {args.arch}")
    
    #set surrogate method
    surrogate_function = surrogate.ATan(alpha=args.surrogate_alpha)
    if args.fast_surrogate:
        logger.warning(f"Using surrogate function FastATan which is experimental.")
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
        logger.info(f"The number of timesteps is set to {args.T}: Using multi-step mode. ")
    else:
        functional.set_step_mode(model,'s')

    #set neuron backend
    if args.cupy:
        functional.set_backend(model,'cupy',neuron.IFNode)
        logger.warning("Setting the neuron backend to 'cupy'. This commonly conflicts with automatic mixed precision (AMP) training if enabled.")
        if not args.disable_amp:
            raise ValueError("Neuron backend 'cupy' conflicts with automatic mixed precision (AMP) training. "
                        "Either remove the --cupy flag from your execution, or disable AMP with --disable-amp.")
    return model

def build_transform(args,is_train,mean=IMAGENET_MEAN,std=IMAGENET_STD):
    if is_train:
        crop_size = args.train_crop_size
        interpolation = InterpolationMode(args.interpolation)
        auto_augment_policy = getattr(args,"auto_augment",None)
        random_erase_prob = getattr(args, "random_erase", 0.0)
        ra_magnitude = getattr(args, "ra_magnitude", None)
        augmix_severity = getattr(args, "augmix_severity", None)
        transform = ClassificationPresetTrain(
            crop_size = crop_size,
            mean=mean,
            std=std,
            interpolation = interpolation,
            auto_augment_policy = auto_augment_policy,
            ra_magnitude = ra_magnitude,
            augmix_severity = augmix_severity,
            random_erase_prob = random_erase_prob,
        )
    else:
        crop_size = args.val_crop_size
        resize_size = args.val_resize_size
        interpolation = InterpolationMode(args.interpolation)
        transform = ClassificationPresetEval(
            crop_size=crop_size,
            resize_size=resize_size,
            mean=mean,
            std=std,
            interpolation=interpolation,
        )
    return transform

def build_dataset(args,is_train,logger): #TODO: implement prototype
    prefix = 'train' if is_train else 'val'
    path = os.path.join(args.data_path,prefix)
    logger.info(f'Loading {prefix} data for dataset {args.dataset} from {path}')
    start = time.time()
    cache_path = get_cache_path(path)
    if args.cache_dataset and os.path.exists(cache_path):
        logger.info(f"Loading ImageNet {prefix} data from {cache_path}")
        dataset, _ = torch.load(cache_path)
    elif args.dataset == 'imagenet':
        transform = build_transform(args,is_train)
        logger.info(f"Using raw ImageNet {prefix} data")
        dataset = datasets.ImageNet(root = args.data_path, split=prefix,transform=transform)
    elif args.dataset == 'cf10':
        transform = build_transform(args,is_train,mean=CIFAR10_MEAN,std=CIFAR10_STD)
        logger.info(f"Using raw CIFAR-10 {prefix} data")
        dataset = datasets.CIFAR10(root=args.data_path,train=is_train,download=True,transform=transform)
    elif args.dataset == 'cf100':
        transform = build_transform(args,is_train,mean=CIFAR100_MEAN,std=CIFAR100_STD)
        logger.info(f"Using raw CIFAR-100 {prefix} data")
        dataset = datasets.CIFAR100(root=args.data_path,train=is_train,download=True,transform=transform)
    else:
        logger.warn(f"Using unimplemented dataset {args.dataset}. The lack of finetuned transformers may impact training and validation accuracy.")
        mean, std = args.data_mean, args.data_std
        if mean is None or not (isinstance(mean, tuple) and len(mean)==3):
            logger.warn(f"Unimplemented dataset mean either not passed using --data-mean or invalid. Defaulting to ImageNet mean")
            mean = IMAGENET_MEAN
        if std is None or not (isinstance(std, tuple) and len(std)==3):
            logger.warn(f"Unimplemented dataset std either not passed using --data-mean or invalid. Defaulting to ImageNet std")
            std = IMAGENET_STD
        transform = build_transform(args,is_train,mean=mean,std=std)
        dataset = datasets.ImageFolder(root=path,transform=transform)
    num_classes = len(dataset.classes)
    if args.cache_dataset and not os.path.exists(cache_path):
        os.makedirs(os.path.dirname(cache_path),exist_ok=True)
        if is_main_process():
            torch.save((dataset,path),cache_path)
    logger.info(f"Took {time.time()-start}")
    return dataset, num_classes

def build_collate(args):
    collate_fn = None
    mixup_transforms = []
    if args.mixup_alpha > 0.0:
        mixup_transforms.append(RandomMixup(args.num_classes, p=1.0, alpha=args.mixup_alpha))
    if args.cutmix_alpha > 0.0:
        mixup_transforms.append(RandomCutmix(args.num_classes, p=1.0, alpha=args.cutmix_alpha))
    if mixup_transforms:
        mixupcutmix = torchvision.transforms.RandomChoice(mixup_transforms)
        def collate_fn(batch):
            return mixupcutmix(*default_collate(batch))
    return collate_fn

def build_loader(args,logger):
    dataset_train, args.num_classes = build_dataset(args,is_train=True,logger=logger)
    dataset_val, _ = build_dataset(args,is_train=False,logger=logger)
    
    if args.distributed:
        if args.ra_sampler:
            train_sampler = RASampler(dataset_train,shuffle=True,repititions=args.ra_reps)
        else:
            train_sampler = data.distributed.DistributedSampler(dataset_train, shufle=True)
        val_sampler = data.distributed.DistributedSampler(dataset_val, shuffle=False)
    else:
        train_sampler = data.RandomSampler(dataset_train)
        val_sampler = data.SequentialSampler(dataset_val)

    collate_fn = build_collate(args)

    data_loader_train = data.DataLoader(
        dataset_train, 
        batch_size = args.batch_size,
        sampler=train_sampler,
        num_workers = args.workers,
        pin_memory= not args.disable_pinmemory,
        collate_fn = collate_fn,
    )

    data_loader_val = data.DataLoader(
        dataset_val,
        batch_size = args.batch_size,
        sampler=val_sampler,
        num_workers =args.workers,
        pin_memory= not args.disable_pinmemory
    )

    return dataset_train, dataset_val, data_loader_train, data_loader_val

def set_weight_decay(
    model: torch.nn.Module,
    weight_decay: float,
    norm_weight_decay: Optional[float] = None,
    norm_classes: Optional[List[type]] = None,
    custom_keys_weight_decay: Optional[List[Tuple[str, float]]] = None,
):
    if not norm_classes:
        norm_classes = [
            torch.nn.modules.batchnorm._BatchNorm,
            torch.nn.LayerNorm,
            torch.nn.GroupNorm,
            torch.nn.modules.instancenorm._InstanceNorm,
            torch.nn.LocalResponseNorm,
        ]
    norm_classes = tuple(norm_classes)

    params = {
        "other": [],
        "norm": [],
    }
    params_weight_decay = {
        "other": weight_decay,
        "norm": norm_weight_decay,
    }
    custom_keys = []
    if custom_keys_weight_decay is not None:
        for key, weight_decay in custom_keys_weight_decay:
            params[key] = []
            params_weight_decay[key] = weight_decay
            custom_keys.append(key)

    def _add_params(module, prefix=""):
        for name, p in module.named_parameters(recurse=False):
            if not p.requires_grad:
                continue
            is_custom_key = False
            for key in custom_keys:
                target_name = f"{prefix}.{name}" if prefix != "" and "." in key else name
                if key == target_name:
                    params[key].append(p)
                    is_custom_key = True
                    break
            if not is_custom_key:
                if norm_weight_decay is not None and isinstance(module, norm_classes):
                    params["norm"].append(p)
                else:
                    params["other"].append(p)

        for child_name, child_module in module.named_children():
            child_prefix = f"{prefix}.{child_name}" if prefix != "" else child_name
            _add_params(child_module, prefix=child_prefix)

    _add_params(model)

    param_groups = []
    for key in params:
        if len(params[key]) > 0:
            param_groups.append({"params": params[key], "weight_decay": params_weight_decay[key]})
    return param_groups

def build_optimizer(args, model):
    custom_keys_weight_decay = []
    if args.bias_weight_decay is not None:
        custom_keys_weight_decay.append(("bias",args.bias_weight_decay))
    if args.transformer_embedding_decay is not None:
        for key in ["class_token", "position_embedding", "relative_position_bias_table"]:
            custom_keys_weight_decay.append((key, args.transformer_embedding_decay))
    parameters = set_weight_decay(
        model,
        args.weight_decay,
        norm_weight_decay=args.norm_weight_decay,
        custom_keys_weight_decay=custom_keys_weight_decay if len(custom_keys_weight_decay) > 0 else None,
    )
    args.opt = args.opt.lower()
    if args.opt.startswith("sgd"):
        optimizer = optim.SGD(parameters,lr=args.lr,momentum=args.momentum,weight_decay=args.weight_decay,nesterov="nesterov" in args.opt)
    elif args.opt == "rmsprop":
        optimizer = optim.RMSprop(parameters,lr = args.lr,momentum=args.momentum,weight_decay=args.weight_decay,eps=0.0316,alpha=0.9)
    elif args.opt == "adam":
        optimizer = optim.Adam(parameters,lr=args.lr,weight_decay=args.weight_decay)
    elif args.opt == 'adamw':
        optimizer = optim.AdamW(parameters,lr=args.lr,weight_decay=args.weight_decay)
    else:
        raise RuntimeError(f"Invalid optimizer {args.opt}. Only sgd, rmsprop, adam, and adamw are supported.")
    return optimizer

def build_scheduler(args,optimizer):
    args.lr_scheduler = args.lr_scheduler.lower()
    if args.lr_scheduler == "step":
        main_lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=args.lr_step_size, gamma=args.lr_gamma)
    elif args.lr_scheduler == "cosa":
        main_lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=args.epochs - args.lr_warmup_epochs, eta_min=args.lr_min
        )
    elif args.lr_scheduler == "exp":
        main_lr_scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=args.lr_gamma)
    else:
        raise RuntimeError(
            f"Invalid lr scheduler '{args.lr_scheduler}'. Only StepLR, CosineAnnealingLR and ExponentialLR "
            "are supported."
        )
    if args.lr_warmup_epochs > 0:
        args.lr_warmup_method = args.lr_warmup_method.lower()
        if args.lr_warmup_method == "linear":
            warmup_lr_scheduler = optim.lr_scheduler.LinearLR(
                optimizer, start_factor=args.lr_warmup_decay, total_iters=args.lr_warmup_epochs
            )
        elif args.lr_warmup_method == "constant":
            warmup_lr_scheduler = optim.lr_scheduler.ConstantLR(
                optimizer, factor=args.lr_warmup_decay, total_iters=args.lr_warmup_epochs
            )
        else:
            raise RuntimeError(
                f"Invalid warmup lr method '{args.lr_warmup_method}'. Only linear and constant are supported."
            )
        lr_scheduler = optim.lr_scheduler.SequentialLR(
            optimizer, schedulers=[warmup_lr_scheduler, main_lr_scheduler], milestones=[args.lr_warmup_epochs]
        )
    else:
        lr_scheduler = main_lr_scheduler

    return lr_scheduler

def build_criterion(args):
    criterion = torch.nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
    return criterion


import os
import numpy as np
import torch
from torchvision.transforms import autoaugment, transforms
from torchvision.transforms.functional import InterpolationMode
from ffcv.loader import Loader, OrderOption
from ffcv.transforms import ToTensor, ToDevice, Squeeze, NormalizeImage, RandomHorizontalFlip, ToTorchImage, ImageMixup, LabelMixup
from ffcv.fields.rgb_image import CenterCropRGBImageDecoder, RandomResizedCropRGBImageDecoder
from ffcv.fields.basics import IntDecoder

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2023, 0.1994, 0.2010)
CIFAR100_MEAN = (0.507, 0.487, 0.441)
CIFAR100_STD = (0.267, 0.256, 0.276)

DATASET_STATS = {
    'imagenet': (IMAGENET_MEAN, IMAGENET_STD),
    'cifar10': (CIFAR10_MEAN, CIFAR10_STD),
    'cifar100': (CIFAR100_MEAN, CIFAR100_STD)
}

def build_train_pipelines(args,mean,std):
    res_tuple = (args.train_crop_size,args.train_crop_size)
    image_pipeline = [RandomResizedCropRGBImageDecoder(res_tuple),
                      RandomHorizontalFlip()]
    if args.mixup_alpha > 0.0:
        image_pipeline.append(ImageMixup(alpha=args.mixup_alpha,same_lambda=True))
    if args.auto_augment is not None:
        interpolation = InterpolationMode(args.interpolation)
        if args.auto_augment == 'ra':
            image_pipeline.append(autoaugment.RandAugment(interpolation=interpolation,magnitude=args.ra_magnitude))
        elif args.auto_augment == 'ta_wide':
            image_pipeline.append(autoaugment.TrivialAugmentWide(interpolation=interpolation))
        elif args.auto_augment == 'augmix':
            image_pipeline.append(autoaugment.AugMix(interpolation=interpolation,severity=args.augmix_severity))
        else:
            aa_policy = autoaugment.AutoAugmentPolicy(args.auto_augment)
            image_pipeline.append(autoaugment.AutoAugment(policy=aa_policy,interpolation=interpolation))
    image_pipeline.extend(
        [
            ToTensor(),
            ToDevice(torch.device(args.device),non_blocking=True),
            ToTorchImage(channels_last=args.channels_last), #TODO: test if needed
            NormalizeImage(mean,std,np.float16), #TODO: disable amp
        ]
    )
    if args.random_erase > 0.0:
        image_pipeline.append(transforms.RandomErasing(p=args.random_erase))

    label_pipeline = [IntDecoder()]
    if args.mixup_alpha > 0.0:
        label_pipeline.append(LabelMixup(alpha=args.mixup_alpha,same_lambda=True))
    label_pipeline.extend([
        ToTensor(),
        Squeeze(),
        ToDevice(torch.device(args.device),non_blocking=True)
    ])

    pipelines = {
        'image': image_pipeline,
        'label': label_pipeline,
    }

    return pipelines

def build_val_pipelines(args,mean,std):
    res_tuple = (args.val_resize_size,args.val_resize_size)
    ratio = float(args.val_crop_size) / float(args.val_resize_size)
    image_pipeline = [
        CenterCropRGBImageDecoder(res_tuple,ratio=ratio),
        ToTensor(),
        ToDevice(torch.device(args.device),non_blocking=True),
        ToTorchImage(channels_last=args.channels_last), #TODO: test if needed
        NormalizeImage(mean,std,np.float16),#TODO: disable amp
    ]
    
    label_pipeline = [
        IntDecoder(),
        ToTensor(),
        Squeeze(),
        ToDevice(torch.device(args.device),non_blocking=True)
    ]

    pipelines = {
        'image': image_pipeline,
        'label': label_pipeline,
    }

    return pipelines

def build_loaders(args):
    if args.data_mean is not None and args.data_std is not None:
        mean,std = args.data_mean,args.data_std
    else:
        mean,std = DATASET_STATS[args.dataset]
    mean = np.array(mean)*255
    std = np.array(std)*255

    train_path = os.path.join(args.data_path,'train.ffcv')
    train_pipelines = build_train_pipelines(args,mean,std)
    train_order = OrderOption.RANDOM if args.distributed else OrderOption.QUASI_RANDOM
    train_loader = Loader(
        train_path,
        batch_size=args.batch_size,
        num_workers=args.workers,
        order=train_order,
        os_cache=args.in_memory,
        drop_last=False,
        pipelines=train_pipelines,
        distributed=args.distributed,
        batches_ahead=args.batches_ahead,
    )

    val_path = os.path.join(args.data_path,'val.ffcv')
    val_pipelines = build_val_pipelines(args,mean,std)
    val_loader = Loader(val_path,
        batch_size=args.batch_size,
        num_workers=args.workers,
        order = OrderOption.SEQUENTIAL,
        drop_last=False,
        pipelines=val_pipelines,
        distributed=args.distributed,
        batches_ahead=args.batches_ahead,
    )
    
    return train_loader, val_loader 
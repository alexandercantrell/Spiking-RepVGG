import os
import numpy as np
import torch
from torchvision.transforms import autoaugment, transforms
from torchvision.transforms.functional import InterpolationMode

from ffcv.loader import Loader, OrderOption
from ffcv.transforms import ToTensor, ToDevice, Squeeze, NormalizeImage, RandomHorizontalFlip, ToTorchImage, ImageMixup, LabelMixup
from ffcv.fields.rgb_image import CenterCropRGBImageDecoder, RandomResizedCropRGBImageDecoder
from ffcv.fields.basics import IntDecoder

from fastargs.decorators import param
from fastargs import Param, Section

from train.utils import get_device_name, is_dist_avail_and_initialized


IMAGENET_MEANS = (0.485, 0.456, 0.406)
IMAGENET_STDS = (0.229, 0.224, 0.225)
CIFAR10_MEANS = (0.4914, 0.4822, 0.4465)
CIFAR10_STDS = (0.2023, 0.1994, 0.2010)
CIFAR100_MEANS = (0.507, 0.487, 0.441)
CIFAR100_STDS = (0.267, 0.256, 0.276)

DATASET_STATS = {
    'imagenet': (IMAGENET_MEANS, IMAGENET_STDS),
    'cifar10': (CIFAR10_MEANS, CIFAR10_STDS),
    'cifar100': (CIFAR100_MEANS, CIFAR100_STDS)
}

DATASET_NUM_CLASSES = {
    'imagenet': 1000,
    'cifar10': 10,
    'cifar100': 100
}


Section('data', 'dataset details').params(
    dataset=Param(str,'dataset name',required=True),
    path=Param(str,'path to train.ffcv and val.ffcv',required=True),
    T=Param(int,'T',default=4),
)

Section('data').enable_if(lambda cfg: cfg['data.dataset'] not in DATASET_STATS.keys()).params(
    num_classes=Param(int,'number of classes in dataset',required=True),
    means=Param(str,'dataset means',required=True),
    stds=Param(str,'dataset standard deviation values',required=True)
)

Section('train.pipeline').params(
    crop_size=Param(int,'',default=176),
    mixup_alpha=Param(float,'',default=0.0),
    auto_augment=Param(str,'',default=None),
    random_erase=Param(float,'',default=0.0),
)

Section('train.pipeline.randaugment').enable_if(lambda cfg: cfg['train.pipeline.auto_augment']=='randaugment').params(
    magnitude=Param(int,'',default=9)
)

Section('train.pipeline.augmix').enable_if(lambda cfg: cfg['train.pipeline.auto_augment']=='augmix').params(
    severity=Param(int,'',default=3)
)

Section('train.loader').params(
    batch_size=Param(int,'training batch size',default=512),
    workers=Param(int,'number of workers for train loader',default=16),
    in_memory=Param(bool,'will dataset fit in memory',is_flag=True)
)

Section('val.pipeline').params(
    resize_size=Param(int,'',default=232),
    crop_size=Param(int,'',224),
)

Section('val.loader').params(
    batch_size=Param(int,'validation batch size',default=512),
    workers=Param(int,'number of workers for val loader',default=16),
    in_memory=Param(bool,'will dataset fit in memory',is_flag=True)
)

@param('data.dataset')
@param('data.means')
@param('data.stds')
def get_dataset_stats(dataset,means=None,stds=None):
    if dataset in DATASET_STATS.keys():
        (means,stds)=DATASET_STATS[dataset]
    elif means is not None and stds is not None:
        means = tuple(map(int, means.split(', ')))
        stds = tuple(map(int, stds.split(', ')))
    means = np.array(means)*255
    stds = np.array(stds)*255
    return means, stds

@param('data.dataset')    
@param('data.num_classes')
def get_num_classes(dataset,num_classes=None):
    if dataset in DATASET_NUM_CLASSES.keys():
        num_classes = DATASET_NUM_CLASSES[dataset]
    return num_classes

@param('train.pipeline.randaugment.magnitude')
def build_random_augment(magnitude):
    interpolation=InterpolationMode('bilinear')
    return autoaugment.RandAugment(interpolation=interpolation,magnitude=magnitude)

def build_trivial_augment():
    interpolation=InterpolationMode('bilinear')
    return autoaugment.TrivialAugmentWide(interpolation=interpolation)

@param('train.pipeline.augmix.severity')
def build_augmix(severity):
    interpolation=InterpolationMode('bilinear')
    return autoaugment.AugMix(interpolation=interpolation,severity=severity)

@param('train.pipeline.auto_augment')
def build_autoaugment(auto_augment):
    interpolation=InterpolationMode('bilinear')
    aa_policy=autoaugment.AutoAugmentPolicy(auto_augment)
    return autoaugment.AutoAugment(policy=aa_policy,interpolation=interpolation)

@param('train.pipeline.crop_size')
@param('train.pipeline.mixup_alpha')
@param('train.pipeline.random_erase')
@param('model.disable_amp')
@param('train.pipeline.auto_augment')
def build_train_pipelines(crop_size,mixup_alpha,random_erase,disable_amp,auto_augment=None):
    device = torch.device(get_device_name())
    res_tuple = (crop_size,crop_size)
    image_pipeline=[RandomResizedCropRGBImageDecoder(res_tuple),
                    RandomHorizontalFlip()]
    if mixup_alpha>0:
        image_pipeline.append(ImageMixup(alpha=mixup_alpha,same_lambda=True))
    if auto_augment is not None:
        if auto_augment=='randaugment':
            image_pipeline.append(build_random_augment())
        elif auto_augment=='trivialaugment':
            image_pipeline.append(build_trivial_augment())
        elif auto_augment=='augmix':
            image_pipeline.append(build_augmix())
        else:
            image_pipeline.append(build_autoaugment())
    image_pipeline.extend([
        ToTensor(),
        ToDevice(device,non_blocking=True),
        ToTorchImage(),
        NormalizeImage(*get_dataset_stats(),np.float32 if disable_amp else np.float16)
    ])
    if random_erase>0:
        image_pipeline.append(transforms.RandomErasing(p=random_erase))

    label_pipeline=[IntDecoder()]
    if mixup_alpha>0:
        label_pipeline.append(LabelMixup(alpha=mixup_alpha,same_lambda=True))
    label_pipeline.extend([
        ToTensor(),
        Squeeze(),
        ToDevice(device,non_blocking=True)
    ])

    pipelines = {
        'image': image_pipeline,
        'label': label_pipeline
    }
    return pipelines

@param('data.path')
@param('train.loader.batch_size')
@param('train.loader.workers')
@param('train.loader.in_memory')
def build_train_loader(path,batch_size,workers,in_memory):
    distributed = is_dist_avail_and_initialized()
    path = os.path.join(path,'train.ffcv')
    pipelines = build_train_pipelines()
    order = OrderOption.RANDOM if distributed else OrderOption.QUASI_RANDOM 
    loader = Loader(
        path,
        batch_size=batch_size,
        num_workers=workers,
        order=order,
        os_cache=in_memory,
        drop_last=False,
        pipelines=pipelines,
        distributed=distributed,
    )
    return loader

@param('val.pipeline.resize_size')
@param('val.pipeline.crop_size')
@param('model.disable_amp')
def build_val_pipelines(resize_size,crop_size,disable_amp):
    device = torch.device(get_device_name())
    res_tuple=(resize_size,resize_size)
    ratio = float(crop_size)/float(resize_size)
    image_pipeline=[
        CenterCropRGBImageDecoder(res_tuple,ratio=ratio),
        ToTensor(),
        ToDevice(device,non_blocking=True),
        ToTorchImage(),
        NormalizeImage(*get_dataset_stats(),np.float32 if disable_amp else np.float16)
    ]

    label_pipeline=[
        IntDecoder(),
        ToTensor(),
        Squeeze(),
        ToDevice(device,non_blocking=True),
    ]

    pipelines={
        'image':image_pipeline,
        'label':label_pipeline
    }

    return pipelines

@param('data.path')
@param('val.loader.batch_size')
@param('val.loader.workers')
@param('val.loader.in_memory')
def build_val_loader(path, batch_size, workers, in_memory):
    distributed = is_dist_avail_and_initialized()
    path = os.path.join(path,'val.ffcv')
    pipelines = build_val_pipelines()
    loader = Loader(
        path,
        batch_size=batch_size,
        num_workers=workers,
        order=OrderOption.SEQUENTIAL,
        os_cache=in_memory,
        drop_last=False,
        pipelines=pipelines,
        distributed=distributed, 
    )
    return loader


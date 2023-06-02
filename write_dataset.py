import os
from torch.utils.data import Subset
from ffcv.writer import DatasetWriter
from ffcv.fields import IntField, RGBImageField
from torchvision.datasets import CIFAR10, CIFAR100, ImageNet, ImageFolder

import argparse

def get_args_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument("--dataset",default='imagenet',type=str)
    parser.add_argument("--split",required=True,type=str,choices=["train","val"])
    parser.add_argument("--data-path",required=True, type=str)
    parser.add_argument("--write-path",required=True,type=str)
    parser.add_argument("--write-mode",default='smart',type=str,choices=['raw','smart','jpg'])
    parser.add_argument("--max-resolution",default=None,type=int)
    parser.add_argument("--num-workers",default=16,type=int)
    parser.add_argument("--chunk-size",default=100,type=int)
    parser.add_argument("--jpeg-quality",default=90,type=float)
    parser.add_argument("--subset",default=-1,type=int)
    parser.add_argument("--compress-probability",default=0.5,type=float)

    return parser

def main(args):
    if args.dataset == 'cifar10':
        dataset = CIFAR10(root=args.data_path,train=(args.split=='train'),download=True)
    elif args.dataset == 'cifar100':
        dataset = CIFAR100(root=args.data_path,train=(args.split=='train'),download=True)
    elif args.dataset == 'imagenet':
        dataset = ImageNet(root=args.data_path,split=args.split)
    else:
        dataset = ImageFolder(root=os.path.join(args.data_path,args.split))
    if args.subset>0: 
        dataset = Subset(dataset,range(args.subset))
    os.makedirs(args.write_path)
    path = os.path.join(args.write_path,f'{args.split}.ffcv')
    writer = DatasetWriter(path, {
        'image': RGBImageField(write_mode=args.write_mode,
                               max_resolution=args.max_resolution,
                               compress_probability=args.compress_probability,
                               jpeg_quality=args.jpeg_quality),
        'label': IntField(),
    }, num_workers = args.num_workers)
    writer.from_indexed_dataset(dataset,chunksize=args.chunk_size)

if __name__ == '__main__':
    args = get_args_parser().parse_args()
    main(args)
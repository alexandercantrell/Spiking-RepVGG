import os
import math
import json
import torch as ch
import numpy as np
from argparse import ArgumentParser

from fastargs import get_current_config
from fastargs.decorators import param
from fastargs import Param, Section
from fastargs.validation import And, OneOf

from dst_models import get_model_by_name
from spikingjelly.datasets import cifar10_dvs

from syops import get_model_complexity_info

SEED=2020
import random
random.seed(SEED)
ch.backends.cudnn.deterministic = True
ch.backends.cudnn.benchmark = False
ch.manual_seed(SEED)
ch.cuda.manual_seed_all(SEED)
np.random.seed(SEED)

Section('run', 'run details').params(
    path=Param(str, 'path to run dir', required=True),
)

Section('data', 'data details').params(
    path=Param(str, 'path to data dir', required=True),
    T=Param(int, 'number of frames', required=True),
    num_workers=Param(int, 'number of workers', default=4),
    train_ratio=Param(float, 'train ratio', default=0.9),
    random_split=Param(bool, 'random split', is_flat=True),
    batch_size=Param(int, 'batch size', default=128),
)

def create_model(arch, block_type, checkpoint):
    model = get_model_by_name(arch)(num_classes=10, block_type=block_type)
    model.load_state_dict(ch.load(checkpoint)['model'], strict=False)
    model.switch_to_deploy()
    return model

@param('data.train_ratio')
@param('data.random_split')
def split_val(dataset, train_ratio, random_split):
    label_idx = [[] for _ in range(len(dataset.classes))]
    for idx, item in enumerate(dataset):
            y = item[1]
            if isinstance(y, np.ndarray) or isinstance(y,ch.Tensor):
                y = y.item()
            label_idx[y].append(idx)
    val_idx = []
    if random_split:
        for idx in range(len(dataset.classes)):
            np.random.shuffle(label_idx[idx])
        for idx in range(len(dataset.classes)):
            pos = math.ceil(label_idx[idx].__len__()*train_ratio)
            val_idx.extend(label_idx[idx][pos:label_idx[idx].__len__()])
        return ch.utils.data.Subset(dataset, val_idx)

@param('data.path')
@param('data.T')
@param('data.num_workers')
@param('data.batch_size')
def create_data_loader(path, T, num_workers, batch_size):
    dataset = cifar10_dvs.CIFAR10DVS(path, data_type='frame', frames_number=T, split_by='number')
    val_set = split_val(dataset)
    val_laoder = ch.utils.data.DataLoader(
        val_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False
    )
    return val_laoder

@param('run.path')
def main(path):
    with open(os.path.join(path, 'params.json'), 'r') as params_file:
        params = json.load(params_file)
    val_loader = create_data_loader()
    model=create_model(params['model.arch'], params['model.block_type'], os.path.join(path, 'pt', 'best_checkpoint.pt'))
    ops, params = get_model_complexity_info(model, (2, 128, 128), val_loader, as_strings=True,
                                                 print_per_layer_stat=True, verbose=True)
    print('ops: ', ops)
    print('params: ', params)
    

def make_config(quiet=False):
    config = get_current_config()
    parser = ArgumentParser(description='Calculate energy of a model')
    config.augment_argparse(parser)
    config.collect_argparse_args(parser)
    config.validate(mode='stderr')
    if not quiet:
        config.summary()

if __name__ == '__main__':
    make_config()
    main()

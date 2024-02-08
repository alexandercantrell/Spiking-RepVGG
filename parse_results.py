import torch
import os
import json
import pandas as pd

from argparse import ArgumentParser
from fastargs import get_current_config
from fastargs.decorators import param
from fastargs import Param, Section

Section('logging','logging params').params(
    folder=Param(str, 'loc location',default='./logs/')
)

@param('logging.folder')
def parse_results(folder):
    results = []
    for dataset in os.listdir(folder):
        ds_path = os.path.join(folder, dataset)
        for arch in os.listdir(ds_path):
            arch_path = os.path.join(ds_path, arch)
            for block_type in os.listdir(arch_path):
                block_path = os.path.join(arch_path, block_type)
                for tag in os.listdir(block_path):
                    tag_path = os.path.join(block_path, tag)
                    if os.path.exists(os.path.join(tag_path, 'results.json')):
                        params = json.load(open(os.path.join(tag_path, 'params.json')))
                        res = json.load(open(os.path.join(tag_path, 'results.json')))
                        check_path = os.path.join(tag_path,'pt','best_checkpoint.pt')
                        checkpoint = torch.load(check_path,map_location='cpu')
                        results.append({
                            'dataset': dataset,
                            'arch': arch,
                            'block_type': block_type,
                            'tag': tag,
                            'T': params['data.T'],
                            'lr': params['lr.lr'],
                            'wd': params['optim.weight_decay'],
                            'params': res['params_string'],
                            'energy': res['energy_string'],
                            'syops': res['syops_string'],
                            'ac_ops': res['ac_ops_string'],
                            'mac_ops': res['mac_ops_string'],
                            'best_epoch':checkpoint['epoch'],
                            'max_acc':checkpoint['max_accuracy'],
                        })
    results = pd.DataFrame(results)
    print(results.head())
    results.to_csv('results_summary.csv',index=False)

def make_config(quiet=False):
    config = get_current_config()
    parser = ArgumentParser(description='Parse results from a folder')
    config.augment_argparse(parser)
    config.collect_argparse_args(parser)
    config.validate(mode='stderr')
    if not quiet:
        config.summary()

if __name__ == '__main__':
    make_config()
    parse_results()
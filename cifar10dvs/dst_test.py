import torch as ch
from torch.cuda.amp import autocast
import torch.distributed as dist
import torchmetrics
import numpy as np
from tqdm import tqdm
import math

import os
import time
import json
from argparse import ArgumentParser

from fastargs import get_current_config
from fastargs.decorators import param
from fastargs import Param, Section
from fastargs.validation import And, OneOf

from dst_repvgg import model_dict as repvgg_model_dict
from dst_resnet import model_dict as resnet_model_dict
from dst_spikeformer import model_dict as spikeformer_model_dict

from spikingjelly.activation_based import neuron, functional, monitor
from spikingjelly.datasets import cifar10_dvs
from syops import get_model_complexity_info
from syops.utils import syops_to_string, params_to_string
from ops import MODULES_MAPPING
import connecting_neuron
from connecting_functions import ConnectingFunction

SEED=2020
import random
random.seed(SEED)
ch.backends.cudnn.deterministic = True
ch.backends.cudnn.benchmark = False
ch.manual_seed(SEED)
ch.cuda.manual_seed_all(SEED)
np.random.seed(SEED)

Section('model','model detials').params(
    arch=Param(str,'model arch',required=True),
    cupy=Param(bool, 'use cupy backend for neurons',is_flag=True),
    block_type=Param(str, 'block type',default='spike_connecting'),
    dsnn=Param(bool, 'use dsnn', is_flag=True),
    cnf=Param(str,'cnf',default=None),
    use_new = Param(bool, 'use new model', is_flag=True)
)

Section('data', 'data related stuff').params(
    path = Param(str, 'path to dataset folder', required=True),
    T=Param(int,'T',default=16),
    num_workers=Param(int,'the number of workers',default=4),
    train_ratio=Param(float, 'The ratio of training set', default=0.9),
    random_split=Param(bool, 'Whether to use random split', is_flag=True),
)
    
Section('logging','logging params').params(
    folder=Param(str, 'loc location',required=True)
)

Section('validation', 'validation related stuff').params(
    batch_size=Param(int, 'batch size', default=16),
)

Section('dist', 'distributed training options').params(
    world_size=Param(int, 'number gpus', default=1),
    address=Param(str, 'address', default='localhost'),
    port=Param(str, 'port', default='12355'),
    distributed=Param(bool, 'use distributed mode', is_flag=True),
)

class Tester:
    @param('dist.distributed')
    def __init__(self, gpu, distributed):
        self.all_params = get_current_config()
        self.gpu = gpu
        self.num_classes=10
        if distributed:
            self.setup_distributed()
        self.initialize_logger()
        self.loader = self.create_data_loader()
        self.model = self.create_model()

    @param('dist.address')
    @param('dist.port')
    @param('dist.world_size')
    def setup_distributed(self, address, port, world_size):
        os.environ['MASTER_ADDR'] = address
        os.environ['MASTER_PORT'] = port

        dist.init_process_group("nccl", rank=self.gpu, world_size=world_size)
        ch.cuda.set_device(self.gpu)

    def cleanup_distributed(self):
        dist.destroy_process_group()

    @param('data.train_ratio')
    @param('data.random_split')
    def _split_val(self, dataset, train_ratio, random_split=False):
        label_idx = [[] for _ in range(self.num_classes)]
        for idx, item in enumerate(dataset):
            y = item[1]
            if isinstance(y, np.ndarray) or isinstance(y,ch.Tensor):
                y = y.item()
            label_idx[y].append(idx)
        val_idx = []
        if random_split:
            for idx in range(self.num_classes):
                np.random.shuffle(label_idx[idx])
        for idx in range(self.num_classes):
            pos = math.ceil(label_idx[idx].__len__()*train_ratio)
            val_idx.extend(label_idx[idx][pos:label_idx[idx].__len__()])
        return ch.utils.data.Subset(dataset, val_idx)

    @param('data.path')
    @param('data.T')
    @param('data.num_workers')
    @param('validation.batch_size')
    def create_data_loader(self, path, T, num_workers, batch_size):
        dataset = cifar10_dvs.CIFAR10DVS(path, data_type='frame',frames_number=T,split_by='number')
        val_set = self._split_val(dataset)
        val_loader = ch.utils.data.DataLoader(
            val_set,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
            drop_last=False,
        )
        return val_loader
    
    @param('model.arch')
    @param('model.block_type')
    @param('model.cupy')
    @param('dist.distributed')
    @param('model.use_new')
    @param('model.dsnn')
    @param('model.cnf')
    def create_model(self, arch, cupy, block_type, distributed, use_new=False, dsnn=False, cnf=None):
        if arch in repvgg_model_dict.keys():
            model = repvgg_model_dict[arch](num_classes=self.num_classes,block_type=block_type)
        elif arch in resnet_model_dict.keys():
            model = resnet_model_dict[arch](num_classes=self.num_classes,block_type=block_type, cnf=cnf, dsnn=dsnn)
        elif arch in spikeformer_model_dict.keys():
            model = spikeformer_model_dict[arch](num_classes=self.num_classes)
        else:
            raise NotImplementedError(f"Model {arch} not implemented")
        functional.set_step_mode(model,'m')
        if cupy:
            functional.set_backend(model,'cupy',instance=neuron.ParametricLIFNode)
            functional.set_backend(model,'cupy',instance=connecting_neuron.ParaConnLIFNode)
        
        model = model.to(self.gpu)
        if not use_new:
            model.load_state_dict(ch.load(os.path.join(self.pt_folder,'best_checkpoint.pt'))['model'], strict=False)
        
        if hasattr(model,'switch_to_deploy'):
            model.switch_to_deploy()

        if distributed:
            model = ch.nn.parallel.DistributedDataParallel(model, device_ids=[self.gpu])

        return model
    
    def evaluate(self):
        self.calculate_complexity()
        stats = self.val_loop()
        self.log(f"Reparameterized stats {stats}")
        self._save_results(stats)
        self.calculate_spike_rates()

    def calculate_complexity(self):
        ops, params = get_model_complexity_info(self.model, (2, 128, 128), self.loader, as_strings=False,
                                                 print_per_layer_stat=True, verbose=True, custom_modules_hooks=MODULES_MAPPING)
        self.syops_count = ops
        self.params_count = params
        self.total_energy = (ops[1]*0.9 + ops[2]*4.6)*1e-9
        self.energy_string = f'{round(self.total_energy,2)} mJ'
        self.syops_string = f'{syops_to_string(ops[0],units="G Ops",precision=2)}'
        self.ac_ops_string = f'{syops_to_string(ops[1],units="G Ops",precision=2)}'
        self.mac_ops_string = f'{syops_to_string(ops[2],units="G Ops",precision=2)}'
        self.params_string = f'{params_to_string(params,units="M",precision=2)}'
        self.log(f'Total Syops: {self.syops_string}')
        self.log(f'AC Ops: {self.ac_ops_string}')
        self.log(f'MAC Ops: {self.mac_ops_string}')
        self.log(f'Total Energy: {self.energy_string}')
        self.log(f'Params: {self.params_string}')

    def val_loop(self):
        model = self.model
        model.eval()

        with ch.no_grad():
            with autocast():
                for images, target in tqdm(self.loader):
                    start = time.time()
                    images = images.to(self.gpu, non_blocking=True).float()
                    target = target.to(self.gpu, non_blocking=True)
                    output = self.model(images)
                    functional.reset_net(model)
                    end = time.time()
                    if type(output) is tuple:
                        output, aac = output
                    if hasattr(self.model,'dsnn') and self.model.dsnn:
                        output = output + aac
                    self.meters['top_1'].update(output, target)
                    self.meters['top_5'].update(output, target)
                    batch_size = target.shape[0]
                    self.meters['thru'].update(ch.tensor(batch_size/(end-start)))

        stats = {k: m.compute().item() for k, m in self.meters.items()}
        [meter.reset() for meter in self.meters.values()]
        return stats

    def calculate_spike_rates(self):
        model=self.model
        model.eval()
        spike_seq_monitor = monitor.OutputMonitor(
            model, 
            (
                neuron.ParametricLIFNode,
                neuron.LIFNode,
                neuron.IFNode,
                connecting_neuron.ParaConnLIFNode,
                connecting_neuron.ConnLIFNode,
                ), 
            lambda x: x.mean().item())
        spike_rates = None
        
        cnf_seq_monitor = monitor.OutputMonitor(
            model, 
            (
                ConnectingFunction,
                ), 
            lambda x: x.mean().item())
        cnf_rates = None

        cnt = 0
        with ch.no_grad():
            with autocast():
                for images, target in tqdm(self.loader):
                    images = images.to(self.gpu, non_blocking=True).float()
                    target = target.to(self.gpu, non_blocking=True)
                    output = self.model(images)
                    functional.reset_net(model)
                    if spike_rates is None:
                        spike_rates = spike_seq_monitor.records
                        cnf_rates = cnf_seq_monitor.records
                    else:
                        spike_rates = [spike_rates[i]+spike_seq_monitor.records[i] for i in range(len(spike_rates))]
                        cnf_rates = [cnf_rates[i]+cnf_seq_monitor.records[i] for i in range(len(cnf_rates))]
                    cnt += 1
                    spike_seq_monitor.clear_recorded_data()
                    cnf_seq_monitor.clear_recorded_data()
        spike_rates = [spike_rate/cnt for spike_rate in spike_rates]
        cnf_rates = [cnf_rate/cnt for cnf_rate in cnf_rates]
        spike_seq_monitor.remove_hooks()
        cnf_seq_monitor.remove_hooks()
        self.log(f'Spike rates: {spike_rates}')
        self.log(f'Cnf rates: {cnf_rates}')
        if self.gpu==0:
            with open(os.path.join(self.log_folder, 'spike_rates.json'), 'w+') as handle:
                json.dump(spike_rates, handle)
            with open(os.path.join(self.log_folder, 'cnf_rates.json'), 'w+') as handle:
                json.dump(cnf_rates, handle)
                    

    @param('logging.folder')
    def initialize_logger(self, folder):
        self.meters = {
            'top_1': torchmetrics.Accuracy(task='multiclass',num_classes=self.num_classes).to(self.gpu),
            'top_5': torchmetrics.Accuracy(task='multiclass',num_classes=self.num_classes, top_k=5).to(self.gpu),
            'thru': MeanScalarMetric().to(self.gpu),
        }

        if self.gpu == 0:
            self.log_folder = folder
            pt_folder = os.path.join(folder,'pt')
            self.pt_folder = pt_folder
            checkpoint_folder= os.path.join(pt_folder,'checkpoints')
            self.checkpoint_folder=checkpoint_folder
            tb_folder = os.path.join(folder,'tb')
            self.tb_folder = tb_folder

            self.log_folder = folder
            self.start_time = time.time()

            print(f'=> Logging in {self.log_folder}')
            params = {
                '.'.join(k): self.all_params[k] for k in self.all_params.entries.keys()
            }

            with open(os.path.join(folder, 'val_params.json'), 'w+') as handle:
                json.dump(params, handle)

    def log(self, content):
        print(f'=> Log: {content}')
        if self.gpu != 0: return
        with open(os.path.join(self.log_folder, 'log'), 'a+') as fd:
            fd.write(content + '\n')
            fd.flush()

    @param('dist.distributed')
    def _save_results(self, stats, distributed):
        if distributed:
            dist.barrier()
        results = {
            'syops_count': self.syops_count.tolist(),
            'params_count': self.params_count,
            'total_energy': self.total_energy,
            'energy_string': self.energy_string,
            'syops_string': self.syops_string,
            'ac_ops_string': self.ac_ops_string,
            'mac_ops_string': self.mac_ops_string,
            'params_string': self.params_string,
            'final_acc': stats['top_1'],
            'final_thru': stats['thru'],
        }
        if self.gpu==0:
            with open(os.path.join(self.log_folder, 'val_results.json'), 'w+') as handle:
                json.dump(results, handle)

    @classmethod
    @param('dist.distributed')
    @param('dist.world_size')
    def launch_from_args(cls, distributed, world_size):
        if distributed:
            ch.multiprocessing.spawn(cls._exec_wrapper, nprocs=world_size, join=True)
        else:
            cls.exec(0)

    @classmethod
    def _exec_wrapper(cls, *args, **kwargs):
        make_config(quiet=True)
        cls.exec(*args, **kwargs)

    @classmethod
    @param('dist.distributed')
    def exec(cls, gpu, distributed):
        tester = cls(gpu=gpu)
        tester.evaluate()

        if distributed:
            tester.cleanup_distributed()


# Utils
class MeanScalarMetric(torchmetrics.Metric):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.add_state('sum', default=ch.tensor(0.), dist_reduce_fx='sum')
        self.add_state('count', default=ch.tensor(0), dist_reduce_fx='sum')

    def update(self, sample: ch.Tensor):
        self.sum += sample.sum()
        self.count += sample.numel()

    def compute(self):
        return self.sum.float() / self.count
    

# Running
def make_config(quiet=False):
    config = get_current_config()
    parser = ArgumentParser(description='Fast training')
    config.augment_argparse(parser)
    config.collect_argparse_args(parser)
    config.validate(mode='stderr')
    if not quiet:
        config.summary()
    
if __name__ == "__main__":
    make_config()
    Tester.launch_from_args()
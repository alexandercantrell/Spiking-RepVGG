import datetime
import torch as ch
from torch.cuda.amp import GradScaler
from torch.cuda.amp import autocast
import torch.distributed as dist
from torch.utils.tensorboard import SummaryWriter
import torchmetrics
import numpy as np
from tqdm import tqdm
import math

import shutil

import os
import time
import json
from argparse import ArgumentParser

from fastargs import get_current_config
from fastargs.decorators import param
from fastargs import Param, Section
from fastargs.validation import And, OneOf

from dst_models import get_model_by_name

from spikingjelly.activation_based import neuron, functional
from spikingjelly.datasets import cifar10_dvs
from syops import get_model_complexity_info
from syops.utils import syops_to_string, params_to_string
from ops import MODULES_MAPPING
import connecting_neuron

SEED=2020
import random
random.seed(SEED)
ch.backends.cudnn.deterministic = True
ch.backends.cudnn.benchmark = False
ch.manual_seed(SEED)
ch.cuda.manual_seed_all(SEED)
np.random.seed(SEED)

Section('model', 'model details').params(
    arch=Param(str, 'model arch', required=True),
    resume = Param(str,'checkpoint to load from',default=None),
    cupy = Param(bool,'use cupy backend for neurons',is_flag=True),
    block_type = Param(str,'block type',default='spike_connecting'),
)

Section('model').enable_if(lambda cfg: cfg['dist.distributed']==True).params(
    sync_bn = Param(bool,'enable batch norm syncing when in distributed mode',is_flag=True),
)

Section('data', 'data related stuff').params(
    path = Param(str,'path to dataset folder',required=True),
    T=Param(int,'T',default=16),
    num_workers=Param(int, 'The number of workers', default=4),
    train_ratio=Param(float, 'The ratio of training set', default=0.9),
    random_split=Param(bool, 'Whether to use random split', is_flag=True),
)

Section('lr','lr scheduling').params(
    lr=Param(float,'',default=0.01),
    eta_min=Param(float,'',default=0.0)
) 

Section('logging', 'how to log stuff').params(
    folder=Param(str, 'log location', default='./logs/'),
    tag=Param(str,'experiment tag',default='default'),
    clean=Param(bool,'clean prior experiment folder if exists',is_flag=True)
)

Section('validation', 'Validation parameters stuff').params(
    batch_size=Param(int, 'The validation batch size for validation', default=16),
    eval_only=Param(bool,'only perform evaluation',is_flag=True)
)

Section('optim','optimizer hyper params').params(
    momentum=Param(float, 'SGD momentum', default=0.9),
    weight_decay=Param(float, 'weight decay', default=0.0),
)

Section('training', 'training hyper param stuff').params(
    batch_size=Param(int, 'The training batch size', default=16),
    epochs=Param(int, 'number of epochs', default=64),
)

Section('dist', 'distributed training options').params(
    world_size=Param(int, 'number gpus', default=1),
    address=Param(str, 'address', default='localhost'),
    port=Param(str, 'port', default='12355'),
    distributed=Param(bool, 'use distributed mode', is_flag=True),
)

class Trainer:
    @param('dist.distributed')
    @param('model.resume')
    def __init__(self, gpu, distributed, resume=None):
        self.all_params = get_current_config()
        self.gpu = gpu
        self.num_classes = 10
        if distributed:
            self.setup_distributed()
        self.train_loader, self.val_loader = self.create_data_loaders()
        self.model, self.scaler = self.create_model_and_scaler()
        self.start_epoch = 0
        self.max_accuracy=-1
        if resume is not None:
            self.load_checkpoint()
        self.create_optimizer()
        self.create_scheduler()
        self.initialize_logger()
    
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
    def _split_train_val(self,dataset,train_ratio, random_split=False):
        label_idx = [[] for _ in range(self.num_classes)]
        for idx, item in enumerate(dataset):
            y = item[1]
            if isinstance(y, np.ndarray) or isinstance(y,ch.Tensor):
                y = y.item()
            label_idx[y].append(idx)
        train_idx = []
        val_idx = []
        if random_split:
            for idx in range(self.num_classes):
                np.random.shuffle(label_idx[idx])
        for idx in range(self.num_classes):
            pos = math.ceil(label_idx[idx].__len__()*train_ratio)
            train_idx.extend(label_idx[idx][0:pos])
            val_idx.extend(label_idx[idx][pos:label_idx[idx].__len__()])
        return ch.utils.data.Subset(dataset, train_idx), ch.utils.data.Subset(dataset, val_idx)

    @param('data.num_workers')
    @param('training.batch_size')
    def _create_train_loader(self, train_set, num_workers, batch_size):
        train_loader = ch.utils.data.DataLoader(
            train_set,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True,
            drop_last=True,
        )
        return train_loader
    
    @param('data.num_workers')
    @param('validation.batch_size')
    def _create_val_loader(self, val_set, num_workers, batch_size):
        val_loader = ch.utils.data.DataLoader(
            val_set,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
            drop_last=False,
        )
        return val_loader

    @param('data.path')
    @param('data.T')
    def create_data_loaders(self,path,T):
        dataset = cifar10_dvs.CIFAR10DVS(path, data_type='frame',frames_number=T,split_by='number')
        train_set, val_set = self._split_train_val(dataset)
        train_loader = self._create_train_loader(train_set)
        val_loader = self._create_val_loader(val_set)
        return train_loader, val_loader
    
    @param('model.arch')
    @param('model.block_type')
    @param('model.cupy')
    @param('dist.distributed')
    @param('model.sync_bn')
    def create_model_and_scaler(self, arch, cupy, block_type, distributed,  sync_bn=None):
        scaler = GradScaler()
        
        model = get_model_by_name(arch)(num_classes=self.num_classes,block_type=block_type)
        functional.set_step_mode(model,'m')
        if cupy:
            functional.set_backend(model,'cupy',instance=neuron.ParametricLIFNode)
            functional.set_backend(model,'cupy',instance=connecting_neuron.ParaConnLIFNode)
        model = model.to(self.gpu)

        if distributed:
            if sync_bn: model = ch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
            model = ch.nn.parallel.DistributedDataParallel(model, device_ids=[self.gpu])

        return model, scaler
    
    @param('lr.lr')
    @param('optim.momentum')
    @param('optim.weight_decay')
    def create_optimizer(self, lr, momentum, weight_decay):
        self.optimizer = ch.optim.SGD(self.model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)
        self.loss = ch.nn.CrossEntropyLoss()

    @param('training.epochs')
    @param('lr.eta_min')
    def create_scheduler(self,epochs,eta_min):
        self.lr_scheduler = ch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer,T_max=epochs,eta_min=eta_min,last_epoch=self.start_epoch-1)

    @param('training.epochs')
    def train(self, epochs):
        start_train = time.time()
        self.tb_writer = SummaryWriter(self.tb_folder,purge_step=self.start_epoch)
        for epoch in range(self.start_epoch,epochs):
            train_stats = self.train_loop()
            if self.gpu==0:
                self.log(f"Epoch[{epoch}/{epochs}] Train: {dict(train_stats,lr=self.optimizer.param_groups[0]['lr'])}")
                for stat in ['loss','top_1','top_5']:
                    self.tb_writer.add_scalar(f'train_{stat}',train_stats[stat],epoch)
            val_stats = self.val_loop()
            self.max_accuracy = max(self.max_accuracy,val_stats['top_1'])
            if self.gpu==0:
                self.log(f"Epoch[{epoch}/{epochs}] Validation: {val_stats}")
                for stat in ['loss','top_1','top_5']:
                    self.tb_writer.add_scalar(f'val_{stat}',val_stats[stat],epoch)
                self.save_latest(epoch)
                self.save_checkpoint(epoch,is_best=False)
                if self.max_accuracy == val_stats['top_1']:
                    self.save_checkpoint(epoch,is_best=True)

            self.lr_scheduler.step()
        train_time = time.time()-start_train
        train_time_str = str(datetime.timedelta(seconds=int(train_time)))
        self.log(f'Training time {train_time_str}')
        self.log(f'Max accuracy {self.max_accuracy}')
        self.calculate_complexity()
        self._save_results()

    def calculate_complexity(self):
        self.model.load_state_dict(ch.load(os.path.join(self.pt_folder,'best_checkpoint.pt'))['model'], strict=False)
        self.model.switch_to_deploy()
        ops, params = get_model_complexity_info(self.model, (2, 128, 128), self.val_loader, as_strings=False,
                                                 print_per_layer_stat=True, verbose=True, custom_modules_hooks=MODULES_MAPPING)
        self.syops_count = ops
        self.params_count = params
        self.total_energy = (ops[1]*0.9 + ops[2]*4.6)*1e-9
        self.energy_string = f'{self.total_energy} mJ'
        self.syops_string = f'{syops_to_string(ops[0],units="G Ops",precision=2)}'
        self.ac_ops_string = f'{syops_to_string(ops[1],units="G Ops",precision=2)}'
        self.mac_ops_string = f'{syops_to_string(ops[2],units="G Ops",precision=2)}'
        self.params_string = f'{params_to_string(params,units="M",precision=2)}'
        self.log(f'Total Syops: {self.syops_string}')
        self.log(f'AC Ops: {self.ac_ops_string}')
        self.log(f'MAC Ops: {self.mac_ops_string}')
        self.log(f'Total Energy: {self.energy_string}')
        self.log(f'Params: {self.params_string}')

    def eval_and_log(self):
        start_val = time.time()
        stats = self.val_loop()
        val_time = time.time() - start_val
        if self.gpu == 0:
            self.log(dict(
                stats,
                current_lr=self.optimizer.param_groups[0]['lr'],
                val_time=val_time
            ))

    def train_loop(self):
        model = self.model
        model.train()

        for images, target in tqdm(self.train_loader):
            start = time.time()
            self.optimizer.zero_grad(set_to_none=True)
            images = images.to(self.gpu, non_blocking=True).float()
            target = target.to(self.gpu, non_blocking=True)
            with autocast():
                (output, aac) = self.model(images)
                loss_train = self.loss(output, target) + self.loss(aac, target)
            self.scaler.scale(loss_train).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()
            functional.reset_net(model)
            end = time.time()
            output=output.detach()
            target=target.detach()
            self.meters['top_1'](output,target)
            self.meters['top_5'](output,target)
            batch_size=target.shape[0]
            self.meters['thru'](ch.tensor(batch_size/(end-start)))
            self.meters['loss'](loss_train.detach())

        stats = {k:m.compute().item() for k, m in self.meters.items()}
        [meter.reset() for meter in self.meters.values()]
        return stats

    def val_loop(self):
        model = self.model
        model.eval()

        with ch.no_grad():
            with autocast():
                for images, target in tqdm(self.val_loader):
                    start = time.time()
                    images = images.to(self.gpu, non_blocking=True).float()
                    target = target.to(self.gpu, non_blocking=True)
                    (output, aac) = self.model(images)
                    functional.reset_net(model)
                    end = time.time()
                    self.meters['top_1'](output, target)
                    self.meters['top_5'](output, target)
                    batch_size = target.shape[0]
                    self.meters['thru'](ch.tensor(batch_size/(end-start)))
                    loss_val = self.loss(output,target)
                    self.meters['loss'](loss_val)

        stats = {k: m.compute().item() for k, m in self.meters.items()}
        [meter.reset() for meter in self.meters.values()]
        return stats

    @param('logging.folder')
    @param('logging.tag')
    @param('model.arch')
    @param('model.block_type')
    @param('logging.clean')
    def initialize_logger(self, folder, tag, arch, block_type, clean=None):
        self.meters = {
            'top_1': torchmetrics.Accuracy(task='multiclass',num_classes=self.num_classes,compute_on_step=False).to(self.gpu),
            'top_5': torchmetrics.Accuracy(task='multiclass',num_classes=self.num_classes,compute_on_step=False, top_k=5).to(self.gpu),
            'thru': MeanScalarMetric(compute_on_step=False).to(self.gpu),
            'loss': MeanScalarMetric(compute_on_step=False).to(self.gpu)
        }

        if self.gpu == 0:
            folder = os.path.join(folder,'cifar10_dvs',arch,block_type,tag)
            if os.path.exists(folder) and clean:
                shutil.rmtree(folder)
            os.makedirs(folder,exist_ok=True)
            self.log_folder = folder
            pt_folder = os.path.join(folder,'pt')
            os.makedirs(pt_folder,exist_ok=True)
            self.pt_folder = pt_folder
            checkpoint_folder= os.path.join(pt_folder,'checkpoints')
            os.makedirs(checkpoint_folder,exist_ok=True)
            self.checkpoint_folder=checkpoint_folder
            tb_folder = os.path.join(folder,'tb')
            os.makedirs(tb_folder,exist_ok=True)
            self.tb_folder = tb_folder

            self.log_folder = folder
            self.start_time = time.time()

            print(f'=> Logging in {self.log_folder}')
            params = {
                '.'.join(k): self.all_params[k] for k in self.all_params.entries.keys()
            }

            with open(os.path.join(folder, 'params.json'), 'w+') as handle:
                json.dump(params, handle)

    def log(self, content):
        print(f'=> Log: {content}')
        if self.gpu != 0: return
        with open(os.path.join(self.log_folder, 'log'), 'a+') as fd:
            fd.write(content + '\n')
            fd.flush()

    @param('dist.distributed')
    def _save_results(self, distributed):
        if distributed:
            dist.barrier()
        results = {
            'syops_count': self.syops_count,
            'params_count': self.params_count,
            'total_energy': self.total_energy,
            'energy_string': self.energy_string,
            'syops_string': self.syops_string,
            'ac_ops_string': self.ac_ops_string,
            'mac_ops_string': self.mac_ops_string,
            'params_string': self.params_string,
            'max_accuracy': self.max_accuracy,
        }
        if self.gpu==0:
            with open(os.path.join(self.log_folder, 'results.json'), 'w+') as handle:
                json.dump(results, handle)

    @param('dist.distributed')
    def _save_checkpoint(self,path,epoch,distributed):
        if distributed:
            dist.barrier()
            model = self.model.module
        else:
            model = self.model
        checkpoint = {
            "model": model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "scaler": self.scaler.state_dict(),
            "epoch": epoch,
            "max_accuracy": self.max_accuracy
        }
        if self.gpu==0:
            ch.save(checkpoint,path)
            self.log(f"Saved checkpoint to: {path}")

    def save_latest(self,epoch):
        path = os.path.join(self.pt_folder,"latest_checkpoint.pt")
        self._save_checkpoint(path,epoch)

    def save_checkpoint(self,epoch, is_best=False):
        if is_best:
            path = os.path.join(self.pt_folder,"best_checkpoint.pt")
        else:
            path = os.path.join(self.checkpoint_folder,f"checkpoint_{epoch}.pt")
        self._save_checkpoint(path,epoch)


    @param('model.resume')
    def get_checkpoint_path(self,resume):
        file_name = os.path.splitext(os.path.basename(resume))[0]
        if resume == 'latest':
            path = os.path.join(self.pt_folder,'latest_checkpoint.pt')
        elif resume == 'best':
            path = os.path.join(self.pt_folder,'best_checkpoint.pt')
        elif os.path.exists(resume):
            path = resume
        elif os.path.exists(os.path.join(self.pt_folder,f'{file_name}.pt')):
            path = os.path.join(self.pt_folder,f'{file_name}.pt')
        elif os.path.exists(os.path.join(self.checkpoint_folder,f'{file_name}.pt')):
            path = os.path.join(self.checkpoint_folder,f'{file_name}.pt')
        else:
            raise FileNotFoundError(f"Cannot find given reserved word, path, or file: {resume}")
        return path

    @param('model.resume')
    @param('dist.distributed')
    def load_checkpoint(self,resume,distributed):
        print(f"Resuming from {resume}")
        if resume.startswith('https'):
            checkpoint = ch.hub.load_state_dict_from_url(
                resume,map_location='cpu',check_hash=True)
        else:
            path = self.get_checkpoint_path()
            checkpoint = ch.load(path,map_location='cpu')
        if distributed:
            msg = self.model.module.load_state_dict(checkpoint['model'],strict=False)
        else:
            msg = self.model.load_state_dict(checkpoint['model'],strict=False)
        print(msg)
        self.start_epoch = checkpoint["epoch"] + 1
        self.max_accuracy=checkpoint['max_accuracy']
        print(f"Successfully loaded '{resume}' (epoch {checkpoint['epoch']})")
        del checkpoint

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
    @param('validation.eval_only')
    def exec(cls, gpu, distributed, eval_only):
        trainer = cls(gpu=gpu)
        if eval_only:
            trainer.eval_and_log()
        else:
            trainer.train()

        if distributed:
            trainer.cleanup_distributed()

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
    Trainer.launch_from_args()

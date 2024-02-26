import datetime
import torch as ch
from torch.cuda.amp import GradScaler
from torch.cuda.amp import autocast
import torch.distributed as dist
from torch.utils.tensorboard import SummaryWriter
import torchmetrics
from torchvision import transforms
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

from dst_repvgg import get_model_by_name

from spikingjelly.activation_based import neuron, functional, monitor
from spikingjelly.datasets import dvs128_gesture
from syops import get_model_complexity_info
from syops.utils import syops_to_string, params_to_string
from ops import MODULES_MAPPING
import connecting_neuron
import batchnorm_neuron
import autoaugment
from timm.data import Mixup

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
    conversion=Param(bool,'use bnplif',is_flag=True),
    conversion_set_y=Param(bool,'set y',is_flag=True),
)

Section('model').enable_if(lambda cfg: cfg['dist.distributed']==True).params(
    sync_bn = Param(bool,'enable batch norm syncing when in distributed mode',is_flag=True),
)

Section('data', 'data related stuff').params(
    path = Param(str,'path to dataset folder',required=True),
    T=Param(int,'T',default=16),
    num_workers=Param(int, 'The number of workers', default=4),
)

Section('lr','lr scheduling').params(
    lr=Param(float,'',default=0.1),
    gamma = Param(float,'lr decay factor',default=0.1),
    step_size = Param(int,'lr decay step size',default=64),
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
    sn_weight_decay=Param(float, 'weight decay for spiking neuron', default=0.0),
)

Section('training', 'training hyper param stuff').params(
    batch_size=Param(int, 'The training batch size', default=16),
    T_train = Param(int,'T_train',default=None),
    epochs=Param(int, 'number of epochs', default=192),
)

Section('dist', 'distributed training options').params(
    world_size=Param(int, 'number gpus', default=1),
    address=Param(str, 'address', default='localhost'),
    port=Param(str, 'port', default='12355'),
    distributed=Param(bool, 'use distributed mode', is_flag=True),
)

Section('augment', 'augmentation options').params(
    enable_augmentation=Param(bool, 'enable augmentation', is_flag=True),
    smoothing=Param(float, 'smoothing', default=0.1),
    mixup=Param(float, 'mixup', default=0.5),
    cutmix=Param(float, 'cutmix', default=0.0),
    cutmix_minmax=Param(float, 'cutmix_minmax', default=None),
    mixup_prob=Param(float, 'mixup_prob', default=0.5),
    mixup_switch_prob=Param(float, 'mixup_switch_prob', default=0.5),
    mixup_mode=Param(str, 'mixup_mode', default='batch'),
)

class Trainer:
    @param('dist.distributed')
    @param('model.resume')
    def __init__(self, gpu, distributed, resume=None):
        self.all_params = get_current_config()
        self.gpu = gpu
        self.num_classes = 11
        if distributed:
            self.setup_distributed()
        self.train_loader = self.create_train_loader()
        self.val_loader = self.create_val_loader()
        self.model, self.scaler = self.create_model_and_scaler()
        self.create_augmentation()
        self.start_epoch=0
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

    @param('data.path')
    @param('data.T')
    @param('data.num_workers')
    @param('training.batch_size')
    @param('dist.distributed')
    def create_train_loader(self, path, T, num_workers, batch_size,
                            distributed):
        dataset = dvs128_gesture.DVS128Gesture(root=path, train=True, data_type='frame',frames_number=T, split_by='number')
        if distributed:
            sampler = ch.utils.data.distributed.DistributedSampler(dataset)
        else:
            sampler = ch.utils.data.RandomSampler(dataset)
        loader = ch.utils.data.DataLoader(
            dataset, 
            batch_size=batch_size, 
            sampler=sampler,
            num_workers=num_workers,
            pin_memory=True, 
            drop_last=True
        )
        return loader

    @param('data.path')
    @param('data.T')
    @param('data.num_workers')
    @param('validation.batch_size')
    @param('dist.distributed')
    def create_val_loader(self, path, T, num_workers, batch_size, 
                          distributed):
        dataset = dvs128_gesture.DVS128Gesture(root=path, train=False, data_type='frame',frames_number=T, split_by='number')
        if distributed:
            sampler = ch.utils.data.distributed.DistributedSampler(dataset)
        else:
            sampler = ch.utils.data.SequentialSampler(dataset)
        loader = ch.utils.data.DataLoader(
            dataset, 
            batch_size=batch_size, 
            sampler=sampler,
            num_workers=num_workers,
            pin_memory=True, 
            drop_last=False
        )
        return loader

    @param('model.arch')
    @param('model.block_type')
    @param('model.cupy')
    @param('dist.distributed')
    @param('model.conversion')
    @param('model.conversion_set_y')
    @param('model.sync_bn')
    def create_model_and_scaler(self, arch, block_type, cupy, distributed, conversion=False, conversion_set_y = False, sync_bn=None):
        scaler = GradScaler()
        
        model = get_model_by_name(arch)(num_classes=self.num_classes,block_type=block_type,conversion=conversion,conversion_set_y=conversion_set_y)
        functional.set_step_mode(model,'m')
        if cupy:
            functional.set_backend(model,'cupy',instance=neuron.ParametricLIFNode)
            functional.set_backend(model,'cupy',instance=connecting_neuron.ParaConnLIFNode)
        model = model.to(self.gpu)

        if distributed:
            if sync_bn: model = ch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
            model = ch.nn.parallel.DistributedDataParallel(model, device_ids=[self.gpu])

        return model, scaler
    
    @param('augment.enable_augmentation')
    @param('augment.smoothing')
    @param('augment.mixup')
    @param('augment.cutmix')
    @param('augment.mixup_prob')
    @param('augment.mixup_switch_prob')
    @param('augment.mixup_mode')
    @param('augment.cutmix_minmax')
    def create_augmentation(self, enable_augmentation, smoothing, mixup, cutmix, mixup_prob, mixup_switch_prob, mixup_mode, cutmix_minmax=None):
        if enable_augmentation:
            self.train_snn_aug = transforms.Compose([
                transforms.RandomHorizontalFlip(p=0.5)
            ])
            self.train_trivalaug = autoaugment.SNNAugmentWide()
            self.mixup_fn = None
            mixup_active = mixup > 0 or cutmix > 0 or cutmix_minmax is not None
            if mixup_active:
                mixup_args = dict(
                    mixup_alpha = mixup, cutmix_alpha = cutmix, cutmix_minmax = cutmix_minmax,
                    prob = mixup_prob, switch_prob = mixup_switch_prob, mode = mixup_mode,
                    label_smoothing = smoothing, num_classes = self.num_classes)
                self.mixup_fn = Mixup(**mixup_args)
        else:
            self.train_snn_aug = None
            self.train_trivalaug = None
            self.mixup_fn = None
    
    @param('lr.lr')
    @param('optim.momentum')
    @param('optim.weight_decay')
    @param('optim.sn_weight_decay')
    def create_optimizer(self, lr, momentum, weight_decay, sn_weight_decay):
        # Only do weight decay on non-batchnorm parameters
        all_params = list(self.model.named_parameters())
        print(f"Total number of parameters: {len(all_params)}")
        bn_params = [v for k, v in all_params if ('bn' in k) or ('.bias' in k)]
        print(f"Number of batchnorm parameters: {len(bn_params)}")
        sn_params = [v for k, v in all_params if ('sn' in k)]
        print(f"Number of sn parameters: {len(sn_params)}")
        other_params = [v for k, v in all_params if not ('bn' in k) and not ('.bias' in k) and not ('sn' in k)]
        print(f"Number of non-batchnorm and non-sn parameters: {len(other_params)}")
        param_groups = [{
            'params': bn_params,
            'weight_decay': 0.
        },
        {
            'params': sn_params,
            'weight_decay': sn_weight_decay
        },         
        {
            'params': other_params,
            'weight_decay': weight_decay
        }]
        self.optimizer = ch.optim.SGD(param_groups, lr=lr, momentum=momentum)
        self.loss = ch.nn.CrossEntropyLoss()


    @param('lr.step_size')
    @param('lr.gamma')
    def create_scheduler(self,step_size,gamma):
        self.lr_scheduler = ch.optim.lr_scheduler.StepLR(self.optimizer,step_size=step_size,gamma=gamma,last_epoch=self.start_epoch-1)

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
                if self.max_accuracy == val_stats['top_1']:
                    self.save_checkpoint(epoch,is_best=True)

            self.lr_scheduler.step()
        train_time = time.time()-start_train
        train_time_str = str(datetime.timedelta(seconds=int(train_time)))
        self.log(f'Training time {train_time_str}')
        self.log(f'Max accuracy {self.max_accuracy}')  
        self.calculate_complexity()
        if hasattr(self.model,'switch_to_deploy'):
            self.model.switch_to_deploy()
        stats = self.val_loop()
        self.log(f"Reparameterized fps: {stats['thru']}")
        self._save_results(stats)
        self.calculate_spike_rates()

    def calculate_complexity(self):
        self.model.load_state_dict(ch.load(os.path.join(self.pt_folder,'best_checkpoint.pt'))['model'], strict=False)
        self.model.switch_to_deploy()
        ops, params = get_model_complexity_info(self.model, (2, 128, 128), self.val_loader, as_strings=False,
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

    def calculate_spike_rates(self):
        model=self.model
        model.eval()
        spike_seq_monitor = monitor.OutputMonitor(
            model, 
            (
                neuron.ParametricLIFNode,
                connecting_neuron.ParaConnLIFNode,
                batchnorm_neuron.BNPLIFNode
                ), 
            lambda x: x.mean().item())
        spike_rates = None

        cnt = 0
        with ch.no_grad():
            with autocast():
                for images, target in tqdm(self.val_loader):
                    images = images.to(self.gpu, non_blocking=True).float()
                    target = target.to(self.gpu, non_blocking=True)
                    output = self.model(images)
                    functional.reset_net(model)
                    if spike_rates is None:
                        spike_rates = spike_seq_monitor.records
                    else:
                        spike_rates = [spike_rates[i]+spike_seq_monitor.records[i] for i in range(len(spike_rates))]
                    cnt += 1
                    spike_seq_monitor.clear_recorded_data()
        spike_rates = [spike_rate/cnt for spike_rate in spike_rates]
        spike_seq_monitor.remove_hooks()
        self.log(f'Spike rates: {spike_rates}')
        if self.gpu==0:
            with open(os.path.join(self.log_folder, 'spike_rates.json'), 'w+') as handle:
                json.dump(spike_rates, handle)

    def eval_and_log(self):
        self.calculate_complexity()
        if hasattr(self.model,'switch_to_deploy'):
            self.model.switch_to_deploy()
        start_val = time.time()
        stats = self.val_loop()
        val_time = time.time() - start_val
        if self.gpu == 0:
            self.log(dict(
                stats,
                current_lr=self.optimizer.param_groups[0]['lr'],
                val_time=val_time
            ))
        self.calculate_spike_rates()
        self._save_results(stats)

    @param('training.T_train')
    def train_loop(self, T_train=None):
        model = self.model
        model.train()

        for images, target in tqdm(self.train_loader):
            start = time.time()
            self.optimizer.zero_grad(set_to_none=True)
            images = images.to(self.gpu, non_blocking=True).float()
            target = target.to(self.gpu, non_blocking=True)
            N,T,C,H,W = images.shape
            if self.train_snn_aug is not None:
                images = ch.stack([(self.train_snn_aug(images[i])) for i in range(N)])
            if self.train_trivalaug is not None:
                images = ch.stack([(self.train_trivalaug(images[i])) for i in range(N)])
            if self.mixup_fn is not None:
                images, target = self.mixup_fn(images, target)
                target_for_compu_acc = target.argmax(dim=-1)
            else:
                target_for_compu_acc = target
            if T_train:
                sec_list = np.random.choice(images.shape[1],T_train,replace=False)
                sec_list.sort()
                images = images[:,sec_list]

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
            target_for_compu_acc=target_for_compu_acc.detach()
            self.meters['top_1'].update(output,target_for_compu_acc)
            self.meters['top_5'].update(output,target_for_compu_acc)
            batch_size=target.shape[0]
            self.meters['thru'].update(ch.tensor(batch_size/(end-start)))
            self.meters['loss'].update(loss_train.detach())

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
                    output = self.model(images)
                    functional.reset_net(model)
                    end = time.time()
                    if type(output) is tuple:
                        output, aac = output
                    self.meters['top_1'].update(output, target)
                    self.meters['top_5'].update(output, target)
                    batch_size = target.shape[0]
                    self.meters['thru'].update(ch.tensor(batch_size/(end-start)))
                    loss_val = self.loss(output, target)
                    self.meters['loss'].update(loss_val)

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
            'top_1': torchmetrics.Accuracy(task='multiclass',num_classes=self.num_classes).to(self.gpu),
            'top_5': torchmetrics.Accuracy(task='multiclass',num_classes=self.num_classes, top_k=5).to(self.gpu),
            'thru': MeanScalarMetric().to(self.gpu),
            'loss': MeanScalarMetric().to(self.gpu)
        }

        if self.gpu == 0:
            folder = os.path.join(folder,'dvsgesture',arch,block_type,tag)
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
            'max_accuracy': self.max_accuracy,
            'thru': stats['thru'],
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

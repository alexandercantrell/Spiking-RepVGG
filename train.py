import datetime
import torch as ch
from torch.cuda.amp import GradScaler
from torch.cuda.amp import autocast
import torch.nn.functional as F
import torch.distributed as dist
from torch.utils.tensorboard import SummaryWriter
import torchmetrics
import numpy as np
from tqdm import tqdm

import shutil

import os
import time
import json
from typing import List
from argparse import ArgumentParser

from fastargs import get_current_config
from fastargs.decorators import param
from fastargs import Param, Section
from fastargs.validation import And, OneOf

from ffcv.pipeline.operation import Operation
from ffcv.loader import Loader, OrderOption
from ffcv.transforms import ToTensor, ToDevice, Squeeze, NormalizeImage, \
    RandomHorizontalFlip, ToTorchImage
from ffcv.fields.rgb_image import CenterCropRGBImageDecoder, \
    RandomResizedCropRGBImageDecoder
from ffcv.fields.basics import IntDecoder

from models.surrogate import FastATan
from models.spiking_repvgg import get_SpikingRepVGG_func_by_name
from models.hybrid_spiking_repvgg import get_HybridSpikingRepVGG_func_by_name
from models.static_spiking_repvgg import get_StaticSpikingRepVGG_func_by_name

from spikingjelly.activation_based import surrogate, neuron, functional

SEED=2020
ch.backends.cudnn.benchmark = True
ch.manual_seed(SEED)
ch.cuda.manual_seed_all(SEED)
np.random.seed(SEED)

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

Section('model', 'model details').params(
    arch=Param(str, 'model arch', required=True),
    fast_atan=Param(bool,'surrogate',is_flag=True),
    atan_alpha=Param(float,'atan alpha',default=2.0),
    zero_init=Param(bool,'initialize all weights to zero',is_flag=True),#TODO: add zero_init in model creation
    cnf = Param(str,'cnf',default='FAST_XOR'),
    cupy = Param(bool,'use cupy backend for neurons',is_flag=True),
    resume = Param(str,'checkpoint to load from',default=None)
)#TODO: add checkopinting to backprop

Section('model').enable_if(lambda cfg: cfg['dist.distributed']==True).params(
    sync_bn = Param(bool,'enable batch norm syncing when in distributed mode',is_flag=True),
)

Section('data', 'data related stuff').params(
    path = Param(str,'path to dataset folder',required=True),
    dataset=Param(str,'dataset name',required=True),
    T=Param(int,'T',default=4),
    num_workers=Param(int, 'The number of workers', default=16),
    in_memory=Param(bool, 'keep dataset in memory', is_flag=True)
)

Section('lr','lr scheduling').params(
    lr=Param(float,'',default=0.1),
    scheduler=Param(And(str,OneOf(['step','cosa'])),'',default='cosa'),
    warmup_epochs=Param(int,'',default=5)
)
Section('lr').enable_if(lambda cfg: cfg['lr.scheduler'] == 'step').params(
    gamma=Param(float,'',default=0.1),
    step_size=Param(int,'',default=30)
)
Section('lr').enable_if(lambda cfg: cfg['lr.scheduler'] == 'cosa').params(
    eta_min=Param(float,'',default=0.0)
)
Section('lr').enable_if(lambda cfg: cfg['lr.warmup_epochs']>0).params(
    warmup_method=Param(And(str,OneOf(['linear','constant'])),'',default='linear'),
    warmup_decay=Param(float,'',default=0.01)
)

Section('logging', 'how to log stuff').params(
    folder=Param(str, 'log location', default='./logs/'),
    tag=Param(str,'experiment tag',default='default'),
    clean=Param(bool,'clean prior experiment folder if exists',is_flag=True)
)

Section('validation', 'Validation parameters stuff').params(
    resize_size=Param(int,'Size to resize validation images to before cropping',default=256),
    crop_size=Param(int,'Size to crop valdiation images to',default=224),
    batch_size=Param(int, 'The validation batch size for validation', default=256),
    lr_tta=Param(bool, 'should do lr flipping/avging at test time', is_flag=True),
    eval_only=Param(bool,'only perform evaluation',is_flag=True)
)

Section('optim','optimizer hyper params').params(
    momentum=Param(float, 'SGD momentum', default=0.9),
    weight_decay=Param(float, 'weight decay', default=1e-4),
)

Section('criterion','criterion hyper params').params(
    label_smoothing=Param(float, 'label smoothing parameter', default=0.1),
)

Section('training', 'training hyper param stuff').params(
    crop_size=Param(int,'Size to crop training images to',default=176),
    batch_size=Param(int, 'The training batch size', default=256),
    epochs=Param(int, 'number of epochs', default=120),
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

        if distributed:
            self.setup_distributed()
        self.set_dataset_stats()
        self.train_loader = self.create_train_loader()
        self.val_loader = self.create_val_loader()
        self.model, self.scaler = self.create_model_and_scaler()
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

    @param('data.dataset')
    def set_dataset_stats(self,dataset):
        means, stds = DATASET_STATS[dataset]
        means = np.array(means)*255
        stds = np.array(stds)*255
        self.means=means
        self.stds = stds
        self.num_classes = DATASET_NUM_CLASSES[dataset]

    @param('data.path')
    @param('training.crop_size')
    @param('data.num_workers')
    @param('training.batch_size')
    @param('dist.distributed')
    @param('data.in_memory')
    def create_train_loader(self, path, crop_size, num_workers, batch_size,
                            distributed, in_memory):
        this_device = f'cuda:{self.gpu}'
        train_path = os.path.join(path,'train.ffcv')
        assert os.path.exists(train_path)
        image_pipeline: List[Operation] = [
            RandomResizedCropRGBImageDecoder((crop_size, crop_size)),
            RandomHorizontalFlip(),
            ToTensor(),
            ToDevice(ch.device(this_device), non_blocking=True),
            ToTorchImage(),
            NormalizeImage(self.means, self.stds, np.float16)
        ]

        label_pipeline: List[Operation] = [
            IntDecoder(),
            ToTensor(),
            Squeeze(),
            ToDevice(ch.device(this_device), non_blocking=True)
        ]

        order = OrderOption.RANDOM if distributed else OrderOption.QUASI_RANDOM
        loader = Loader(train_path,
                        batch_size=batch_size,
                        num_workers=num_workers,
                        order=order,
                        os_cache=in_memory,
                        drop_last=True,
                        pipelines={
                            'image': image_pipeline,
                            'label': label_pipeline
                        },
                        distributed=distributed)

        return loader

    @param('data.path')
    @param('validation.resize_size')
    @param('validation.crop_size')
    @param('data.num_workers')
    @param('validation.batch_size')
    @param('dist.distributed')
    @param('data.in_memory')
    def create_val_loader(self, path, resize_size, crop_size, num_workers, 
                          batch_size, distributed, in_memory):
        this_device = f'cuda:{self.gpu}'
        val_path = os.path.join(path,'val.ffcv')
        assert(os.path.exists(val_path))
        res_tuple = (crop_size, crop_size)
        ratio = float(crop_size)/float(resize_size)
        image_pipeline = [
            CenterCropRGBImageDecoder(res_tuple, ratio=ratio),
            ToTensor(),
            ToDevice(ch.device(this_device), non_blocking=True),
            ToTorchImage(),
            NormalizeImage(self.means, self.stds, np.float16)
        ]

        label_pipeline = [
            IntDecoder(),
            ToTensor(),
            Squeeze(),
            ToDevice(ch.device(this_device),
            non_blocking=True)
        ]

        loader = Loader(val_path,
                        batch_size=batch_size,
                        num_workers=num_workers,
                        order=OrderOption.SEQUENTIAL,
                        os_cache=in_memory,
                        drop_last=False,
                        pipelines={
                            'image': image_pipeline,
                            'label': label_pipeline
                        },
                        distributed=distributed)
        return loader

    @param('model.arch')
    @param('model.fast_atan')
    @param('model.atan_alpha')
    @param('model.cnf')
    @param('model.zero_init') #TODO: implement
    @param('model.cupy')
    @param('data.T')
    @param('dist.distributed')
    @param('model.sync_bn')
    def create_model_and_scaler(self, arch, fast_atan, atan_alpha, cnf, zero_init, cupy, T, distributed, sync_bn=None):
        scaler = GradScaler()
        surrogate_function = surrogate.ATan(alpha=atan_alpha)
        if fast_atan:
            surrogate_function = FastATan(alpha=atan_alpha/2.0)
        if 'StaticSpikingRepVGG' in arch:
            model = get_StaticSpikingRepVGG_func_by_name(arch)(num_classes=self.num_classes,deploy=False,use_checkpoint=False,
                            cnf=cnf,spiking_neuron=neuron.IFNode,surrogate_function=surrogate_function,detach_reset=True)
        elif 'HybridSpikingRepVGG' in arch:
            model = get_HybridSpikingRepVGG_func_by_name(arch)(num_classes=self.num_classes,deploy=False,use_checkpoint=False,
                            cnf=cnf,spiking_neuron=neuron.IFNode,surrogate_function=surrogate_function,detach_reset=True)
        elif 'SpikingRepVGG' in arch:
            model = get_SpikingRepVGG_func_by_name(arch)(num_classes=self.num_classes,deploy=False,use_checkpoint=False,
                            cnf=cnf,spiking_neuron=neuron.IFNode,surrogate_function=surrogate_function,detach_reset=True)
        else:
            raise ValueError(f"Model architecture {arch} does not exist!")
        
        if T>0:
            functional.set_step_mode(model,'m')
        else:
            functional.set_step_mode(model,'s')

        if cupy:
            functional.set_backend(model,'cupy',instance=neuron.IFNode)

        model = model.to(memory_format=ch.channels_last)
        model = model.to(self.gpu)

        if distributed:
            if sync_bn: model = ch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
            model = ch.nn.parallel.DistributedDataParallel(model, device_ids=[self.gpu])

        return model, scaler
    
    @param('lr.lr')
    @param('optim.momentum')
    @param('optim.weight_decay')
    @param('criterion.label_smoothing')
    def create_optimizer(self, lr, momentum, weight_decay,
                         label_smoothing):

        # Only do weight decay on non-batchnorm parameters
        all_params = list(self.model.named_parameters())
        print(f"Total number of parameters: {len(all_params)}")
        bn_params = [v for k, v in all_params if ('bn' in k) or ('.bias' in k)]
        print(f"Number of batchnorm parameters: {len(bn_params)}")
        other_params = [v for k, v in all_params if not ('bn' in k) and not ('.bias' in k)]
        print(f"Number of non-batchnorm parameters: {len(other_params)}")
        param_groups = [{
            'params': bn_params,
            'weight_decay': 0.
        }, {
            'params': other_params,
            'weight_decay': weight_decay
        }]

        self.optimizer = ch.optim.SGD(param_groups, lr=lr, momentum=momentum)
        self.loss = ch.nn.CrossEntropyLoss(label_smoothing=label_smoothing)

    @param('lr.scheduler')
    @param('training.epochs')
    @param('lr.warmup_epochs')
    @param('lr.step_size')
    @param('lr.gamma')
    @param('lr.eta_min')
    @param('lr.warmup_method')
    @param('lr.warmup_decay')
    def create_scheduler(self,scheduler,epochs,warmup_epochs,step_size=None,gamma=None,eta_min=None,warmup_method=None,warmup_decay=None):
        scheduler=scheduler.lower()
        if scheduler == "step":
            main_lr_scheduler = ch.optim.lr_scheduler.StepLR(self.optimizer,step_size=step_size,gamma=gamma,last_epoch=self.start_epoch-1)
        elif scheduler == 'cosa':
            main_lr_scheduler = ch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer,T_max=epochs-warmup_epochs,eta_min=eta_min,last_epoch=self.start_epoch-1)
        else:
            raise RuntimeError(
                f"Invalid lr scheduler '{scheduler}'. Only StepLR and CosineAnnealingLR are supported."
            )
        if warmup_epochs>0:
            warmup_method=warmup_method.lower()
            if warmup_method=='linear':
                warmup_lr_scheduler=ch.optim.lr_scheduler.LinearLR(
                    self.optimizer,start_factor=warmup_decay,total_iters=warmup_epochs
                )
            elif warmup_method=='constant':
                warmup_lr_scheduler=ch.optim.lr_scheduler.ConstantLR(
                    self.optimizer,factor=warmup_decay,total_iters=warmup_epochs
                )
            else:
                raise RuntimeError(
                    f"Invalid warmup lr method '{warmup_method}'. Only linear and constant are supported."
                )
            lr_scheduler = ch.optim.lr_scheduler.SequentialLR(
                self.optimizer, schedulers=[warmup_lr_scheduler, main_lr_scheduler], milestones=[warmup_epochs], last_epoch=self.start_epoch-1
            )
        else:
            lr_scheduler = main_lr_scheduler
        self.lr_scheduler = lr_scheduler

    @param('data.T')
    def preprocess(self,x,T):
        if T > 0:
            return x.unsqueeze(0).repeat(T,1,1,1,1)
        else:
            return x
        
    @param('data.T')
    def postprocess(self,y,T):
        if T > 0:
            return y.mean(0)
        else:
            return y

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
            with autocast():
                images = self.preprocess(images)
                output = self.model(images)
                output = self.postprocess(output)
                loss_train = self.loss(output, target)
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

    @param('validation.lr_tta')
    def val_loop(self, lr_tta):
        model = self.model
        model.eval()

        with ch.no_grad():
            with autocast():
                for images, target in tqdm(self.val_loader):
                    start = time.time()
                    images = self.preprocess(images)
                    output = self.model(images)
                    functional.reset_net(model)
                    if lr_tta:
                        output += self.model(ch.flip(images, dims=[-1]))
                    output = self.postprocess(output)
                    functional.reset_net(model)
                    end = time.time()

                    self.meters['top_1'](output, target)
                    self.meters['top_5'](output, target)
                    batch_size = target.shape[0]
                    if lr_tta: batch_size*=2
                    self.meters['thru'](ch.tensor(batch_size/(end-start)))
                    loss_val = self.loss(output, target)
                    self.meters['loss'](loss_val)

        stats = {k: m.compute().item() for k, m in self.meters.items()}
        [meter.reset() for meter in self.meters.values()]
        return stats

    @param('logging.folder')
    @param('logging.tag')
    @param('model.arch')
    @param('model.cnf')
    @param('logging.clean')
    def initialize_logger(self, folder, tag, arch, cnf,clean=None):
        self.meters = {
            'top_1': torchmetrics.Accuracy(task='multiclass',num_classes=self.num_classes,compute_on_step=False).to(self.gpu),
            'top_5': torchmetrics.Accuracy(task='multiclass',num_classes=self.num_classes,compute_on_step=False, top_k=5).to(self.gpu),
            'thru': MeanScalarMetric(compute_on_step=False).to(self.gpu),
            'loss': MeanScalarMetric(compute_on_step=False).to(self.gpu)
        }

        if self.gpu == 0:
            folder = os.path.join(folder,arch,f'{tag}_{cnf}')
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
        path = os.path.join(self.pt_folder,"latest_checkpoint.pth")
        self._save_checkpoint(path,epoch)

    def save_checkpoint(self,epoch, is_best=False):
        if is_best:
            path = os.path.join(self.pt_folder,"best_checkpoint.pth")
        else:
            path = os.path.join(self.checkpoint_folder,f"checkpoint_{epoch}.pth")
        self._save_checkpoint(path,epoch)


    @param('model.resume')
    def get_checkpoint_path(self,resume):
        file_name = os.path.splitext(os.path.basename(resume))[0]
        if resume == 'latest':
            path = os.path.join(self.pt_folder,'latest_checkpoint.pth')
        elif resume == 'best':
            path = os.path.join(self.pt_folder,'best_checkpoint.pth')
        elif os.path.exists(resume):
            path = resume
        elif os.path.exists(os.path.join(self.pt_folder,f'{file_name}.pth')):
            path = os.path.join(self.pt_folder,f'{file_name}.pth')
        elif os.path.exists(os.path.join(self.checkpoint_folder,f'{file_name}.pth')):
            path = os.path.join(self.checkpoint_folder,f'{file_name}.pth')
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

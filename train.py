import datetime
import json
import os
import time
import torch
from argparse import ArgumentParser
from fastargs import get_current_config
from fastargs.decorators import param
from fastargs import Param, Section
from tqdm.auto import tqdm
import torch.backends.cudnn as cudnn
from torch.utils.tensorboard import SummaryWriter
import torchmetrics

from spikingjelly.activation_based import functional

from train.utils import *

from train.logger import create_logger
from train.metrics import MetricLogger

from train import build_train_loader, build_val_loader, get_num_classes, build_model, build_optimizer, build_scheduler

Section('train','').params(
    epochs=Param(int,'',default=120),
)

Section('val','').params(
    test_flip=Param(bool,'',is_flag=True),
    eval_only=Param(bool,'',is_flag=True),
    throughput_only=Param(bool,'',is_flag=True),
)

Section('criterion','').params(
    label_smoothing=Param(float,'',default=0.1)
)

def main():
    data_loader_train = build_train_loader()
    data_loader_val = build_val_loader()
    
    model, model_without_ddp = build_model()

    logger.info(str(model))

    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"number of params: {n_parameters}")
    if hasattr(model_without_ddp, 'flops'):
        flops = model_without_ddp.flops()
        logger.info(f"number of GFLOPs: {flops / 1e9}")

    if config['val.throughput_only']:
        cudnn.benchmark = False #TODO: check diff
        load_weights(model_without_ddp)
        thru = throughput(model, data_loader_val)
        logger.info(f"Only throughput samples/sec: {thru:.2f}")
        return

    criterion = torch.nn.CrossEntropyLoss(label_smoothing=config['criterion.label_smoothing'])

    if config['val.eval_only']:
        cudnn.benchmark=False
        cudnn.deterministic=True
        load_weights(model_without_ddp)
        acc1, acc5, loss = validate(model, criterion, data_loader_val)
        logger.info(f"Only eval. top-1 acc, top-5 acc, loss: {acc1:.3f}, {acc5:.3f}, {loss:.5f}")
        return
    
    if config['model.disable_amp']:
        scaler = None
    else:
        scaler = torch.cuda.amp.GradScaler()


    optimizer = build_optimizer(model)

    lr_scheduler = build_scheduler(optimizer)

    max_accuracy = -1.0
        
    if config['model.resume'] is not None:
        start_epoch, max_accuracy = load_checkpoint(model_without_ddp, optimizer, lr_scheduler)

    #TODO: generate deploy file.

    tb_writer = SummaryWriter(get_tb_dir(),purge_step=start_epoch)

    logger.info("Start training")
    start_time = time.time()
    for epoch in range(start_epoch, config['train.epochs']):
        train_loss, train_acc1, train_acc5 = train_one_epoch(model,criterion,optimizer,data_loader_train,epoch,scaler=scaler)
        lr_scheduler.step()
        if is_main_process():
            save_latest(config,epoch,model_without_ddp, max_accuracy, optimizer, lr_scheduler, scaler=scaler)
            tb_writer.add_scalar('train_loss',train_loss,epoch)
            tb_writer.add_scalar('train_acc1',train_acc1,epoch)
            tb_writer.add_scalar('train_acc5',train_acc5,epoch)

        if epoch % config['output.save_freq'] == 0 or epoch >= (config['train.epochs'] - 10):
            val_loss, val_acc1, val_acc5 = validate(model,criterion,data_loader_val)
            logger.info(f"Accuracy of the network at epoch {epoch}: {val_acc1:.3f}%")
            max_accuracy = max(max_accuracy, val_acc1)
            logger.info(f'Max accuracy: {max_accuracy:.2f}%')
            if is_main_process():
                save_checkpoint(config,epoch,model_without_ddp, max_accuracy, optimizer, lr_scheduler, scaler=scaler)
                tb_writer.add_scalar('val_loss',val_loss,epoch)
                tb_writer.add_scalar('val_acc1',val_acc1,epoch)
                tb_writer.add_scalar('val_acc5',val_acc5,epoch)
                if max_accuracy == val_acc1:
                    save_checkpoint(config, epoch, model_without_ddp, max_accuracy, optimizer, lr_scheduler,
                                    is_best=True, scaler=scaler)
    
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    logger.info('Training time {}'.format(total_time_str))

@param('data.T')
def preprocess_sample(x:torch.Tensor,T : int):
    if T > 0:
        return x.unsqueeze(0).repeat(T,1,1,1,1)
    else:
        return x
    
@param('data.T')
def process_model_output(y:torch.Tensor,T: int): 
    if T > 0:
        return y.mean(0)
    else:
        return y
    
def train_one_epoch(model, criterion, optimizer, data_loader, epoch, scaler=None):
    model.train()
    metric_logger = MetricLogger(delimiter=" ")
    metric_logger.add_meter("acc1", torchmetrics.Accuracy(task='multiclass',num_classes=get_num_classes(),compute_on_step=False).to(torch.device(get_device_name())))
    metric_logger.add_meter("acc1", torchmetrics.Accuracy(task='multiclass',num_classes=get_num_classes(),compute_on_step=False,top_k=5).to(torch.device(get_device_name())))
    metric_logger.add_meter("loss", torchmetrics.MeanMetric(compute_on_step=False).to(torch.device(get_device_name())))
    metric_logger.add_meter("img/s", torchmetrics.MeanMetric(compute_on_step=False).to(torch.device(get_device_name())))

    for (samples, targets) in tqdm(data_loader):
        start_time = time.time()
        optimizer.zero_grad(set_to_none=True)
        with torch.cuda.amp.autocast(dtype=torch.float16, enabled = not config['model.disable_amp']):
            samples = preprocess_sample(samples)
            outputs = model(samples)
            outputs = process_model_output(outputs)
            loss = criterion(outputs, targets)
        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()
        functional.reset_net(model)
        
        metric_logger.meters["acc1"](outputs,targets)
        metric_logger.meters["acc5"](outputs,targets)
        metric_logger.meters["loss"](loss.item())
        metric_logger.meters["img/s"](targets.shape[0] / (time.time() - start_time))
    logger.info(f'Train Epoch [{epoch}/{config["train.epochs"]}]: {str(metric_logger)}')
    return metric_logger.compute('loss','acc1','acc5')

@torch.no_grad()
def validate(model,criterion,data_loader):
    model.eval()
    metric_logger = MetricLogger(delimiter="  ")
    metric_logger.add_meter("acc1", torchmetrics.Accuracy(task='multiclass',num_classes=get_num_classes(),compute_on_step=False).to(torch.device(get_device_name())))
    metric_logger.add_meter("acc1", torchmetrics.Accuracy(task='multiclass',num_classes=get_num_classes(),compute_on_step=False,top_k=5).to(torch.device(get_device_name())))
    metric_logger.add_meter("loss", torchmetrics.MeanMetric(compute_on_step=False).to(torch.device(get_device_name())))
    metric_logger.add_meter("img/s", torchmetrics.MeanMetric(compute_on_step=False).to(torch.device(get_device_name())))
    
    with torch.cuda.amp.autocast(dtype=torch.float16, enabled = not config['model.disable_amp']):
        for samples, targets in tqdm(data_loader):
            start_time = time.time()
            samples = preprocess_sample(samples)
            outputs = model(samples)
            functional.reset_net(model)
            if config['val.test_flip']:
                outputs += model(torch.flip(samples,dims=[-1]))
                functional.reset_net(model)
            outputs = process_model_output(outputs)
            loss = criterion(outputs, targets)
            batch_size = targets.shape[0]
            if config['val.test_flip']:batch_size*=2
            metric_logger.meters["acc1"](outputs,targets)
            metric_logger.meters["acc5"](outputs,targets)
            metric_logger.meters["loss"](loss.item())
            metric_logger.meters["img/s"](batch_size / (time.time() - start_time))

    logger.info(f'Test: {str(metric_logger)}')
    return metric_logger.compute('loss','acc1','acc5')

@torch.no_grad()
def throughput(model,data_loader):
    model.eval()
    with torch.cuda.amp.autocast(dtype=torch.float16, enabled = not config['model.disable_amp']):
        for samples, targets in data_loader:
            samples = preprocess_sample(samples)
            batch_size = targets.shape[0]
            for _ in range(50):
                model(samples)
                functional.reset_net(model)
            torch.cuda.synchronize()
            logger.info(f"throughput averaged with 30 times")
            tic1 = time.time()
            for _ in range(30):
                model(samples)
                functional.reset_net(model)
            torch.cuda.synchronize()
            tic2 = time.time()
            throughput = 30 * batch_size / (tic2-tic1)
            logger.info(f"batch_size {batch_size} throughput {throughput}")
            return

def make_config(quiet=False):
    config = get_current_config()
    parser = ArgumentParser(description="")
    config.augment_argparse(parser)
    config.collect_argparse_args(parser)
    config.validate(mode='stderr')
    if not quiet:
        config.summary()
    return config

if __name__ == '__main__':
    config=make_config()
    init_distributed_mode()
    make_output_tree()
    set_seeds()
    set_deterministic()
    logger = create_logger()
    #if is_main_process():   
    #    with open(os.path.join(get_tb_dir(),'command_args.txt'),'w') as command_args:
    #        json.dump(config.content, command_args, indent=4)
    #logger.info(json.dumps(config.content))
    main()

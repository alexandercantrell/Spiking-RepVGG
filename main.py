from datetime import datetime
import json
import os
import time
import torch
import torch.backends.cudnn as cudnn
from torch.utils.tensorboard import SummaryWriter
import warnings

from spikingjelly.activation_based import functional

from train.arguments import get_args_parser
from train.utils import set_seeds, set_deterministic, make_output_tree, init_distributed_mode, is_main_process, load_checkpoint, save_latest, save_checkpoint, reduce_across_processes
from train.logger import create_logger
from train import build_loaders, build_model, build_scheduler, build_optimizer
from train.ema import ExponentialMovingAverage
from train.metrics import MetricLogger, SmoothedValue, accuracy

def main(args):
    data_loader_train, data_loader_val = build_loaders(args)

    model = build_model(args)
    logger.info(str(model))
    
    if args.distributed and args.sync_bn:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)

    criterion = torch.nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)

    optimizer = build_optimizer(args, model)

    if args.disable_amp:
        scaler = None
    elif args.cupy:
        raise ValueError("Automatic mixed precision (AMP) training conflicts with the 'cupy' neuron backend. "
                        "Either disable AMP with --disable-amp, or remove the --cupy flag from your execution.")
    else:
        scaler = torch.cuda.amp.GradScaler()
    
    lr_scheduler = build_scheduler(args, optimizer)

    model_without_ddp = model
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        model_without_ddp = model.module

    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"number of params: {n_parameters}")
    if hasattr(model_without_ddp, 'flops'):
        flops = model_without_ddp.flops()
        logger.info(f"number of GFLOPs: {flops / 1e9}")

    max_accuracy = -1.0


    model_ema = None
    max_ema_accuracy = None
    if args.model_ema: #TODO: build ema function?
        adjust = args.world_size * args.batch_size * args.model_ema_steps / args.epochs
        alpha = 1.0 - args.model_ema_decay
        alpha = min(1.0,alpha*adjust)
        model_ema = ExponentialMovingAverage(model_without_ddp,device=torch.device(args.device), decay=1.0 - alpha)
        max_ema_accuracy = -1.0
        
    if args.resume is not None:
        max_accuracy, max_ema_accuracy = load_checkpoint(args, model_without_ddp, optimizer, lr_scheduler, logger, model_ema=model_ema)

    #TODO: throughput and eval for deploy.
    #TODO: generate deploy file.

    if args.throughput:
        torch.backends.cudnn.benchmark = False #TODO: check diff
        if model_ema:
            thru = throughput(args, model_ema, data_loader_val)
            logger.info(f"EMA Only throughput samples/sec: {thru:.2f}")
        else:
            thru = throughput(args, model, data_loader_val)
            logger.info(f"Only throughput samples/sec: {thru:.2f}")
        return

    if args.eval:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        if model_ema:
            acc1, acc5, loss = validate(args, model_ema, criterion, data_loader_val, is_ema=True)
            logger.info(f"EMA Only eval. top-1 acc, top-5 acc, loss: {acc1:.3f}, {acc5:.3f}, {loss:.5f}")
        else:
            acc1, acc5, loss = validate(args, model, criterion, data_loader_val)
            logger.info(f"Only eval. top-1 acc, top-5 acc, loss: {acc1:.3f}, {acc5:.3f}, {loss:.5f}")
        return
    
    logger.info("Start training")
    start_time = time.time()
    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            data_loader_train.sampler.set_epoch(epoch)
        train_loss, train_acc1, train_acc5 = train_one_epoch(args,model,criterion,optimizer,data_loader_train,epoch, 
                        model_ema=model_ema,scaler=scaler)
        lr_scheduler.step()
        if is_main_process():
            save_latest(args,epoch,model_without_ddp, max_accuracy, optimizer, lr_scheduler, logger, model_ema=model_ema, max_ema_accuracy=max_ema_accuracy, scaler=scaler)
            tb_writer.add_scalar('train_loss',train_loss,epoch)
            tb_writer.add_scalar('train_acc1',train_acc1,epoch)
            tb_writer.add_scalar('train_acc5',train_acc5,epoch)

        if epoch % args.save_freq == 0 or epoch >= (args.epochs - 10):
            val_loss, val_acc1, val_acc5 = validate(args,model,criterion,data_loader_val)
            logger.info(f"Accuracy of the network at epoch {epoch}: {val_acc1:.3f}%")
            max_accuracy = max(max_accuracy, val_acc1)
            logger.info(f'Max accuracy: {max_accuracy:.2f}%')
            if is_main_process():
                save_checkpoint(args,epoch,model_without_ddp, max_accuracy, optimizer, lr_scheduler, logger, model_ema=model_ema, max_ema_accuracy=max_ema_accuracy, scaler=scaler)
                tb_writer.add_scalar('val_loss',val_loss,epoch)
                tb_writer.add_scalar('val_acc1',val_acc1,epoch)
                tb_writer.add_scalar('val_acc5',val_acc5,epoch)
                if max_accuracy == val_acc1:
                    save_checkpoint(args, epoch, model_without_ddp, max_accuracy, optimizer, lr_scheduler, logger,
                                    is_best=True, model_ema=model_ema, max_ema_accuracy=max_ema_accuracy, scaler=scaler)
            if model_ema:
                ema_val_loss, ema_val_acc1, ema_val_acc5 = validate(args,model_ema,criterion,data_loader_val,is_ema=True)
                logger.info(f"EMA Accuracy of the network at epoch {epoch} test images: {ema_val_acc1:.3f}%")
                max_ema_accuracy = max(max_ema_accuracy, ema_val_acc1)
                logger.info(f'EMA Max accuracy: {max_ema_accuracy:.2f}%')
                if is_main_process():
                    tb_writer.add_scalar('ema_val_loss',ema_val_loss,epoch)
                    tb_writer.add_scalar('ema_val_acc1',ema_val_acc1,epoch)
                    tb_writer.add_scalar('ema_val_acc5',ema_val_acc5,epoch)
                    if max_ema_accuracy == ema_val_acc1:
                        save_checkpoint(args, epoch, model_without_ddp, max_accuracy, optimizer, lr_scheduler, logger,
                                    is_best=True, is_ema=True, model_ema=model_ema, max_ema_accuracy=max_ema_accuracy, scaler=scaler)
    
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    logger.info('Training time {}'.format(total_time_str))

def preprocess_sample(T : int, x:torch.Tensor):
    if T > 0:
        return x.unsqueeze(0).repeat(T,1,1,1,1)
    else:
        return x

def process_model_output(T: int, y:torch.Tensor): 
    #to consider: could check len(y.size()) instead of passing T
    if T > 0:
        return y.mean(0)
    else:
        return y
    
def train_one_epoch(args, model, criterion, optimizer, data_loader, epoch, model_ema=None, scaler=None):
    model.train()
    metric_logger = MetricLogger(delimiter=" ")
    metric_logger.add_meter("lr", SmoothedValue(window_size=1, fmt="{value:.4f}"))
    metric_logger.add_meter("img/s", SmoothedValue(window_size=10, fmt="{value:.2f}"))

    header = f"Epoch: [{epoch}/{args.epochs}]"
    for idx, (samples, targets) in enumerate(metric_logger.log_every(data_loader,args.print_freq, header,logger=logger)):
        start_time = time.time()

        with torch.cuda.amp.autocast(dtype=torch.float16, enabled=scaler is not None):
            samples = preprocess_sample(args.T,samples)
            outputs = process_model_output(args.T,model(samples))
            loss = criterion(outputs, targets)

        optimizer.zero_grad(set_to_none=True)
        if scaler is not None:
            scaler.scale(loss).backward()
            if args.clip_grad_norm is not None:
                #we should unscale the gradients of optimizer's assigned params if performing gradient clipping
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(),args.clip_grad_norm)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            if args.clip_grad_norm is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad_norm)
            optimizer.step()
        functional.reset_net(model)
        if model_ema:                       #FIXME: unsure if needed
            functional.reset_net(model_ema)

        if model_ema and idx % args.model_ema_steps == 0:
            model_ema.update_parameters(model)
            if epoch < args.lr_warmup_epochs:
                model.ema.n_averaged.fill_(0)
        
        acc1, acc5 = accuracy(outputs,targets,topk=(1,5))
        batch_size = targets.shape[0]
        metric_logger.update(loss=loss.item(), lr=optimizer.param_groups[0]["lr"])
        metric_logger.meters["acc1"].update(acc1.item(), n=batch_size)
        metric_logger.meters["acc5"].update(acc5.item(), n=batch_size)
        metric_logger.meters["img/s"].update(batch_size / (time.time() - start_time))
    metric_logger.synchronize_between_processes()
    train_loss, train_acc1, train_acc5 = metric_logger.loss.global_avg, metric_logger.acc1.global_avg, metric_logger.acc5.global_avg
    logger.info(f'Train: train_acc1={train_acc1:.3f}, train_acc5={train_acc5:.3f}, train_loss={train_loss:.6f}, samples/s={metric_logger.meters["img/s"]}')
    return train_loss, train_acc1, train_acc5

@torch.no_grad()
def validate(args,model,criterion,data_loader,is_ema=False,print_freq=100):
    model.eval()
    metric_logger = MetricLogger(delimiter="  ")
    header = f"Validation: {'EMA' if is_ema else ''}"
    
    num_processed_samples = 0
    start_time = time.time()
    for samples, targets in metric_logger.log_every(data_loader, print_freq, header, logger=logger):
        with torch.cuda.amp.autocast(dtype=torch.float16, enabled=scaler is not None):
            samples = preprocess_sample(args.T,samples)
            outputs = process_model_output(args.T,model(samples))
            loss = criterion(outputs, targets)
        functional.reset_net(model)
        
        acc1, acc5 = accuracy(outputs,targets,topk=(1,5))
        # FIXME need to take into account that the datasets
        # could have been padded in distributed setup
        batch_size = targets.shape[0]
        metric_logger.update(loss=loss.item())
        metric_logger.meters["acc1"].update(acc1.item(),n=batch_size)
        metric_logger.meters["acc5"].update(acc5.item(), n=batch_size)
        num_processed_samples += batch_size
    num_processed_samples = reduce_across_processes(num_processed_samples)
    if (
        hasattr(data_loader.dataset, "__len__")
        and len(data_loader.dataset) != num_processed_samples
        and torch.distributed.get_rank() == 0
    ):
        # See FIXME above
        warnings.warn(
            f"It looks like the dataset has {len(data_loader.dataset)} samples, but {num_processed_samples} "
            "samples were used for the validation, which might bias the results. "
            "Try adjusting the batch size and / or the world size. "
            "Setting the world size to 1 is always a safe bet."
        )

    metric_logger.synchronize_between_processes()

    val_loss, val_acc1, val_acc5 = metric_logger.loss.global_avg, metric_logger.acc1.global_avg, metric_logger.acc5.global_avg
    logger.info(f'Test: val_acc1={val_acc1:.3f}, val_acc5={val_acc5:.3f}, val_loss={val_loss:.6f}, samples/s={num_processed_samples / (time.time() - start_time):.3f}')
    return val_loss, val_acc1, val_acc5

@torch.no_grad()
def throughput(args,model,data_loader):
    model.eval()
    for samples, targets in data_loader:
        samples = preprocess_sample(args.T,samples)
        batch_size = targets.shape[0]
        for _ in range(50):
            model(samples)
            if args.throughput_reset:
                functional.reset_net(model)
        torch.cuda.synchronize()
        logger.info(f"throughput averaged with 30 times")
        tic1 = time.time()
        for _ in range(30):
            model(samples)
            if args.throughput_reset:
                functional.reset_net(model)
        torch.cuda.synchronize()
        tic2 = time.time()
        throughput = 30 * batch_size / (tic2-tic1)
        logger.info(f"batch_size {batch_size} throughput {throughput}")
        return

if __name__ == '__main__':
    args = get_args_parser().parse_args()
    init_distributed_mode(args)
    make_output_tree(args)
    set_seeds(args.seed)
    if args.deterministic:
        cudnn.benchmark=False
        set_deterministic()
    else:
        cudnn.benchmark = True
    logger = create_logger(output_dir=args.output,dist_rank=0 if not args.distributed else args.rank, name=f"{args.arch}")
    if is_main_process():
        tb_writer = SummaryWriter(args.tb,purge_step=args.start_epoch)
        with open(os.path.join(args.tb,'command_args.txt'),'w') as command_args:
            json.dump(args.__dict__, command_args, indent=4)
    logger.info(json.dumps(args.__dict__))
    main(args)

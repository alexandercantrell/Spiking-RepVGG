# --------------------------------------------------------
# RepVGG: Making VGG-style ConvNets Great Again (https://openaccess.thecvf.com/content/CVPR2021/papers/Ding_RepVGG_Making_VGG-Style_ConvNets_Great_Again_CVPR_2021_paper.pdf)
# Github source: https://github.com/DingXiaoH/RepVGG
# Licensed under The MIT License [see LICENSE for details]
# The training script is based on the code of Swin Transformer (https://github.com/microsoft/Swin-Transformer)
# --------------------------------------------------------
import time
import argparse
import datetime
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.distributed as dist
from timm.loss import LabelSmoothingCrossEntropy, SoftTargetCrossEntropy
from timm.utils import accuracy, AverageMeter
import copy

from train.config import get_config
from data import build_loader
from train.lr_scheduler import build_scheduler
from train.logger import create_logger
from utils import load_checkpoint, save_checkpoint, get_grad_norm, auto_resume_helper, reduce_tensor, save_latest, update_model_ema, unwrap_model, load_weights
from train.optimizer import build_optimizer

from static_spiking_repvgg import get_StaticSpikingRepVGG_func_by_name
from hybrid_spiking_repvgg import get_HybridSpikingRepVGG_func_by_name
from spiking_repvgg import get_SpikingRepVGG_func_by_name
from spikingjelly.activation_based import functional, neuron


def parse_option():
    parser = argparse.ArgumentParser('RepOpt-VGG training script built on the codebase of Swin Transformer', add_help=False)
    parser.add_argument(
        "--opts",
        help="Modify config options by adding 'KEY VALUE' pairs. ",
        default=None,
        nargs='+',
    )

    # easy config modification
    parser.add_argument('--arch', default=None, type=str, help='arch name')
    parser.add_argument('--batch-size', default=128, type=int, help="batch size for single GPU")
    parser.add_argument('--data-path', default='/your/path/to/dataset', type=str, help='path to dataset')
    parser.add_argument('--scales-path', default=None, type=str, help='path to the trained Hyper-Search model')
    parser.add_argument('--zip', action='store_true', help='use zipped dataset instead of folder dataset')
    parser.add_argument('--cache-mode', type=str, default='part', choices=['no', 'full', 'part'],
                        help='no: no cache, '
                             'full: cache all data, '
                             'part: sharding the dataset into nonoverlapping pieces and only cache one piece')
    parser.add_argument('--resume', help='resume from checkpoint')
    parser.add_argument('--accumulation-steps', type=int, help="gradient accumulation steps")
    parser.add_argument('--use-checkpoint', action='store_true',
                        help="whether to use gradient checkpointing to save memory")
    parser.add_argument('--output', default='./logs/', type=str, metavar='PATH',
                        help='root of output folder, the full path is <output>/<model_name>/<tag> (default: output)')
    parser.add_argument('--tag', help='tag of experiment')
    parser.add_argument('--eval', action='store_true', help='Perform evaluation only')
    parser.add_argument('--throughput', action='store_true', help='Test throughput only')
    parser.add_argument('--amp-opt-level', type=str, default='O0', choices=['O0', 'O1', 'O2'],
                        help='depreciated, use --disable-amp if you want to turn off amp')

    # distributed training
    parser.add_argument("--local_rank", type=int, default=0, help='local rank for DistributedDataParallel')

    parser.add_argument("--cupy",action="store_true", help="set the neurons to use cupy backend")
    parser.add_argument("--disable-amp",action="store_true", help="disable AMP") #TODO: always use
    parser.add_argument("--T",type=int,default=4,help='number of times to repeat image')
    parser.add_argument("--save-freq",type=int,default=20,help='save frequency')
    parser.add_argument("--cnf",type=str,default="ADD")

    args, unparsed = parser.parse_known_args()

    config = get_config(args)

    return args, config





def main(config,device_id):
    dataset_train, dataset_val, data_loader_train, data_loader_val, mixup_fn = build_loader(config)

    logger.info(f"Creating model:{config.MODEL.ARCH}")
    arch = config.MODEL.ARCH
    if 'StaticSpikingRepVGG' in arch:
        model = get_StaticSpikingRepVGG_func_by_name(arch)(deploy=False,use_checkpoint=args.use_checkpoint,cnf=config.MODEL.CNF)
    elif 'HybridSpikingRepVGG' in arch:
        model = get_HybridSpikingRepVGG_func_by_name(arch)(deploy=False,use_checkpoint=args.use_checkpoint,cnf=config.MODEL.CNF)
    elif 'SpikingRepVGG' in arch:
        model = get_SpikingRepVGG_func_by_name(arch)(deploy=False,use_checkpoint=args.use_checkpoint,cnf=config.MODEL.CNF)
    else:
        raise NotImplementedError
    if config.DATA.T > 0:
        functional.set_step_mode(model,'m')
    else:
        functional.set_step_mode(model,'s')
    if config.MODEL.CUPY:
        functional.set_backend(model,'cupy',neuron.IFNode)

    optimizer = build_optimizer(config, model)

    logger.info(str(model))
    model.cuda()

    if torch.cuda.device_count() > 1:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[device_id], broadcast_buffers=False)
        model_without_ddp = model.module
    else:
        model_without_ddp = model

    if config.AMP:
        scaler = torch.cuda.amp.GradScaler()
    else:
        scaler = None

    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"number of params: {n_parameters}")
    if hasattr(model_without_ddp, 'flops'):
        flops = model_without_ddp.flops()
        logger.info(f"number of GFLOPs: {flops / 1e9}")

    if config.THROUGHPUT_MODE:
        throughput(data_loader_val, model, logger)
        return

    if config.EVAL_MODE:
        load_weights(model, config.MODEL.RESUME)
        acc1, acc5, loss = validate(config, data_loader_val, model)
        logger.info(f"Only eval. top-1 acc, top-5 acc, loss: {acc1:.3f}, {acc5:.3f}, {loss:.5f}")
        return

    lr_scheduler = build_scheduler(config, optimizer, len(data_loader_train))

    if config.AUG.MIXUP > 0.:
        # smoothing is handled with mixup label transform
        criterion = SoftTargetCrossEntropy()
    elif config.MODEL.LABEL_SMOOTHING > 0.:
        criterion = LabelSmoothingCrossEntropy(smoothing=config.MODEL.LABEL_SMOOTHING)
    else:
        criterion = torch.nn.CrossEntropyLoss()

    max_accuracy = 0.0
    max_ema_accuracy = 0.0

    if config.TRAIN.EMA_ALPHA > 0 and (not config.EVAL_MODE) and (not config.THROUGHPUT_MODE):
        model_ema = copy.deepcopy(model)
    else:
        model_ema = None

    if config.TRAIN.AUTO_RESUME:
        resume_file = auto_resume_helper(config.OUTPUT)
        if resume_file:
            if config.MODEL.RESUME:
                logger.warning(f"auto-resume changing resume file from {config.MODEL.RESUME} to {resume_file}")
            config.defrost()
            config.MODEL.RESUME = resume_file
            config.freeze()
            logger.info(f'auto resuming from {resume_file}')
        else:
            logger.info(f'no checkpoint found in {config.OUTPUT}, ignoring auto resume')

    if (not config.THROUGHPUT_MODE) and config.MODEL.RESUME:
        max_accuracy = load_checkpoint(config, model_without_ddp, optimizer, lr_scheduler, logger, model_ema=model_ema)


    logger.info("Start training")
    start_time = time.time()
    for epoch in range(config.TRAIN.START_EPOCH, config.TRAIN.EPOCHS):
        data_loader_train.sampler.set_epoch(epoch)

        train_one_epoch(config, model, criterion, data_loader_train, optimizer, epoch, mixup_fn, lr_scheduler, model_ema=model_ema, scaler=scaler)
        if dist.get_rank() == 0:
            save_latest(config, epoch, model_without_ddp, max_accuracy, optimizer, lr_scheduler, logger, model_ema=model_ema, scaler=scaler)
            if epoch % config.SAVE_FREQ == 0:
                save_checkpoint(config, epoch, model_without_ddp, max_accuracy, optimizer, lr_scheduler, logger, model_ema=model_ema, scaler=sclaer)

        if epoch % config.SAVE_FREQ == 0 or epoch >= (config.TRAIN.EPOCHS - 10):

            if data_loader_val is not None:
                acc1, acc5, loss = validate(config, data_loader_val, model)
                logger.info(f"Accuracy of the network at epoch {epoch}: {acc1:.3f}%")
                max_accuracy = max(max_accuracy, acc1)
                logger.info(f'Max accuracy: {max_accuracy:.2f}%')
                if max_accuracy == acc1 and dist.get_rank() == 0:
                    save_checkpoint(config, epoch, model_without_ddp, max_accuracy, optimizer, lr_scheduler, logger,
                                    is_best=True, model_ema=model_ema, scaler=scaler)

            if model_ema is not None:
                if data_loader_val is not None:
                    acc1, acc5, loss = validate(config, data_loader_val, model_ema)
                    logger.info(f"EMAAccuracy of the network at epoch {epoch} test images: {acc1:.3f}%")
                    max_ema_accuracy = max(max_ema_accuracy, acc1)
                    logger.info(f'EMAMax accuracy: {max_ema_accuracy:.2f}%')
                    if max_ema_accuracy == acc1 and dist.get_rank() == 0:
                        best_ema_path = os.path.join(config.OUTPUT, 'best_ema.pth')
                        logger.info(f"{best_ema_path} best EMA saving......")
                        torch.save(unwrap_model(model_ema).state_dict(), best_ema_path)
                else:
                    latest_ema_path = os.path.join(config.OUTPUT, 'latest_ema.pth')
                    logger.info(f"{latest_ema_path} latest EMA saving......")
                    torch.save(unwrap_model(model_ema).state_dict(), latest_ema_path)

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    logger.info('Training time {}'.format(total_time_str))

def preprocess_sample(config, x:torch.Tensor):
    if config.DATA.T > 0:
        return x.unsqueeze(0).repeat(config.DATA.T,1,1,1,1)
    else:
        return x
    
def process_model_output(config, y:torch.Tensor):
    if config.DATA.T > 0:
        return y.mean(0)
    else:
        return y

def train_one_epoch(config, model, criterion, data_loader, optimizer, epoch, mixup_fn, lr_scheduler, model_ema=None, scaler=None):
    model.train()
    optimizer.zero_grad()

    num_steps = len(data_loader)
    batch_time = AverageMeter()
    loss_meter = AverageMeter()
    norm_meter = AverageMeter()
    acc5_meter = AverageMeter()

    start = time.time()
    end = time.time()
    for idx, (samples, targets) in enumerate(data_loader):
        samples = samples.cuda(non_blocking=True)
        targets = targets.cuda(non_blocking=True)

        if mixup_fn is not None:
            samples, targets = mixup_fn(samples, targets)

        with torch.cuda.amp.autocast(enabled=scaler is not None, dtype=torch.float16):
            samples = preprocess_sample(config,samples)
            outputs = process_model_output(config,model(samples))
            loss = criterion(outputs, targets)
        
        if config.TRAIN.ACCUMULATION_STEPS > 1:
            loss = loss / config.TRAIN.ACCUMULATION_STEPS
            if scaler is not None:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                if config.TRAIN.CLIP_GRAD:
                    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(),config.TRAIN.CLIP_GRAD)
                else:
                    grad_norm = get_grad_norm(model.parameters())
                if (idx + 1) % config.TRAIN.ACCUMULATION_STEPS == 0:
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad()
                    lr_scheduler.step_update(epoch * num_steps + idx)
            else:
                loss.backward()
                if config.TRAIN.CLIP_GRAD:
                    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), config.TRAIN.CLIP_GRAD)
                else:
                    grad_norm = get_grad_norm(model.parameters())
                if (idx + 1) % config.TRAIN.ACCUMULATION_STEPS == 0:
                    optimizer.step()
                    optimizer.zero_grad()
                    lr_scheduler.step_update(epoch * num_steps + idx)

        else:
            optimizer.zero_grad()
            if scaler is not None:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                if config.TRAIN.CLIP_GRAD:
                    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), config.TRAIN.CLIP_GRAD)
                else:
                    grad_norm = get_grad_norm(model.parameters())
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                if config.TRAIN.CLIP_GRAD:
                    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), config.TRAIN.CLIP_GRAD)
                else:
                    grad_norm = get_grad_norm(model.parameters())
                optimizer.step()
            lr_scheduler.step_update(epoch * num_steps + idx)

        torch.cuda.synchronize()

        #acc5 = accuracy(outputs, targets, topk=(5,))
        #acc5 = reduce_tensor(acc5)
        loss_meter.update(loss.item(), targets.size(0))
        norm_meter.update(grad_norm)
        batch_time.update(time.time() - end)
        #acc5_meter.update(acc5.item(), targets.size(0))
        
        

        if model_ema is not None:
            update_model_ema(config, dist.get_world_size(), model=model, model_ema=model_ema, cur_epoch=epoch, cur_iter=idx)

        functional.reset_net(model)

        end = time.time()

        if idx % config.PRINT_FREQ == 0:
            lr = optimizer.param_groups[0]['lr']
            memory_used = torch.cuda.max_memory_allocated() / (1024.0 * 1024.0)
            etas = batch_time.avg * (num_steps - idx)
            logger.info(
                f'Train: [{epoch}/{config.TRAIN.EPOCHS}][{idx}/{num_steps}]\t'
                f'ETA {datetime.timedelta(seconds=int(etas))} lr {lr:.6f}\t'
                f'Time {batch_time.val:.4f} ({batch_time.avg:.4f})\t'
                f'Loss {loss_meter.val:.4f} ({loss_meter.avg:.4f})\t'
                f'Grad Norm {norm_meter.val:.4f} ({norm_meter.avg:.4f})\t'
                #f'Acc@5 {acc5_meter.val:.3f}\t'
                f'Mem {memory_used:.0f}MB')
    epoch_time = time.time() - start
    logger.info(f"EPOCH {epoch} training takes {datetime.timedelta(seconds=int(epoch_time))}")


@torch.no_grad()
def validate(config, data_loader, model):
    criterion = torch.nn.CrossEntropyLoss()
    model.eval()

    batch_time = AverageMeter()
    loss_meter = AverageMeter()
    acc1_meter = AverageMeter()
    acc5_meter = AverageMeter()

    end = time.time()
    for idx, (images, target) in enumerate(data_loader):
        images = images.cuda(non_blocking=True)
        target = target.cuda(non_blocking=True)

        # compute output
        output = model(preprocess_sample(config,images))

        #   =============================== deepsup part
        if type(output) is dict:
            output = output['main']

        output = process_model_output(config,output)

        # measure accuracy and record loss
        loss = criterion(output, target)
        acc1, acc5 = accuracy(output, target, topk=(1, 5))

        acc1 = reduce_tensor(acc1)
        acc5 = reduce_tensor(acc5)
        loss = reduce_tensor(loss)

        loss_meter.update(loss.item(), target.size(0))
        acc1_meter.update(acc1.item(), target.size(0))
        acc5_meter.update(acc5.item(), target.size(0))

        functional.reset_net(model)

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if idx % config.PRINT_FREQ == 0:
            memory_used = torch.cuda.max_memory_allocated() / (1024.0 * 1024.0)
            logger.info(
                f'Test: [{idx}/{len(data_loader)}]\t'
                f'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                f'Loss {loss_meter.val:.4f} ({loss_meter.avg:.4f})\t'
                f'Acc@1 {acc1_meter.val:.3f} ({acc1_meter.avg:.3f})\t'
                f'Acc@5 {acc5_meter.val:.3f} ({acc5_meter.avg:.3f})\t'
                f'Mem {memory_used:.0f}MB')
    logger.info(f' * Acc@1 {acc1_meter.avg:.3f} Acc@5 {acc5_meter.avg:.3f}')
    return acc1_meter.avg, acc5_meter.avg, loss_meter.avg


@torch.no_grad()
def throughput(data_loader, model, logger):
    model.eval()

    for idx, (images, _) in enumerate(data_loader):
        images = images.cuda(non_blocking=True)

        batch_size = images.shape[0]
        for i in range(50):
            model(images)
        torch.cuda.synchronize()
        logger.info(f"throughput averaged with 30 times")
        tic1 = time.time()
        for i in range(30):
            model(images)
        torch.cuda.synchronize()
        tic2 = time.time()
        throughput = 30 * batch_size / (tic2 - tic1)
        logger.info(f"batch_size {batch_size} throughput {throughput}")
        return


import os

if __name__ == '__main__':
    args, config = parse_option()
    dist.init_process_group("nccl")
    rank = dist.get_rank()
    device_id = rank % torch.cuda.device_count()
    torch.cuda.set_device(device_id)
    dist.barrier()
    
    seed = config.SEED + rank
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed_all(seed)
    cudnn.benchmark = True

    if not config.EVAL_MODE:
        # linear scale the learning rate according to total batch size, may not be optimal
        linear_scaled_lr = config.TRAIN.BASE_LR * config.DATA.BATCH_SIZE * dist.get_world_size() / 256.0
        linear_scaled_warmup_lr = config.TRAIN.WARMUP_LR * config.DATA.BATCH_SIZE * dist.get_world_size() / 256.0
        linear_scaled_min_lr = config.TRAIN.MIN_LR * config.DATA.BATCH_SIZE * dist.get_world_size() / 256.0
        # gradient accumulation also need to scale the learning rate
        if config.TRAIN.ACCUMULATION_STEPS > 1:
            linear_scaled_lr = linear_scaled_lr * config.TRAIN.ACCUMULATION_STEPS
            linear_scaled_warmup_lr = linear_scaled_warmup_lr * config.TRAIN.ACCUMULATION_STEPS
            linear_scaled_min_lr = linear_scaled_min_lr * config.TRAIN.ACCUMULATION_STEPS
        config.defrost()
        config.TRAIN.BASE_LR = linear_scaled_lr
        config.TRAIN.WARMUP_LR = linear_scaled_warmup_lr
        config.TRAIN.MIN_LR = linear_scaled_min_lr
        config.freeze()

    print('==========================================')
    print('real base lr: ', config.TRAIN.BASE_LR)
    print('==========================================')

    os.makedirs(config.OUTPUT, exist_ok=True)

    logger = create_logger(output_dir=config.OUTPUT, dist_rank=0 if torch.cuda.device_count() == 1 else dist.get_rank(), name=f"{config.MODEL.ARCH}")

    if torch.cuda.device_count() == 1 or dist.get_rank() == 0:
        path = os.path.join(config.OUTPUT, "config.json")
        with open(path, "w") as f:
            f.write(config.dump())
        logger.info(f"Full config saved to {path}")

    # print config
    logger.info(config.dump())

    main(config,device_id)

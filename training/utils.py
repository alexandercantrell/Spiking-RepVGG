import os
import shutil
import torch
import torch.nn as nn
from torch.backends import cudnn
import torch.distributed as dist
import numpy as np
import copy
import hashlib
from spikingjelly.activation_based import functional

def set_seeds(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)

def set_deterministic():
    cudnn.deterministic = True
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    # set a debug environment variable CUBLAS_WORKSPACE_CONFIG to ":16:8" (may limit overall performance) or ":4096:8" (will increase library footprint in GPU memory by approximately 24MiB).
    torch.use_deterministic_algorithms(True)

def make_output_tree(args):
    args.output = os.path.join(args.output,args.arch,f'{args.tag}_{args.cnf}')
    if os.path.exists(args.output) and args.clean:
        shutil.rmtree(args.output)
    os.makedirs(args.output,exist_ok=True)
    if is_main_process():
        args.pt = os.path.join(args.output,'pt')
        args.checkpoints = os.path.join(args.pt,'checkpoints')
        args.tb = os.path.join(args.output,'tb')
        os.makedirs(args.pt,exist_ok=True)
        os.makedirs(args.checkpoints,exist_ok=True)
        os.makedirs(args.tb,exist_ok=True)

def init_distributed_mode(args):
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        args.rank = int(os.environ["RANK"])
        args.world_size = int(os.environ["WORLD_SIZE"])
        args.gpu = int(os.environ["LOCAL_RANK"])

    else:
        print("Not using distributed mode")
        torch.cuda.set_device('cuda')
        args.distributed = False
        return
    args.distributed = True
    torch.cuda.set_device(args.gpu)
    args.dist_backend="nccl"
    print(f"| distributed init (rank {args.rank}): {args.dist_url}", flush=True)
    dist.init_process_group(
        backend=args.dist_backend, init_method=args.dist_url, world_size=args.world_size, rank=args.rank
    )
    dist.barrier()

def is_dist_avail_and_initialized():
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True


def get_world_size():
    if not is_dist_avail_and_initialized():
        return 1
    return dist.get_world_size()


def get_rank():
    if not is_dist_avail_and_initialized():
        return 0
    return dist.get_rank()


def is_main_process():
    return get_rank() == 0

def reduce_across_processes(val):
    if not is_dist_avail_and_initialized():
        # nothing to sync, but we still convert to tensor for consistency with the distributed case.
        return torch.tensor(val)

    t = torch.tensor(val, device="cuda")
    dist.barrier()
    dist.all_reduce(t)
    return t

def get_cache_path(filepath):
    h = hashlib.sha1(filepath.encode()).hexdigest()
    cache_path = os.path.join("~",".torch","vision","datasets","imagefolder",h[:10]+".pt")
    cache_path = os.path.expanduser(cache_path)
    return cache_path

def get_checkpoint_path(args):
    file_name = os.path.splitext(os.path.basename(args.resume))[0]
    if args.resume == 'latest':
        path = os.path.join(args.pt,'latest_checkpoint.pth')
    elif args.resume == 'best':
        path = os.path.join(args.pt,'best_checkpoint.pth')
    elif args.resume == 'best_ema':
        path = os.path.join(args.pt,'best_ema_checkpoint.pth')
    elif os.path.exists(args.resume):
        path = args.resume
    if os.path.exists(os.path.join(args.pt,f'{file_name}.pth')):
        path = os.path.join(args.pt,f'{file_name}.pth')
    elif os.path.exists(os.path.join(args.checkpoints,f'{file_name}.pth')):
        path = os.path.join(args.checkpoints,f'{file_name}.pth')
    else:
        raise FileNotFoundError(f"Cannot find given reserved word, path, or file: {args.resume}")
    return path

def load_checkpoint(args, model, optimizer, lr_scheduler, logger, model_ema=None, scaler=None):
    logger.info(f"Resuming from {args.resume}")
    if args.resume.startswith('https'):
        checkpoint = torch.hub.load_state_dict_from_url(
            args.resume,map_location='cpu',check_hash=True)
    else:
        path = get_checkpoint_path(args)
        checkpoint = torch.load(path,map_location='cpu')
    msg = model.load_state_dict(checkpoint['model'],strict=False)
    logger.info(msg)
    max_accuracy = -1.0
    if not (args.eval or args.throughput) and 'optimizer' in checkpoint and 'lr_scheduler' in checkpoint and 'epoch' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer'])
        lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
        args.start_epoch = checkpoint["epoch"] + 1
        if scaler:
            scaler.load_state_dict(checkpoint['scaler'])
        logger.info(f"Successfully loaded '{args.resume}' (epoch {checkpoint['epoch']})")
        if 'max_accuracy' in checkpoint:
            max_accuracy = checkpoint['max_accuracy']
    max_ema_accuracy=None
    if model_ema:
        model_ema.load_state_dict(checkpoint['model_ema'],strict=False)
        if 'max_ema_accuracy' in checkpoint:
            max_ema_accuracy = checkpoint['max_ema_accuracy']

    del checkpoint
    torch.cuda.empty_cache()
    return max_accuracy, max_ema_accuracy

def load_weights(args,model,logger):
    logger.info(f"Loading model from {args.resume}")
    if args.resume.startswith('https'):
        checkpoint = torch.hub.load_state_dict_from_url(
            args.resume,map_location='cpu',check_hash=True)
    else:
        path = get_checkpoint_path(args)
        checkpoint = torch.load(path,map_location='cpu')
    if 'model' in checkpoint:
        checkpoint = checkpoint['model']
    if 'state_dict' in checkpoint:
        checkpoint = checkpoint['state_dict']
    model.load_state_dict(checkpoint, strict=False)

def _save_checkpoint(path,args,epoch,model,max_accuracy,optimizer,lr_scheduler,logger,model_ema=None,max_ema_accuracy=None,scaler=None):
    checkpoint = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "lr_scheduler": lr_scheduler.state_dict(),
        "epoch": epoch,
        "args": args,
        "max_accuracy": max_accuracy
    }
    if model_ema:
        checkpoint["model_ema"] = model_ema.state_dict()
        checkpoint["max_ema_accuracy"] = max_ema_accuracy
    if scaler:
        checkpoint["scaler"] = scaler.state_dict()
    if is_main_process():
        torch.save(checkpoint,path)
        logger.info(f"Saved checkpoint to: {path}")

def save_latest(args,epoch,model, max_accuracy, optimizer, lr_scheduler, logger, model_ema=None, max_ema_accuracy=None, scaler=None):
    path = os.path.join(args.pt,"latest_checkpoint.pth")
    _save_checkpoint(path,args,epoch,model,max_accuracy,optimizer,lr_scheduler,logger,model_ema=model_ema,max_ema_accuracy=max_ema_accuracy,scaler=scaler)

def save_checkpoint(args, epoch, model, max_accuracy, optimizer, lr_scheduler, logger, model_ema=None, max_ema_accuracy=None, scaler=None, is_best=False, is_ema=False):
    if is_ema:
        path = os.path.join(args.pt,"best_ema_checkpoint.pth")
    elif is_best:
        path = os.path.join(args.pt,"best_checkpoint.pth")
    else:
        path = os.path.join(args.checkpoints,f"checkpoint_{epoch}.pth")
    _save_checkpoint(path,args,epoch,model,max_accuracy,optimizer,lr_scheduler,logger,model_ema=model_ema,max_ema_accuracy=max_ema_accuracy,scaler=scaler)


def store_model_weights(model, checkpoint_path, checkpoint_key="model", strict=True):
    """
    This method can be used to prepare weights files for new models. It receives as
    input a model architecture and a checkpoint from the training script and produces
    a file with the weights ready for release.

    Examples:
        from torchvision import models as M

        # Classification
        model = M.mobilenet_v3_large(weights=None)
        print(store_model_weights(model, './class.pth'))

        # Quantized Classification
        model = M.quantization.mobilenet_v3_large(weights=None, quantize=False)
        model.fuse_model(is_qat=True)
        model.qconfig = torch.ao.quantization.get_default_qat_qconfig('qnnpack')
        _ = torch.ao.quantization.prepare_qat(model, inplace=True)
        print(store_model_weights(model, './qat.pth'))

        # Object Detection
        model = M.detection.fasterrcnn_mobilenet_v3_large_fpn(weights=None, weights_backbone=None)
        print(store_model_weights(model, './obj.pth'))

        # Segmentation
        model = M.segmentation.deeplabv3_mobilenet_v3_large(weights=None, weights_backbone=None, aux_loss=True)
        print(store_model_weights(model, './segm.pth', strict=False))

    Args:
        model (pytorch.nn.Module): The model on which the weights will be loaded for validation purposes.
        checkpoint_path (str): The path of the checkpoint we will load.
        checkpoint_key (str, optional): The key of the checkpoint where the model weights are stored.
            Default: "model".
        strict (bool): whether to strictly enforce that the keys
            in :attr:`state_dict` match the keys returned by this module's
            :meth:`~torch.nn.Module.state_dict` function. Default: ``True``

    Returns:
        output_path (str): The location where the weights are saved.
    """
    # Store the new model next to the checkpoint_path
    checkpoint_path = os.path.abspath(checkpoint_path)
    output_dir = os.path.dirname(checkpoint_path)

    # Deep copy to avoid side effects on the model object.
    model = copy.deepcopy(model)
    checkpoint = torch.load(checkpoint_path, map_location="cpu")

    # Load the weights to the model to validate that everything works
    # and remove unnecessary weights (such as auxiliaries, etc.)
    if checkpoint_key == "model_ema":
        del checkpoint[checkpoint_key]["n_averaged"]
        torch.nn.modules.utils.consume_prefix_in_state_dict_if_present(checkpoint[checkpoint_key], "module.")
    model.load_state_dict(checkpoint[checkpoint_key], strict=strict)

    tmp_path = os.path.join(output_dir, str(model.__hash__()))
    torch.save(model.state_dict(), tmp_path)

    sha256_hash = hashlib.sha256()
    with open(tmp_path, "rb") as f:
        # Read and update hash string value in blocks of 4K
        for byte_block in iter(lambda: f.read(4096), b""):
            sha256_hash.update(byte_block)
        hh = sha256_hash.hexdigest()

    output_path = os.path.join(output_dir, "weights-" + str(hh[:8]) + ".pth")
    os.replace(tmp_path, output_path)

    return output_path

def repvgg_model_convert(model:nn.Module, save_path=None, do_copy=True):
    if do_copy:
        model = copy.deepcopy(model)
    functional.reset_net(model)
    for module in model.modules():
        if hasattr(module, 'switch_to_deploy'):
            module.switch_to_deploy()
    if save_path is not None:
        torch.save(model.state_dict(), save_path)
    return model
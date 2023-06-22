import os
import shutil
import numpy as np
import torch
import torch.nn as nn
from torch.backends import cudnn
import torch.distributed as dist
import copy
import hashlib

from spikingjelly.activation_based import functional

from fastargs.decorators import param
from fastargs import Param, Section

Section('output','').params(
    dir=Param(str,'',default='./logs/'),
    tag=Param(str,'',default='default'),
    save_freq=Param(int,'',default=1),
    clean=Param(bool,'',is_flag=True)
)

Section('deterministic','').params(
    seed=Param(int,'',default=2020),
    deterministic=Param(bool,'',is_flag=True)
)

@param('output.dir')
@param('model.arch')
@param('output.tag')
@param('model.cnf')
def get_output_dir(dir,arch,tag,cnf):
    return os.path.join(dir,arch,f'{tag}_{cnf}')

def get_pt_dir():
    return os.path.join(get_output_dir(),'pt')

def get_checkpoint_dir():
    return os.path.join(get_pt_dir(),'checkpoints')

def get_tb_dir():
    return os.path.join(get_output_dir(),'tb')

@param('model.resume')
def get_checkpoint_path(resume):
    file_name = os.path.splitext(os.path.basename(resume))[0]
    if resume == 'latest':
        path = os.path.join(get_pt_dir(),'latest_checkpoint.pth')
    elif resume == 'best':
        path = os.path.join(get_pt_dir(),'best_checkpoint.pth')
    elif os.path.exists(resume):
        path = resume
    elif os.path.exists(os.path.join(get_pt_dir(),f'{file_name}.pth')):
        path = os.path.join(get_pt_dir(),f'{file_name}.pth')
    elif os.path.exists(os.path.join(get_checkpoint_dir(),f'{file_name}.pth')):
        path = os.path.join(get_checkpoint_dir(),f'{file_name}.pth')
    else:
        raise FileNotFoundError(f"Cannot find given reserved word, path, or file: {resume}")
    return path

@param('output.clean')
def make_output_tree(clean):
    output = get_output_dir()
    if os.path.exists(output) and clean:
        shutil.rmtree(output)
    os.makedirs(output,exist_ok=True)
    if is_main_process():
        os.makedirs(get_pt_dir(),exist_ok=True)
        os.makedirs(get_checkpoint_dir(),exist_ok=True)
        os.makedirs(get_tb_dir(),exist_ok=True)

@param('model.resume')
def load_checkpoint(model, optimizer, lr_scheduler, scaler=None, resume='latest'):
    print(f"Resuming from {resume}")
    if resume.startswith('https'):
        checkpoint = torch.hub.load_state_dict_from_url(
            resume,map_location='cpu',check_hash=True)
    else:
        path = get_checkpoint_path()
        checkpoint = torch.load(path,map_location='cpu')
    msg = model.load_state_dict(checkpoint['model'],strict=False)
    print(msg)
    max_accuracy = -1.0
    if 'optimizer' in checkpoint and 'lr_scheduler' in checkpoint and 'epoch' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer'])
        lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
        start_epoch = checkpoint["epoch"] + 1
        if scaler:
            scaler.load_state_dict(checkpoint['scaler'])
        print(f"Successfully loaded '{resume}' (epoch {checkpoint['epoch']})")
        if 'max_accuracy' in checkpoint:
            max_accuracy = checkpoint['max_accuracy']
    del checkpoint
    torch.cuda.empty_cache()
    return start_epoch, max_accuracy

@param('model.resume')
def load_weights(model,resume):
    print(f"Loading model from {resume}")
    if resume.startswith('https'):
        checkpoint = torch.hub.load_state_dict_from_url(
            resume,map_location='cpu',check_hash=True)
    else:
        path = get_checkpoint_path()
        checkpoint = torch.load(path,map_location='cpu')
    if 'model' in checkpoint:
        checkpoint = checkpoint['model']
    if 'state_dict' in checkpoint:
        checkpoint = checkpoint['state_dict']
    model.load_state_dict(checkpoint, strict=False)

def _save_checkpoint(path,config,epoch,model,max_accuracy,optimizer,lr_scheduler,scaler=None):
    if is_dist_avail_and_initialized():
        dist.barrier()
    checkpoint = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "lr_scheduler": lr_scheduler.state_dict(),
        "epoch": epoch,
        "args": config.content,
        "max_accuracy": max_accuracy
    }
    if scaler:
        checkpoint["scaler"] = scaler.state_dict()
    if is_main_process():
        torch.save(checkpoint,path)
        print(f"Saved checkpoint to: {path}")

def save_latest(config,epoch,model, max_accuracy, optimizer, lr_scheduler, scaler=None):
    path = os.path.join(get_pt_dir(),"latest_checkpoint.pth")
    _save_checkpoint(path,config,epoch,model,max_accuracy,optimizer,lr_scheduler,scaler=scaler)

def save_checkpoint(config, epoch, model, max_accuracy, optimizer, lr_scheduler, scaler=None, is_best=False):
    if is_best:
        path = os.path.join(get_pt_dir(),"best_checkpoint.pth")
    else:
        path = os.path.join(get_checkpoint_dir(),f"checkpoint_{epoch}.pth")
    _save_checkpoint(path,config,epoch,model,max_accuracy,optimizer,lr_scheduler,scaler=scaler)

@param('deterministic.seed')
def set_seeds(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)

@param('deterministic.deterministic')
def set_deterministic(deterministic):
    if deterministic:
        cudnn.benchmark=False
        cudnn.deterministic = True
        os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
        # set a debug environment variable CUBLAS_WORKSPACE_CONFIG to ":16:8" (may limit overall performance) or ":4096:8" (will increase library footprint in GPU memory by approximately 24MiB).
        torch.use_deterministic_algorithms(True)
    #else:
    #    cudnn.benchmark=True

def init_distributed_mode():
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
        dist.init_process_group(
            backend='nccl',world_size=int(os.environ["WORLD_SIZE"]),rank=int(os.environ["RANK"])
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

def get_local_rank():
    if not is_dist_avail_and_initialized():
        return 0
    return int(os.environ["LOCAL_RANK"])

def get_device_name():
    device = 'cuda'
    if is_dist_avail_and_initialized():
        device += os.environ['LOCAL_RANK']
    return device

def is_main_process():
    return get_rank() == 0






#TODO: implement use cases

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
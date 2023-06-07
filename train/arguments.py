import argparse

def get_args_parser():
    parser = argparse.ArgumentParser(description="Spiking Rep-VGG training script built on the codebase of Rep-VGG and SpikingJelly",add_help=True)
    
    parser.add_argument("--num-classes",default=1000,type=int)#TODO: check if ffcv loader or reader has num_classes attribute
    parser.add_argument("--batches-ahead",default=3,type=int)#TODO:position correctly

    #experiment settings
    parser.add_argument("--epochs", default=120, type=int, metavar="N", help="number of total epochs to run")
    parser.add_argument("--resume", default=None, type=str, help="path of checkpoint. If set to 'latest', it will try to load the latest checkpoint")
    parser.add_argument("--start-epoch", default=0, type=int, metavar="N", help="start epoch")
    parser.add_argument('--T', default=4, type=int, help="total time-steps")

    #dataset
    parser.add_argument("--data-path", default="/your/path/to/dataset", type=str, help="dataset path")
    parser.add_argument("--dataset",default=None,type=str,help="")

    #pipelines
    parser.add_argument("--train-crop-size", default=176, type=int, help="the random crop size used for training (default: 176)")
    parser.add_argument("--val-resize-size", default=232, type=int, help="the resize size used for validation (default: 232)")
    parser.add_argument("--val-crop-size", default=224, type=int, help="the central crop size used for validation (default: 224)")
    parser.add_argument("--auto-augment", default=None, type=str, help="auto augment policy (default: None)")
    parser.add_argument("--interpolation", default="bilinear", type=str, help="the interpolation method (default: bilinear)")
    parser.add_argument("--ra-magnitude", default=9, type=int, help="magnitude of auto augment policy")
    parser.add_argument("--augmix-severity", default=3, type=int, help="severity of augmix policy")
    parser.add_argument("--data-mean",default=None,type=tuple,help="mean values for custom datasets e.g. (0.485, 0.456, 0.406)")
    parser.add_argument("--data-std",default=None,type=tuple,help="std values for custom dataset e.g. '(0.229, 0.224, 0.225)'")
    parser.add_argument("--random-erase", default=0.1, type=float, help="random erasing probability (default: 0.1)")
    parser.add_argument("--mixup-alpha", default=0.2, type=float, help="mixup alpha (default: 0.2)")

    #loader
    parser.add_argument("-b", "--batch-size", default=128, type=int, help="images per gpu, the total batch size is $NGPU x batch_size")
    parser.add_argument("-j", "--workers", default=16, type=int, metavar="N", help="number of data loading workers (default: 16)")
    parser.add_argument("--in-memory",default=1,choices=[0,1])

    #model
    parser.add_argument("--arch", default=None, type=str, help="model name")
    parser.add_argument("--cnf",type=str,default="ADD")
    parser.add_argument("--fast-surrogate",dest="fast_surrogate",help="uses experimental and potentially faster surrogate method",action="store_true")
    parser.add_argument("--surrogate-alpha",default=2.0)
    parser.add_argument("--cupy",dest='cupy', help="set the neurons to use cupy backend", action="store_true")
    parser.add_argument("--label-smoothing", default=0.1, type=float, help="label smoothing (default: 0.1)")
    parser.add_argument('--use-checkpoint', action='store_true', help="whether to use gradient checkpointing to save memory")
    parser.add_argument('--detach-reset',action='store_true')
    #learning rate scheduler
    parser.add_argument("--lr-scheduler", default="cosa", type=str, help="the lr scheduler (default: cosa)")
    parser.add_argument("--lr-warmup-epochs", default=5, type=int, help="the number of epochs to warmup (default: 5)")
    parser.add_argument("--lr-warmup-method", default="linear", type=str, help="the warmup method (default: linear)")
    parser.add_argument("--lr-warmup-decay", default=0.01, type=float, help="the decay for lr")
    parser.add_argument("--lr-step-size", default=30, type=int, help="decrease lr every step-size epochs")
    parser.add_argument("--lr-gamma", default=0.1, type=float, help="decrease lr by a factor of lr-gamma")
    parser.add_argument("--lr-min", default=0.0, type=float, help="minimum lr of lr schedule (default: 0.0)")

    #optimizer
    parser.add_argument("--opt", default="sgd", type=str, help="optimizer")
    parser.add_argument("--lr", default=0.1, type=float, help="initial learning rate")
    parser.add_argument("--momentum", default=0.9, type=float, metavar="M", help="momentum")
    parser.add_argument("--wd","--weight-decay",default=0.,type=float,metavar="W",help="weight decay (default: 0.)",dest="weight_decay")
    parser.add_argument("--norm-weight-decay", default=None, type=float, help="weight decay for Normalization layers (default: None, same value as --wd)")
    parser.add_argument("--bias-weight-decay",default=None,type=float,help="weight decay for bias parameters of all layers (default: None, same value as --wd)")
    parser.add_argument("--transformer-embedding-decay",default=None,type=float,help="weight decay for embedding parameters for vision transformer models (default: None, same value as --wd)")

    #automatic mixed precision
    parser.add_argument("--disable-amp",dest='disable_amp', help="not use automatic mixed precision training",action="store_true") #TODO: always use

    #clip-grad
    parser.add_argument("--clip-grad-norm", default=None, type=float, help="the maximum gradient norm (default None)")

    #ema
    parser.add_argument("--model-ema", action="store_true", help="enable tracking Exponential Moving Average of model parameters")
    parser.add_argument("--model-ema-steps",type=int,default=32,help="the number of iterations that controls how often to update the EMA model (default: 32)")
    parser.add_argument("--model-ema-decay",type=float,default=0.99998,help="decay factor for Exponential Moving Average of model parameters (default: 0.99998)")

    #output
    parser.add_argument('--output', default='./logs/', type=str, metavar='PATH', help='root of output folder, the full path is <output>/<model_name>/<tag> (default: output)')
    parser.add_argument('--tag', default='default', type=str, help='tag of experiment')
    parser.add_argument("--clean", action="store_true", help="delete the dirs for tensorboard logs and pt files")
    parser.add_argument("--save-freq",default=20,type=int,help='save frequency')
    parser.add_argument("--print-freq", default=10, type=int, help="print frequency")
    
    #mode
    parser.add_argument("--eval",dest="eval",help="Only evaluate the model",action="store_true")
    parser.add_argument("--throughput",dest="throughput",help="Test throughput only",action="store_true")
    parser.add_argument("--throughput-reset",dest="throughput_reset",action="store_true")
    #other 
    parser.add_argument("--device", default="cuda", type=str, help="device (Use cuda or cpu Default: cuda)")   
    parser.add_argument("--deterministic", action="store_true", help="set 'torch.use_deterministic_algorithms(True)', which can cause errors with some functions that do not have a deterministic implementation")
    parser.add_argument("--seed", default=2020, type=int, help="the random seed")
    parser.add_argument("--sync-bn",dest="sync_bn",help="Use sync batch norm",action="store_true")
    parser.add_argument("--channels-last",dest="channels_last",action="store_true")
    parser.add_argument("--use-blurpool",dest="use_blurpool",action="store_true")
    return parser
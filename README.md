# Spiking-RepVGG
## Introduction
NOTE: This repository is home to my work revolving around taking the Spiking Element-Wise Resnet (SEWResnet) framework, and applying it to RepVGG. It is still a work in progress so I cannot promise code cleanliness or that all code will work.
## Code structure

### Dependencies
```shell
pip install -r requirements.txt
```

## Running the code
### ImageNet
```bash
python imagenet/dst_train.py --data.path <data_path> --data.T 4 --model.arch SRepVGG_A0 --model.cupy --lr.lr 0.01 --optim.weight_decay 0.0005 --logging.tag <tag>
```

```bash
python imagenet/dst_train.py --data.path <data_path> --data.T 4 --model.arch SRepVGG_A0 --model.cupy --model.conversion --model.conversion_set_y --lr.lr 0.01 --optim.weight_decay 0.0005 --logging.tag <tag>
```

### CIFAR10-DVS
```bash
python cifar10dvs/dst_train.py --data.path <data_path> --data.T 16 --model.arch SRepVGG_N1 --model.cupy --lr.lr 0.005 --optim.weight_decay 0.0005 --logging.tag <tag>
```

```bash
python cifar10dvs/dst_train.py --data.path <data_path> --data.T 16 --model.arch SRepVGG_N1 --model.cupy --model.conversion --model.conversion_set_y --lr.lr 0.005 --optim.weight_decay 0.0005 --logging.tag <tag>
```

### DVS Gesture
```bash
python dvsgesture/dst_train.py --data.path <data_path> --data.T 16 --training.T_train 12 --model.arch SRepVGG_N0 --model.cupy --lr.lr 0.01 --optim.weight_decay 0.0005 --logging.tag <tag>
```

```bash
python dvsgesture/dst_train.py --data.path <data_path> --data.T 16 --training.T_train 12 --model.arch SRepVGG_N0 --model.cupy --model.conversion --model.conversion_set_y --lr.lr 0.01 --optim.weight_decay 0.0005 --logging.tag <tag>
```
# FEVER-OOD: Free Energy Vulnerability Elimination for Robust Out-of-Distribution Detection
Implementation of [FEVER-OOD](https://arxiv.org/abs/2412.01596), a novel method for out-of-distribution detection 
that leverages free energy-based OOD methods, such as [VOS](https://github.com/deeplearning-wisc/vos) or [Dream-OOD](https://github.com/deeplearning-wisc/dream-ood), to eliminate the vulnerability of these methods 
to regions in the feature space that, while abnormal, are still labelled as in-distribution. The reduction of these regions
improve such free energy-based methods. We achieve state-of-the-art results when applying FEVER-OOD to Dream-OOD using the Imagenet-100 dataset as in-distribution.

![energy_in_feature_space.png](imgs/energy_in_feature_space.jpg)

This repository is based in the [VOS](https://github.com/deeplearning-wisc/vos) and [Dream-OOD](https://github.com/deeplearning-wisc/dream-ood) repositories, 
with some changes for compatibility between VOS and Dream-OOD, and the addition of the FEVER-OOD method.

## Requirements 

The classification and object detection models are implemented in PyTorch. To use this code, setup an environment and then:

1. Install PyTorch and torchvision following the directions on the [official website](https://pytorch.org/).
2. Install the requirements using `pip install -r requirements.txt`.
3. _Object Deteciton Only_: Install Detectron2 using the official instructions [here](https://detectron2.readthedocs.io/en/latest/tutorials/install.html).

## FEVER-OOD - Classification

We train and evaluate VOS, FFS and Dream-OOD in the CIFAR10, CIFAR100 and Imagenet-100 datasets. The training and testing procedures are described below. All the code regarding classification is its corresponding directory.
### In-Distribution Datasets
All the datasets are considered to be in a directory `<DATA-DIR>`.

#### CIFAR10 and CIFAR100
Download and extract the CIFAR10 and CIFAR100 datasets from the [official website](https://www.cs.toronto.edu/~kriz/cifar.html) and place them in `<DATA-DIR>/cifarpy`. The dir structure should be:

```
<DATA-DIR>
│
└───cifarpy
    │
    └───cifar-10-batches-py
    │
    └───cifar-100-python
```

#### Imagenet-100
Download and extract the Imagenet-1k dataset from [Kaggle](https://www.kaggle.com/c/imagenet-object-localization-challenge/overview/description) and place it in `<DATA-DIR>/imagenet-1k`.
Then create an `imagenet-100` directory and run the `classification/tools/create_imagenet-100.py` script inside of it, pointing to the downloaded imagenet-1k dataset. The dir structure should be:

```
<DATA-DIR>
│
└───imagenet-100
    │
    └───train
    │
    └───val
```

### OOD Datasets
For the CIFAR experiments, we follow the same procedure for the OOD datasets as in [here](https://github.com/deeplearning-wisc/npos?tab=readme-ov-file#out-of-distribution-dataset) (we don't use the LSUN-R dataset). The datasets are placed in `<DATA-DIR>/ood_datasets`, with the following structure:

``` 
<DATA-DIR>
│
└───dtd
    │
    └───images
    │
    └───imdb
    │
    └───labels
│
└───iSUN
    │
    └───iSUN_patches
│
└───LSUN_C
    │
    └───test
│
└───places365
    │
    └───test_samples
│
└───svhn
    │
    └───test_32x32.mat
```

For the Imagenet-100 experiments, we use the same OOD datasets as in [here](https://github.com/deeplearning-wisc/knn-ood?tab=readme-ov-file#out-of-distribution-dataset). The dir structure should be:

```
<DATA-DIR>
│
└───dtd
    │
    └───images
    │
    └───imdb
    │
    └───labels
│
└───SUN
    │
    └───images
│
└───Places
    │
    └───images
│
└───iNaturalist
    │
    └───test_samples
```

For Dream-OOD experiments, we use the generated CIFAR-100 and Imagenet-100 synthetic outliers from the [Dream-OOD](https://github.com/deeplearning-wisc/dream-ood).

### Training
#### VOS/FFS
Training involves the use of in-distribution data only. To train a VOS/FFS model, use the `classification/train-vos.py` script, with the following options:

```
train-vos.py [--dataset {cifar10,cifar100,imagenet-1k,imagenet-100}] [--data-root DATA_ROOT] [--model {wrn,rn34,rn50}] 
                    [--vos-start-epoch VOS_START_EPOCH] [--vos-sample-number VOS_SAMPLE_NUMBER] [--vos-select VOS_SELECT] [--vos-sample-from VOS_SAMPLE_FROM]
                    [--vos-loss-weight VOS_LOSS_WEIGHT] [--use_ffs] [--smin_loss_weight SMIN_LOSS_WEIGHT] [--use_conditioning] [--null-space-red-dim NULL_SPACE_RED_DIM] [<OTHER OPTIONS>]

Trains a Classifier with VOS/FFS for OOD Detection

optional arguments:
  --dataset {cifar10,cifar100,imagenet-1k,imagenet-100}
                        Choose between CIFAR-10, CIFAR-100, Imagenet-100, Imagenet-1k. (default: None)
  --data-root DATA_ROOT
                        Root for the dataset. (default: ./data)
  --model {wrn,rn34,rn50}, -m {wrn,rn34,rn50}
                        Choose architecture. (default: wrn)
  --vos-start-epoch VOS_START_EPOCH
                        Epoch to start VOS/FFS. (default: 40)
  --vos-sample-number VOS_SAMPLE_NUMBER
                        Number of samples to keep per class. (default: 1000)
  --vos-select VOS_SELECT
                        Number of least-likely samples to select from OOD samples. (default: 1)
  --vos-sample-from VOS_SAMPLE_FROM
                        Number of samplings to construct OOD samples. (default: 10000)
  --vos-loss-weight VOS_LOSS_WEIGHT
                        Weight for VOS loss. (default: 0.1)
  --use_ffs             Use FFS instead of VOS. (default: False)
  --smin_loss_weight SMIN_LOSS_WEIGHT
                        Weight for least singular value/conditioning number loss. (default: 0.0)
  --use_conditioning    Use conditioning number instead of least singular value. (default: False)
  --null-space-red-dim NULL_SPACE_RED_DIM
                        Dimensionality reduction for null space. (default: -1)

```
Other options are also available (please check the script for more information). For instance, to train FFS in the CIFAR-10 dataset with a Wide ResNet-40 model using a Null Space Reduction of 64 and Least Singular Value Regularizer, use the following command:

```
python train-vos.py --dataset cifar10 --model wrn --vos-loss-weight 0.1 --use_ffs --smin_loss_weight 0.1 --null-space-red-dim 64
```

#### Dream-0OD
Training Dream-OOD models involves using synthetic outliers generated from the in-distribution data. To train a Dream-OOD model, use the `classification/train-dream-ood.py` script, with the following options:

```
train-dream-ood.py [--dataset {cifar100,imagenet-100}] [--model {r34,r50}] 
                          [--energy_weight ENERGY_WEIGHT] [--smin_loss_weight SMIN_LOSS_WEIGHT] [--use_conditioning]
                          [--null-space-red-dim NULL_SPACE_RED_DIM] [--id-root ID_ROOT] [--ood-root OOD_ROOT] [<OTHER OPTIONS>]

Trains a Classifier with OOD Detection using Dream-OOD and FEVER-OOD

optional arguments:
  -h, --help            show this help message and exit
  --dataset {cifar100,imagenet-100}
                        Choose CIFAR-100 and Imagenet-100 (default: None)
  --model {r34,r50}, -m {r34,r50}
                        Choose architecture. (default: r50)
  --energy_weight ENERGY_WEIGHT
                        Energy regularization weight (default: 2.5)
  --smin_loss_weight SMIN_LOSS_WEIGHT
                        Weight for least singular value/conditioning number loss. (default: 0.0)
  --use_conditioning    Use conditioning number instead of least singular value. (default: False)
  --null-space-red-dim NULL_SPACE_RED_DIM
                        Dimensionality reduction for null space. (default: -1)
  --id-root ID_ROOT     Path to CIFAR-100 in-distribution training data (default: ./data/cifarpy)
  --ood-root OOD_ROOT   Path to OOD data (default: ./data/dream-ood-cifar-outliers)

```

FEVER-OOD related options are similar to VOS/FFS. Therefore, to train Dream-OOD in the Imagenet-100 dataset with a ResNet-34 using a Null Space Reduction of 114 and Conditioning Number Regularizer, use the following command:
    
```
python train-dream-ood.py --dataset imagenet-100 --model r34 --energy_weight 2.5 --use_conditioning --null-space-red-dim 114 --id-root ./<DATA-DIR>/imagenet-100 --ood-root ./<DATA-DIR>/<IN-OUTLIERS-DIR>
```
### Testing

## FEVER-OOD - Object Detection

### Datasets

### Training

### Testing

## Null Space Projection

## Citation
You can cite this work as follows:

```
@article{isaac-medina24fever-ood, 
    author = {Isaac-Medina, B.K.S. and Che, M. and Gaus, Y.F.A. and Akcay, S. and Breckon, T.P.}, 
    title = {FEVER-OOD: Free Energy Vulnerability Elimination for Robust Out-of-Distribution Detection}, 
    journal={arXiv preprint arXiv:2412.01596}, 
    year = {2024}, 
    month = {December}
}
```


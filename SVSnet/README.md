# PyTorch Implementation of the Sequential vessel segmentation via deep channel attention network

<a href="https://pytorch.org/"><img src="https://img.shields.io/badge/PyTorch-v2.7+-red.svg?logo=PyTorch&style=for-the-badge" /></a>
<a href="#"><img src="https://img.shields.io/badge/python-v3.10+-blue.svg?logo=python&style=for-the-badge" /></a>

This repository contains code for a image segmentation model base on [Sequential vessel segmentation via deep channel attention network](https://www.sciencedirect.com/science/article/pii/S0893608020301672) implemented in PyTorch.

<img src="https://github.com/wgyhhhh/PyTorch-SVSnet/blob/main/svsnet.jpg" width="700" />


## Installation

1. Create an Anaconda environment.
```sh
conda create -n svsnet python=3.10
conda activate svsnet
```

2. Install PyTorch
```sh
pip install -r requirements.txt
```

3. Install pip Packages
```sh
pip install -r requirements.txt
```

## Training on Your Dataset

1. The file origin structure is the following:
   
```
inputs
└── Origin Dataset
    ├── train
    |   ├── images
    │   │   ├── 001.png
    │   │   ├── 002.png...
    │   ├── masks
    │   │   ├── 001.png
    │   │   ├── 002.png...
    ├── test
    |   ├── images
    │   │   ├── 001.png
    │   │   ├── 002.png...
    │   ├── masks
    │   │   ├── 001.png
    │   │   ├── 002.png...
    ...
```
2. Proprocess.
```sh
python convert_png_to_npy.py
```
Then the file structure becomes:
```
inputs
└── train
    └──train_images.npy
    │
    └──train_labels.npy
└── test
    └──test_images.npy
    │
    └──test_labels.npy
```
3. Train the model.
```sh
python train.py
```
4. Evaluate the model.
```sh
python eval.py
```

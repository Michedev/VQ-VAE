![PyPI - Downloads](https://img.shields.io/pypi/dm/vqvae)
![PyPI](https://img.shields.io/pypi/v/vqvae)
![PyPI - Python Version](https://img.shields.io/pypi/pyversions/vqvae)

# VQ-VAE

VQ-VAE implementation based on _Pytorch, Pytorch Lightning, Anaconda-project and Hydra_.

## Install pip package

```bash
pip install vqvae
```

Note that _pip_ package contains only _model/_ folder

## Anaconda-project

1. Clone the repository

```bash
git clone https://github.com/Michedev/VQ-VAE
```

2. Install [anaconda](https://www.anaconda.com/) if you don't have it



## Train

Train your model

```bash
anaconda-project run train-gpu
```

_Note: First time will download and install all dependencies_

You can also specify additional arguments according to `config/train.yaml` like

```bash
anaconda-project run train-cpu  # train on cpu
```


## Project structure

    ├── data  # Data storage folder
    ├── callbacks  # train/test callbacks
    ├── config
    │   ├── dataset  # Dataset config
    │   ├── model  # Model config
    │   ├── model_dataset  # model and dataset specific config
    │   ├── test.yaml   # testing configuration
    │   └── train.yaml  # training configuration
    ├── dataset  # Dataset definition
    ├── model  # Model definition
    │   └── callbacks  # model callbacks
    ├── utils
    │   ├── experiment_tools.py # Iterate over experiments
    │   └── paths.py  # common paths
    ├── train.py  # Entrypoint point for training
    ├── test.py  # Entrypoint point for testing
    ├── anaconda-project.yml  # Project configuration
    ├── saved_models  # where models are saved
    └── readme.md  # This file

### Design keypoints
- root folder should contain only entrypoints and folders
- Add tasks to anaconda-project.yml via the command `anaconda-project add-command`


### Anaconda-project FAQ

#### How to add a new command?
Example:
```bash
anaconda-project add-command generate "python ddpm_pytorch/generate.py
```
#### Mac OS support in lock file

_[Short]_ Run these commands:

```bash
anaconda-project remove-packages cudatoolkit;
anaconda-project add-platforms osx-64;
```

_[Long]_
1. Remove cudatoolkit dependency from _anaconda-project.yml_
```bash
anaconda-project remove-packages cudatoolkit
```
2. Add Mac OS platform to _anaconda-project-lock.yml_:
```bash
anaconda-project add-platforms osx-64
``` 

![PyPI - Downloads](https://img.shields.io/pypi/dm/vqvae)
![PyPI](https://img.shields.io/pypi/v/vqvae)
![PyPI - Python Version](https://img.shields.io/pypi/pyversions/vqvae)

# Deep Learning Template

This is a template for deep learning projects based on _Pytorch, Pytorch Lightning, Anaconda-project and Hydra_.

## How to initialize a new project

1. Clone this repository
2. Rename deep_learning_template to your project name in _anaconda-project.yml_ and folder name
3. Create your model and, optionally, your dataset in _{src}/model_ and _{src}/dataset_ respectively


## Install

1. Clone the repository

```bash
git clone https://github.com/Michedev/deep-learning-template.git
```

2. Install [anaconda](https://www.anaconda.com/) if you don't have it



## Train

Train your model

```bash
anaconda-project run train-gpu
```

_Note: First time will download and install all dependencies_

You can also specify additional arguments according to `deep_learning_template/config/train.yaml` like

```bash
anaconda-project run train accelerator=cpu  # train on cpu
```


## Project structure

    ├── data  # Data folder
    ├── deep_learning_template  # source code
    │   ├── config
    │   │   ├── dataset  # Dataset config
    │   │   ├── model  # Model config
    │   │   ├── model_dataset  # model and dataset specific config
    │   │   ├── test.yaml   # testing configuration
    │   │   └── train.yaml  # training configuration
    │   ├── dataset  # Dataset definition
    │   ├── model  # Model definition
    │   ├── utils
    │   │   ├── experiment_tools.py # Iterate over experiments
    │   │   └── paths.py  # common paths
    │   ├── train.py  # Entrypoint point for training
    │   └── test.py  # Entrypoint point for testing
    ├── anaconda-project.yml  # Project configuration
    ├── saved_models  # where models are saved
    └── readme.md  # This file

### Design keypoints
- Code root folder (_deep_learning_template/_) should contain only entrypoint
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

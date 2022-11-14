from setuptools import setup

# setup.py to publish only model/ folder

setup(
    name='pytorch_vqvae',
    version='1.0.0',
    packages=['pytorch_vqvae'],
    url='https://github.com/Michedev/VQ-VAE',
    license='MIT',
    author='mikedev',
    author_email='mik3dev@gmail.com',
    description='PyTorch implementation of VQ-VAE',
    dependencies=['pytorch-lightning', 'torchvision', 'torch', 'tensorguard'],

)

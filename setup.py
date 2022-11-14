from setuptools import setup

setup(
    name='pytorch_vqvae',
    version='1.0.0',
    packages=['model', 'utils', 'dataset', 'callbacks'],
    url='https://github.com/Michedev/VQ-VAE',
    license='MIT',
    author='mikedev',
    author_email='mik3dev@gmail.com',
    description='PyTorch implementation of VQ-VAE',
    dependencies=['pytorch-lightning', 'torchvision', 'torch', 'tensorguard']
)

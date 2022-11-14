from setuptools import setup

# setup.py to publish only model/ folder

long_description = \
"""
# Pytorch VQVAE implementation

## Example

```python
from vqvae import VQVAE, sequential_encoder, sequential_decoder
from torch.optim import  Adam
from functools import partial

input_channels = 3
output_channels = 3
embedding_length = 256
hidden_channels = 64
beta = 0.25
embedding_size = 512
opt = partial(Adam, lr=2e-4)

encoder = sequential_encoder(input_channels, embedding_size, hidden_channels)  # Encoder from the paper
decoder = sequential_decoder(embedding_size, output_channels, hidden_channels)  # Decoder from the paper
vqvae = VQVAE(encoder, decoder, beta, embedding_length, embedding_size, opt)  # Pytorch-Lightning module, 
                                                                              # hence usable to train the model

```
"""

setup(
    name='vqvae',
    version='1.0.0',
    packages=['vqvae'],
    url='https://github.com/Michedev/VQ-VAE',
    license='MIT',
    author='mikedev',
    author_email='mik3dev@gmail.com',
    description='PyTorch implementation of VQ-VAE',
    dependencies=['pytorch-lightning', 'torchvision', 'torch', 'tensorguard'],
    long_description=long_description,
    long_description_content_type='text/markdown',
)

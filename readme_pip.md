# Pytorch VQVAE implementation
Pytorch implementation of [Neural Discrete Representation Learning (Van den Oord, 2017)](https://arxiv.org/abs/1711.00937)

The model (VQVAE) is implemented as Pytorch Lightning module, hence the training step is already implemented.

The model has been tested on https://github.com/Michedev/VQ-VAE
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
vqvae = VQVAE(encoder, decoder, opt, beta, embedding_length, embedding_size)  # Pytorch-Lightning module, 
                                                                              # hence usable to train the model
```

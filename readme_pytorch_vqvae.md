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

encoder = sequential_encoder(input_channels, embedding_size, hidden_channels)
decoder = sequential_decoder(embedding_size, output_channels, hidden_channels)
vqvae = VQVAE(encoder, decoder, beta, embedding_length, embedding_size, opt)

```
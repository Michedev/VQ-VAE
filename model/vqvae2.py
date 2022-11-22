import pytorch_lightning as pl
from torch import nn
import torch

from model.vector_quantization import reshape2d_quantize


class VQVAE2(pl.LightningModule):

    def __init__(self, encoder_bottom: nn.Module, encoder_top: nn.Module, decoder: nn.Module,
                 codebook_length: int, embedding_size: int):
        """

        @param encoder_bottom: The encoder for the low-level image features.
                               It receives the image as input and outputs the downscaled image by a factor of 4
        @param encoder_top: The encoder for the high-level image features. It receives the output of encoder_bottom
                            as input and outputs the feature map downscaled by a factor of 2
        @param decoder: Receives the output of encoder_top (upsampled by a factor of 2) and encoder_bottom as inputs
                        and outputs the reconstructed image
        """
        super().__init__()
        self.encoder_bottom = encoder_bottom
        self.encoder_top = encoder_top
        self.decoder = decoder

        self.upsample_top = nn.ConvTranspose2d(encoder_top.output_channels, encoder_top.output_channels, 4, stride=2)
        self.top_codebook = nn.Parameter(torch.zeros(codebook_length, embedding_size))
        self.bottom_codebook = nn.Parameter(torch.zeros(codebook_length, embedding_size))
        self.register_parameter('top_codebook', self.top_codebook)
        self.register_parameter('bottom_codebook', self.bottom_codebook)
        with torch.no_grad():
            nn.init.xavier_normal_(self.top_codebook)
            nn.init.xavier_normal_(self.bottom_codebook)



    def decode(self, e_top, e_bottom):
        e_top_bottom = torch.cat([self.upsample_top(e_top), e_bottom], dim=-1)
        return self.decoder(e_top_bottom)

    def encode(self, x):
        e_bottom = self.encoder_bottom(x)
        e_top = self.encoder_top(e_bottom)
        return e_top, e_bottom

    def forward(self, x):
        e_top, e_bottom = self.encode(x)
        e_top_flatten, e_top_flatten_quantized, e_top_quantized = reshape2d_quantize(e_top, self.top_codebook)
        e_bottom_flatten, e_bottom_flatten_quantized, e_bottom_quantized = \
            reshape2d_quantize(e_bottom, self.bottom_codebook)
        x_hat = self.decode(e_top_quantized, e_bottom_quantized)
        return dict(x_hat=x_hat, e_top_flatten=e_top_flatten, e_top_flatten_quantized=e_top_flatten_quantized,
                    e_top_quantized=e_top_quantized, e_bottom_flatten=e_bottom_flatten,
                    e_bottom_flatten_quantized=e_bottom_flatten_quantized, e_bottom_quantized=e_bottom_quantized)
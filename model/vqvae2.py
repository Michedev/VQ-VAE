from typing import Union, Sequence

import pytorch_lightning as pl
from pytorch_lightning import Callback
from torch import nn
import torch
import tensorguard as tg
from callbacks.ema_embedding import EMAEmbedding
from model.vector_quantization import reshape2d_quantize


class VQVAE2(pl.LightningModule):

    def __init__(self, encoder_bottom: nn.Module, encoder_top: nn.Module, decoder: nn.Module,
                 codebook_length: int, embedding_size: int, beta: float = 0.25, freq_log: int=1000):
        """

        @param encoder_bottom: The encoder for the low-level image features.
                               It receives the image as input and outputs the downscaled image by a factor of 4
        @param encoder_top: The encoder for the high-level image features. It receives the output of encoder_bottom
                            as input and outputs the feature map downscaled by a factor of 2
        @param decoder: Receives the output of encoder_top (upsampled by a factor of 2) and encoder_bottom as inputs
                        and outputs the reconstructed image
        @param codebook_length: The number of entries in the codebook
        @param embedding_size: The size of the embedding vector
        @param beta: The weight of the commitment loss
        @param freq_log: The frequency of logging the validation loss
        """
        super().__init__()
        self.freq_log = freq_log
        self.encoder_bottom = encoder_bottom
        self.encoder_top = encoder_top
        self.decoder = decoder
        self.beta = beta

        self.upsample_top = nn.ConvTranspose2d(encoder_top.output_channels, encoder_top.output_channels, 4, stride=2)
        self.top_codebook = nn.Parameter(torch.zeros(codebook_length, embedding_size))
        self.bottom_codebook = nn.Parameter(torch.zeros(codebook_length, embedding_size))
        self.register_parameter('top_codebook', self.top_codebook)
        self.register_parameter('bottom_codebook', self.bottom_codebook)
        self.bce = nn.BCEWithLogitsLoss(reduction='none')
        with torch.no_grad():
            nn.init.xavier_normal_(self.top_codebook)
            nn.init.xavier_normal_(self.bottom_codebook)

    def configure_callbacks(self) -> Union[Sequence[Callback], Callback]:
        return EMAEmbedding()

    def decode(self, e_top: torch.Tensor, e_bottom: torch.Tensor):
        e_top_bottom = torch.cat([self.upsample_top(e_top), e_bottom], dim=-1)
        return self.decoder(e_top_bottom)

    def encode(self, x: torch.Tensor):
        e_bottom = self.encoder_bottom(x)
        e_top = self.encoder_top(e_bottom)
        return e_top, e_bottom

    def forward(self, x: torch.Tensor):
        e_top, e_bottom = self.encode(x)
        e_top_flatten, e_top_flatten_quantized, e_top_quantized = reshape2d_quantize(e_top, self.top_codebook)
        e_bottom_flatten, e_bottom_flatten_quantized, e_bottom_quantized = \
            reshape2d_quantize(e_bottom, self.bottom_codebook)
        x_hat = self.decode(e_top_quantized, e_bottom_quantized)
        return dict(x_hat=x_hat, e_top_flatten=e_top_flatten, e_top_flatten_quantized=e_top_flatten_quantized,
                    e_top_quantized=e_top_quantized, e_bottom_flatten=e_bottom_flatten,
                    e_bottom_flatten_quantized=e_bottom_flatten_quantized, e_bottom_quantized=e_bottom_quantized)

    def training_step(self, batch, batch_idx):
        return self._step(batch, batch_idx, 'train')

    def validation_step(self, batch, batch_idx):
        result = self._step(batch, batch_idx, 'valid')

        tg.guard(batch[0], '*, C, H, W')
        tg.guard(result['x_hat'], '*, C, W, H')
        tg.guard(result['e_top_flatten'], '*, CS2, ES')
        tg.guard(result['e_top_flatten_quantized'], '*, CS2, ES')
        tg.guard(result['e_top_quantized'], '*, ES, W2, H2')
        tg.guard(result['e_bottom_flatten'], '*, CS1, ES')
        tg.guard(result['e_bottom_flatten_quantized'], '*, CS1, ES')
        tg.guard(result['e_bottom_quantized'], '*, ES, W1, H1')

        return result

    def _step(self, batch, batch_idx, dataset_split: str):
        x, _ = batch
        outputs = self(x)
        x_hat = outputs['x_hat']
        e_top_flatten = outputs['e_top_flatten']
        e_bottom_flatten = outputs['e_bottom_flatten']
        e_top_flatten_quantized = outputs['e_top_flatten_quantized']
        e_bottom_flatten_quantized = outputs['e_bottom_flatten_quantized']
        loss_dict = self.loss_function(x, x_hat, e_top_flatten, e_bottom_flatten,
                                  e_top_flatten_quantized, e_bottom_flatten_quantized)
        if dataset_split == 'valid' or self.global_step % self.freq_log == 0:
            self._log_metrics(dataset_split, loss_dict, x, x_hat)
            if (dataset_split == 'valid' and batch_idx == 0) or dataset_split == 'train':
                # log true image
                self.logger.experiment.add_images(f'{dataset_split}/x', x, self.global_step)
                # log the image reconstruction
                self.logger.experiment.add_images(f'{dataset_split}/x_hat', x_hat, self.global_step)

        return loss_dict

    @torch.no_grad()
    def _log_metrics(self, dataset_split, loss_dict, x, x_hat):
        for loss_name, loss_value in loss_dict.items():
            self.log(f'{dataset_split}/{loss_name}', loss_value, on_step=False, on_epoch=True)

    def loss_function(self, x, x_hat, e_top_flatten, e_bottom_flatten,
                      e_top_flatten_quantized, e_bottom_flatten_quantized):
        recon_loss = self.bce(x_hat, x).mean(dim=0).sum()
        top_loss = torch.mean((e_top_flatten - e_top_flatten_quantized.detach()) ** 2)
        bottom_loss = torch.mean((e_bottom_flatten - e_bottom_flatten_quantized.detach()) ** 2)
        commitment_loss = self.beta * (top_loss + bottom_loss)
        loss = recon_loss + commitment_loss
        return dict(loss=loss, recon_loss=recon_loss, top_loss=top_loss, bottom_loss=bottom_loss,
                    commitment_loss=commitment_loss)
from typing import Union, Sequence

import pytorch_lightning as pl
from pytorch_lightning import Callback
from torch import nn
import torch
import tensorguard as tg
from torch.optim import Adam

from .callbacks.ema_embedding import EMAEmbedding
from .vector_quantization import reshape2d_quantize


class VQVAE2(pl.LightningModule):

    def __init__(self, encoder_bottom: nn.Module, encoder_top: nn.Module, decoder_bottom: nn.Module,
                 decoder_top: nn.Module, codebook_length: int, embedding_size: int, beta: float = 0.25, freq_log: int=1000):
        """

        @param encoder_bottom: The encoder for the low-level image features.
                               It receives the image as input and outputs the downscaled image by a factor of 4
        @param encoder_top: The encoder for the high-level image features. It receives the output of encoder_bottom
                            as input and outputs the feature map downscaled by a factor of 2
        @param decoder_bottom: Receives the output of encoder_top (upsampled by a factor of 2) and encoder_bottom as inputs
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
        self.decoder_bottom = decoder_bottom
        self.decoder_top = decoder_top
        self.beta = beta

        self.conv_merge_top_bottom = nn.Conv2d(2 * embedding_size, embedding_size, kernel_size=1)
        self.codebook_top = nn.Parameter(torch.zeros(codebook_length, embedding_size))
        self.codebook_bottom = nn.Parameter(torch.zeros(codebook_length, embedding_size))
        self.register_parameter('codebook_top', self.codebook_top)
        self.register_parameter('codebook_bottom', self.codebook_bottom)
        self.bce = nn.BCEWithLogitsLoss(reduction='none')
        with torch.no_grad():
            nn.init.xavier_normal_(self.codebook_top)
            nn.init.xavier_normal_(self.codebook_bottom)

    def configure_callbacks(self) -> Union[Sequence[Callback], Callback]:
        return [EMAEmbedding(suffix='top'),
                EMAEmbedding(suffix='bottom')]

    def encode(self, x: torch.Tensor):
        e_bottom = self.encoder_bottom(x)
        e_top = self.encoder_top(e_bottom)
        return e_top, e_bottom

    def forward(self, x: torch.Tensor):
        e_top, e_bottom = self.encode(x)
        decoded = self.decode(e_bottom, e_top)
        return dict(**decoded, e_top=e_top, e_bottom=e_bottom)

    def decode(self, e_bottom, e_top) -> dict:
        # Quantize the top embedding
        e_top_flatten, e_top_flatten_quantized, e_top_quantized = reshape2d_quantize(e_top, self.codebook_top)
        # Upsample the top embedding by a factor of 2
        e_top_quantized_upsampled = self.decoder_top(e_top_quantized)
        tg.guard(e_bottom, "*, ES, WB, HB")
        tg.guard(e_top_quantized_upsampled, "*, ES, WB, HB")
        # Concatenate the bottom embedding and the upsampled top embedding
        e_bottom = torch.cat([e_bottom, e_top_quantized_upsampled], dim=1)
        # Apply a 1x1 convolution to merge the two embeddings and to return at embedding_size channels
        e_bottom = self.conv_merge_top_bottom(e_bottom)
        # Quantize the merged embedding
        e_bottom_flatten, e_bottom_flatten_quantized, e_bottom_quantized = \
            reshape2d_quantize(e_bottom, self.codebook_bottom)
        # Decode the merged embedding
        x_hat = self.decoder_bottom(e_bottom_quantized)
        return dict(e_bottom_flatten=e_bottom_flatten, e_bottom_flatten_quantized=e_bottom_flatten_quantized,
                    e_bottom_quantized=e_bottom_quantized, e_top_flatten=e_top_flatten,
                    e_top_flatten_quantized=e_top_flatten_quantized, e_top_quantized=e_top_quantized, x_hat=x_hat)

    def training_step(self, batch, batch_idx):
        return self._step(batch, batch_idx, 'train')

    def validation_step(self, batch, batch_idx):
        tg.guard(batch[0], '*, C, H, W')

        result = self._step(batch, batch_idx, 'valid')

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

        return dict(**loss_dict, **outputs)

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

    def configure_optimizers(self):
        return Adam(self.parameters(), lr=2e-4)  # todo: move into config
from functools import partial

import pytorch_lightning as pl
import tensorguard as tg
import torch
from torch import nn

from .vector_quantization import reshape2d_quantize


class VQVAE(pl.LightningModule):

    def __init__(self, encoder: nn.Module, decoder: nn.Module, opt: partial, beta: float = 0.25,
                 codebook_length: int = 512, embedding_size: int = 64, debug: bool = False,
                 logging_train_freq: int = 1_000):
        """
        VQ-VAE model.
        @param encoder: The encoder network. The output shape is (batch_size, codebook_length, w1, h1)
        @param decoder: The decoder network. Input size is  (batch_size, codebook_length, w1, h1)
        @param beta: Commitment loss weight
        @param codebook_length: Number of embedding vectors
        @param embedding_size: Size of embedding vectors
        @param opt: Partial function that returns an optimizer. Inside the object, it will be called with the parameters
        @param debug: If True, the model will print the shapes of the tensors
        @param logging_train_freq: How often to log the training loss
        """
        super().__init__()
        self.logging_train_freq = logging_train_freq
        self.encoder = encoder
        self.decoder = decoder
        self.beta = beta
        self._opt_partial = opt
        self.mse = nn.MSELoss(reduction='none')
        self.codebook_length = codebook_length
        self.embedding_size = embedding_size
        self.w_embedding = nn.Parameter(torch.randn(codebook_length, embedding_size))
        self.register_parameter('w_embedding', self.w_embedding)
        tg.set_dim('LS', self.codebook_length)
        self.debug = debug
        tg.set_dim('ES', embedding_size)
        self.bce = nn.BCEWithLogitsLoss(reduction='none')
        with torch.no_grad():
            nn.init.xavier_normal_(self.w_embedding)

    def forward(self, x):
        encoded = self.encoder(x)
        e_flatten, e_quantized_flatten, e_quantized = reshape2d_quantize(encoded, self.w_embedding)
        x_hat = self.decoder(e_quantized)
        return dict(x_hat=x_hat, e_quantized_flatten=e_quantized_flatten,
                    e_flatten=e_flatten, e_quantized=e_quantized)

    def training_step(self, batch, batch_idx):
        return self._step(batch, batch_idx, dataset_split='train')

    def validation_step(self, batch, batch_idx):
        result = self._step(batch, batch_idx, dataset_split='valid')

        tg.guard(batch[0], "*, C, W, H")
        tg.guard(self.w_embedding, "LS, ES")
        tg.guard(result['x_hat'], "*, C, W, H")
        tg.guard(result['e'], "*, L1, ES")
        tg.guard(result['e_quantized_flatten'], "*, L1, ES")

        return result

    def _print_grad(self, loss_dict):
        old_value = self.automatic_optimization
        self.automatic_optimization = False
        self.manual_backward(loss_dict['loss'])
        print('encoder')
        for m in self.encoder.modules():
            if isinstance(m, nn.Conv2d):
                print('\tgradients')
                wgrad = m.weight.grad
                bgrad = m.bias.grad
                print('\t\tweight is na =', wgrad is None)
                print('\t\tbias is '
                      'na =', bgrad is None)
                if not wgrad is None:
                    print('\t\tmean weight grad =', wgrad.mean().item())
                if not bgrad is None:
                    print('\t\tmean bias grad =', bgrad.mean().item())
        print('decoder')
        for m in self.decoder.modules():
            if isinstance(m, nn.ConvTranspose2d):
                print('\tgradients')
                wgrad = m.weight.grad
                bgrad = m.bias.grad
                print('\t\tweight is na =', wgrad is None)
                print('\t\tbias is na =', bgrad is None)
                if not wgrad is None:
                    print('\t\tmean weight grad =', wgrad.mean().item())
                if not bgrad is None:
                    print('\t\tmean bias grad =', bgrad.mean().item())
        print(f'{self.w_embedding.grad.mean().item() =}')
        self.automatic_optimization = old_value
        self.zero_grad()

    @torch.no_grad()
    def log_metrics(self, loss_dict, forward_result: dict, dataset_split='train'):
        x = forward_result['x']
        x_hat = forward_result['x_hat']
        self.logger.experiment.add_images(f'{dataset_split}/x', x, self.global_step)
        self.logger.experiment.add_images(f'{dataset_split}/x_hat', x_hat.sigmoid(), self.global_step)
        self.log('%s/loss' % dataset_split, loss_dict['loss'], on_step=True, on_epoch=True, prog_bar=False)
        self.log('%s/loss_recon' % dataset_split, loss_dict['loss_recon'], on_step=True, on_epoch=True, prog_bar=True)
        self.log('%s/loss_vq' % dataset_split, loss_dict['loss_vq'], on_step=True, on_epoch=True, prog_bar=True)
        self.log('%s/loss_commit' % dataset_split, loss_dict['loss_commit'], on_step=True, on_epoch=True, prog_bar=True)

    def _step(self, batch, batch_idx, dataset_split='train'):
        """Step function, both for training and validation"""
        x, _ = batch
        if self.debug:
            with torch.no_grad():
                print(f'{x.mean().item()=}, {x.std().item()=}')
        forward_result = self(x)
        forward_result['x'] = x
        x_hat = forward_result['x_hat']
        e_flatten = forward_result['e_flatten']
        e_quantized_flatten = forward_result['e_quantized_flatten']
        loss_dict = self.calc_loss(x, x_hat, e_flatten, e_quantized_flatten)
        if (dataset_split == 'valid' and batch_idx < 3) or self.global_step % self.logging_train_freq == 0:
            self.log_metrics(loss_dict, forward_result, dataset_split)
        result = {**loss_dict, **forward_result}
        return result

    def calc_loss(self, x, x_recon, e, e_quantized) -> dict:
        loss_recon = self.bce(x_recon, x).mean(dim=0).sum()
        loss_vq = self.mse(e_quantized.detach(), e).mean(dim=0).sum()
        loss_commit = self.beta * self.mse(e_quantized.detach(), e).mean()
        if self.debug:
            with torch.no_grad():
                print(f'{loss_recon=}, {loss_vq=}, {loss_commit=}')
                print(
                    f'{e.mean().item()=}, {e.std().item()=}, {e_quantized.mean().item()=}, {e_quantized.std().item()=}')
        return dict(loss=loss_recon + loss_vq + loss_commit,
                    loss_recon=loss_recon, loss_vq=loss_vq,
                    loss_commit=loss_commit)

    def configure_optimizers(self):
        optimizer = self._opt_partial(params=self.parameters())
        return optimizer

from functools import partial
from typing import Any

import pytorch_lightning as pl
import tensorguard as tg
import torch
from torch import nn, autograd


class VectorQuantizer(autograd.Function):

    @staticmethod
    def forward(ctx: Any, e: torch.Tensor, w_embedding: torch.Tensor) -> Any:
        """
        Quantize the embedding

        >>> w = torch.tensor([[1, 2, 3], [4, 5, 6]]).float() # 2 x 3
        >>> e = torch.zeros(32, 15, 3).float()
        >>> result = VectorQuantizer.apply(e, w)
        >>> result.shape
        torch.Size([32, 15, 3])
        >>> (result.sum().item() == 6 * 15 * 32)
        True
        >>> e = torch.zeros(2, 4, 3)
        >>> e[1] += 5
        >>> result = VectorQuantizer.apply(e, w)
        >>> (result[0].sum().item() == 4 * 6)
        True
        >>> (result[1].sum().item() == 4 * 15)
        True

        @param e: the embedding tensor with shape (batch_size, length, embedding_dim)
        @param w_embedding: the embedding dictionary with shape (num_embeddings, embedding_dim)

        @return the quantized embedding with shape (batch_size, length, embedding_dim)
        """
        B = e.shape[0]  # batch size
        E = w_embedding.shape[-1]  # embedding size
        with torch.no_grad():
            dist = torch.cdist(e, w_embedding)
            i_min = torch.argmin(dist, dim=-1)
        result = w_embedding.unsqueeze(0).expand(B, -1, -1).gather(dim=1, index=i_min.unsqueeze(-1).expand(-1, -1, E))
        ctx.save_for_backward(e, w_embedding, i_min, result)
        return result


    @staticmethod
    def backward(ctx: Any, grad_output: torch.Tensor) -> Any:
        grad_e = None
        grad_w_embedding = None
        # print('grad_output.shape =', grad_output.shape)
        if ctx.needs_input_grad[0]:
            grad_e = grad_output.clone()
        if ctx.needs_input_grad[1]:
            e, w_embedding, i_min, e_hat = ctx.saved_tensors
            grad_w_embedding = torch.zeros_like(w_embedding)
            embedding_size = grad_output.shape[-1]
            grad_output_flatten = grad_output.contiguous().view(-1, embedding_size)
            grad_w_embedding = grad_w_embedding.index_add(dim=0, index=i_min.view(-1),
                                                          source=grad_output_flatten)
        return grad_e, grad_w_embedding


quantize = VectorQuantizer.apply


class VQVAE(pl.LightningModule):

    def __init__(self, encoder, decoder, beta: float, latent_size: int,
                 embedding_size: int, opt: partial, debug: bool = False, logging_train_freq=1_000):
        super().__init__()
        self.logging_train_freq = logging_train_freq
        self.encoder = encoder
        self.decoder = decoder
        self.beta = beta
        self._opt_partial = opt
        self.mse = nn.MSELoss(reduction='none')
        self.latent_size = latent_size
        self.embedding_size = embedding_size
        self.register_parameter('w_embedding', nn.Parameter(torch.randn(latent_size, embedding_size)))
        tg.set_dim('LS', self.latent_size)
        self.debug = debug
        tg.set_dim('ES', embedding_size)
        self.bce = nn.BCEWithLogitsLoss(reduction='none')
        with torch.no_grad():
            nn.init.xavier_normal_(self.w_embedding)

    def forward(self, x):
        e = self.encoder(x)
        w_e, h_e = e.shape[2:]
        e = e.flatten(start_dim=2)
        e = e.permute(0, 2, 1)
        e_quantized = quantize(e, self.w_embedding)
        e_quantized_reshaped = e_quantized.permute(0, 2, 1)
        e_quantized_reshaped = e_quantized_reshaped.reshape(-1, self.embedding_size, w_e, h_e)
        x_recon = self.decoder(e_quantized_reshaped)
        return dict(x_recon=x_recon, e_quantized=e_quantized,
                    e=e, e_quantized_reshaped=e_quantized_reshaped)

    def training_step(self, batch, batch_idx):
        return self._step(batch, batch_idx, dataset_split='train')

    def validation_step(self, batch, batch_idx):
        return self._step(batch, batch_idx, dataset_split='valid')

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
        x_recon = forward_result['x_recon']
        self.logger.experiment.add_images(f'{dataset_split}/x', x , self.global_step)
        self.logger.experiment.add_images(f'{dataset_split}/x_recon', x_recon.sigmoid(), self.global_step)
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
        x_recon = forward_result['x_recon']
        e = forward_result['e']
        e_quantized = forward_result['e_quantized']
        if dataset_split == 'valid':
            tg.guard(x, "*, C, W, H")
            tg.guard(self.w_embedding, "LS, ES")
            tg.guard(x_recon, "*, C, W, H")
            tg.guard(e, "*, L1, ES")
            tg.guard(e_quantized, "*, L1, ES")
        loss_dict = self.calc_loss(x, x_recon, e, e_quantized)
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

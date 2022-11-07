from typing import Any

import pytorch_lightning as pl
import tensorguard as tg
import torch
from torch import nn, autograd


def sequential_encoder(input_channels: int, output_channels: int):
    return nn.Sequential(
        nn.Conv2d(input_channels, 32, kernel_size=5, padding=2),
        nn.GroupNorm(1, 32),
        nn.ReLU(),
        nn.Conv2d(32, 32, kernel_size=5, padding=2, stride=2),
        nn.GroupNorm(1, 32),
        nn.ReLU(),
        nn.Conv2d(32, 32, kernel_size=5, padding=2),
        nn.GroupNorm(1, 32),
        nn.ReLU(),
        nn.Conv2d(32, 32, kernel_size=5, padding=2, stride=2),
        nn.GroupNorm(1, 32),
        nn.ReLU(),
        nn.Conv2d(32, output_channels, kernel_size=5, padding=2),
    )


def sequential_decoder(input_channels: int, output_channels: int):
    x = nn.Sequential(
        nn.ConvTranspose2d(input_channels, 32, kernel_size=5),
        nn.GroupNorm(1, 32),
        nn.ReLU(),
        nn.ConvTranspose2d(32, 32, kernel_size=5),
        nn.GroupNorm(1, 32),
        nn.ReLU(),
        nn.ConvTranspose2d(32, 32, kernel_size=5),
        nn.GroupNorm(1, 32),
        nn.ReLU(),
        nn.ConvTranspose2d(32, 32, kernel_size=5),
        nn.GroupNorm(1, 32),
        nn.ReLU(),
        nn.ConvTranspose2d(32, output_channels, kernel_size=6),
    )
    for m in x.modules():
        if isinstance(m, nn.Linear): print(m.weight.shape)
    return x


class VectorQuantizer(autograd.Function):

    @staticmethod
    def forward(ctx: Any, e, w_embedding) -> Any:
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

        @return the quantized embedding
        """
        B = e.shape[0]  # batch size
        E = w_embedding.shape[-1]  # embedding size
        with torch.no_grad():
            # e: B, LS, ES
            # w_embedding: LS, ES
            # dist: B, LS, LS
            dist = torch.cdist(e, w_embedding)
            # min_dist: B, LS
            i_min = torch.argmin(dist, dim=-1)
        ctx.save_for_backward(e, w_embedding, i_min)
        result = w_embedding.unsqueeze(0).expand(B, -1, -1).gather(dim=1, index=i_min.unsqueeze(-1).expand(-1, -1, E))
        return result

    @staticmethod
    def backward(ctx: Any, grad_output) -> Any:
        grad_e = None
        grad_w_embedding = None
        if ctx.needs_input_grad[0]:
            grad_e = grad_output.clone()
        if ctx.needs_input_grad[1]:
            e, w_embedding, i_min = ctx.saved_tensors
            # print('============================')
            # print(f'{e.shape=}, {w_embedding.shape=}, {i_min.shape=}, {grad_output.shape=}')
            # print('============================')
            grad_w_embedding: torch.Tensor = torch.zeros_like(w_embedding)
            # grad_div: torch.Tensor = torch.zeros_like(w_embedding)
            #
            embedding_size = grad_output.shape[-1]
            grad_output_flatten = grad_output.contiguous().view(-1, embedding_size)
            grad_w_embedding = grad_w_embedding.index_add(dim=0, index=i_min.view(-1),
                                                          source=- grad_output_flatten)

        return grad_e, grad_w_embedding


quantize = VectorQuantizer.apply


class VQVAE(pl.LightningModule):

    def __init__(self, encoder, decoder, beta: float, latent_size: int,
                 embedding_size: int, debug: bool = False):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.beta = beta
        self.mse = nn.MSELoss(reduction='none')
        self.latent_size = latent_size
        self.embedding_size = embedding_size
        self.register_parameter('w_embedding', nn.Parameter(torch.randn(latent_size, embedding_size)))
        tg.set_dim('LS', self.latent_size)
        self.debug = debug
        tg.set_dim('ES', embedding_size)
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
        x, _ = batch
        result = self(x)
        x_recon = result['x_recon']
        e = result['e']
        e_quantized = result['e_quantized']
        loss_dict = self.calc_loss(x, x_recon, e, e_quantized)
        if self.debug:
            self._print_grad(loss_dict)

        if self.global_step % 1_000 == 0:
            self.log_metrics(loss_dict, result)
        return loss_dict

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
                print('\t\tbias is na =', bgrad is None)
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

        self.automatic_optimization = old_value
        self.zero_grad(set_to_none=True)

    def log_metrics(self, loss_dict, forward_result: dict, dataset_split='train'):
        x_recon = forward_result['x_recon']
        self.logger.experiment.add_images('x_recon', x_recon, self.global_step)
        self.log('%s/loss' % dataset_split, loss_dict['loss'], on_step=True, on_epoch=True, prog_bar=False)
        self.log('%s/recon_loss' % dataset_split, loss_dict['recon_loss'], on_step=True, on_epoch=True, prog_bar=True)
        self.log('%s/embedding_loss' % dataset_split, loss_dict['embedding_loss'], on_step=True, on_epoch=True, prog_bar=True)
        self.log('%s/commit_loss' % dataset_split, loss_dict['commit_loss'], on_step=True, on_epoch=True, prog_bar=True)

    def calc_loss(self, x, x_recon, e, e_quantized) -> dict:
        recon_loss = self.mse(x_recon, x).mean(dim=0).sum()
        embedding_loss = self.mse(e_quantized, e.detach()).mean()
        commit_loss = self.beta * self.mse(e_quantized.detach(), e).mean()
        if self.debug:
            with torch.no_grad():
                print(f'{recon_loss=}, {embedding_loss=}, {commit_loss=}')
                print(f'{e.mean().item()=}, {e.std().item()=}, {e_quantized.mean().item()=}, {e_quantized.std().item()=}')
        return dict(loss=recon_loss + embedding_loss + commit_loss,
                    recon_loss=recon_loss, embedding_loss=embedding_loss,
                    commit_loss=commit_loss)

    def validation_step(self, batch, batch_idx):
        x, _ = batch
        if self.debug:
            with torch.no_grad():
                print(f'{x.mean().item()=}, {x.std().item()=}')
        tg.guard(x, "*, C, W, H")
        forward_result = self(x)
        x_recon = forward_result['x_recon']
        e = forward_result['e']
        e_quantized = forward_result['e_quantized']
        tg.guard(self.w_embedding, "LS, ES")
        tg.guard(x_recon, "*, C, W, H")
        tg.guard(e, "*, L1, ES")
        tg.guard(e_quantized, "*, L1, ES")

        loss_dict = self.calc_loss(x, x_recon, e, e_quantized)
        self.log_metrics(loss_dict, forward_result, dataset_split='valid')
        result = {**loss_dict, **forward_result}
        return result

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-4)
        return optimizer

from typing import Tuple, List
from math import prod

import pytorch_lightning as pl
import torch
from torch import nn
import tensorguard as tg
import torch.distributions as dist


def sequential_encoder(input_channels: int, output_channels: int):
    return nn.Sequential(
        nn.Conv2d(input_channels, 32, kernel_size=5, padding=2),
        nn.ReLU(),
        nn.Conv2d(32, 32, kernel_size=5, padding=2, stride=2),
        nn.ReLU(),
        nn.Conv2d(32, 32, kernel_size=5, padding=2),
        nn.ReLU(),
        nn.Conv2d(32, 32, kernel_size=5, padding=2, stride=2),
        nn.ReLU(),
        nn.Conv2d(32, output_channels, kernel_size=5, padding=2),
    )


def sequential_decoder(input_channels: int, output_channels: int):
    x = nn.Sequential(
        nn.ConvTranspose2d(input_channels, 32, kernel_size=5),
        nn.ReLU(),
        nn.ConvTranspose2d(32, 32, kernel_size=5),
        nn.ReLU(),
        nn.ConvTranspose2d(32, 32, kernel_size=5),
        nn.ReLU(),
        nn.ConvTranspose2d(32, 32, kernel_size=5),
        nn.ReLU(),
        nn.ConvTranspose2d(32, 32, kernel_size=5),
        nn.ReLU(),
        nn.ConvTranspose2d(32, output_channels, kernel_size=6),
    )
    for m in x.modules():
        if isinstance(m, nn.Linear): print(m.weight.shape)
    return x


def quantize(e, w_embedding):
    """
    Quantize the embedding

    >>> w = torch.tensor([[1, 2, 3], [4, 5, 6]]).float() # 2 x 3
    >>> e = torch.zeros(32, 15, 3).float()
    >>> result = quantize(e, w)
    >>> result.shape
    torch.Size([32, 15, 3])
    >>> (result.sum().item() == 6 * 15 * 32)
    True
    >>> e = torch.zeros(2, 4, 3)
    >>> e[1] += 5
    >>> result = quantize(e, w)
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
    result = w_embedding.unsqueeze(0).expand(B, -1, -1).gather(dim=1, index=i_min.unsqueeze(-1).expand(-1, -1, E))
    return result


class VQVAE(pl.LightningModule):

    def __init__(self, encoder, decoder, beta: float, latent_size: int,
                 embedding_size: int):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.beta = beta
        self.mse = nn.MSELoss()
        self.latent_size = latent_size
        self.embedding_size = embedding_size
        self.register_parameter('w_embedding', nn.Parameter(torch.randn(latent_size, embedding_size)))
        tg.set_dim('LS', self.latent_size)
        tg.set_dim('ES', embedding_size)
        with torch.no_grad():
            self.w_embedding *= 0.02  # init with normal(0, 0.02)

    def forward(self, x):
        e = self.encoder(x)
        w_e, h_e = e.shape[2:]
        e = e.flatten(start_dim=2)
        e = e.permute(0, 2, 1)
        e_quantized = quantize(e, self.w_embedding)
        e_quantized = e_quantized.permute(0, 2, 1)
        e_quantized = e_quantized.reshape(-1, self.embedding_size, w_e, h_e)
        x_recon = self.decoder(e_quantized)
        return dict(x_recon=x_recon, e_quantized=e_quantized, e=e)

    def training_step(self, batch, batch_idx):
        x, _ = batch
        result = self(x)
        x_recon = result['x_recon']
        e = result['e']
        e_quantized = result['e_quantized']
        loss_dict = self.calc_loss(x, x_recon, e, e_quantized)

        return loss_dict

    def calc_loss(self, x, x_recon, e, e_quantized) -> dict:
        recon_loss = self.mse(x_recon, x)
        embedding_loss = self.mse(e_quantized, e.detach())
        commit_loss = self.beta * self.mse(e_quantized.detach(), e)
        return dict(loss=recon_loss + embedding_loss + commit_loss,
                    recon_loss=recon_loss, embedding_loss=embedding_loss,
                    commit_loss=commit_loss)

    def validation_step(self, batch, batch_idx):
        x, _ = batch
        tg.guard(x, "*, C, W, H")
        result = self(x)
        x_recon = result['x_recon']
        e = result['e']
        e_quantized = result['e_quantized']

        tg.guard(x_recon, "*, C, W, H")

        loss_dict = self.calc_loss(x, x_recon, e, e_quantized)
        return loss_dict

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer

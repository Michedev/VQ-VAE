import torch
from pytorch_lightning.callbacks import Callback
import pytorch_lightning as pl

from model import VQVAE


class EMAEmbedding(Callback):

    def __init__(self, decay: float = 0.999):
        super().__init__()
        self.decay = decay
        self.ema_w_embedding = None
        self.N_i = None
        self.m_i = None

    @torch.no_grad()
    def on_train_batch_end(
        self, trainer: "pl.Trainer", vqvae: "VQVAE",
            outputs: dict, batch: tuple, batch_idx: int
    ) -> None:
        e_x = outputs['e']  # bs, L1, ES
        dist = torch.cdist(e_x, vqvae.w_embedding)
        i_min = torch.argmin(dist, dim=-1)  # bs, L1
        n_i = torch.zeros(vqvae.w_embedding.shape[0])  # L2
        i_flatten = i_min.view(-1)
        n_i.index_add_(dim=0, index=i_flatten, source=torch.ones_like(i_flatten))
        sum_e_i = torch.zeros_like(vqvae.w_embedding)  # L2, ES
        sum_e_i.index_add_(dim=0, index=i_flatten, source=e_x.view(-1, vqvae.embedding_size))
        if self.N_i is None:
            assert self.m_i is None, 'self.N_i and self.m_i must be both None'
            self.N_i = n_i
            self.m_i = sum_e_i
        self.N_i = self.decay * self.N_i + (1 - self.decay) * n_i
        self.m_i = self.decay * self.m_i + (1 - self.decay) * sum_e_i
        vqvae.w_embedding = self.m_i / torch.maximum(self.N_i.unsqueeze(-1), torch.tensor([1.0]))
from typing import Optional

import torch
from pytorch_lightning.callbacks import Callback
import pytorch_lightning as pl

from model import VQVAE


class EMAEmbedding(Callback):

    def __init__(self, decay: float = 0.999, suffix: Optional[str] = None):
        super().__init__()
        self.decay = decay
        self.ema_w_embedding = None
        self.N_i = None
        self.m_i = None
        self.suffix = None
        self.embedding_key = 'e'
        self.codebook_key = 'codebook'
        if self.suffix is not None:
            self.embedding_key += f'_{suffix}'
            self.codebook_key += f'_{suffix}'


    @torch.no_grad()
    def on_train_batch_end(
        self, trainer: "pl.Trainer", vqvae: "VQVAE",
            outputs: dict, batch: tuple, batch_idx: int
    ) -> None:
        e_x = outputs[self.embedding_key]  # bs, L1, ES
        codebook = getattr(vqvae, self.codebook_key)
        dist = torch.cdist(e_x, codebook)
        i_min = torch.argmin(dist, dim=-1)  # bs, L1
        n_i = torch.zeros(codebook.shape[0])  # L2
        i_flatten = i_min.view(-1)
        n_i.index_add_(dim=0, index=i_flatten, source=torch.ones_like(i_flatten))
        sum_e_i = torch.zeros_like(codebook)  # L2, ES
        sum_e_i.index_add_(dim=0, index=i_flatten, source=e_x.view(-1, vqvae.embedding_size))
        if self.N_i is None:
            assert self.m_i is None, 'self.N_i and self.m_i must be both None'
            self.N_i = n_i
            self.m_i = sum_e_i
        self.N_i = self.decay * self.N_i + (1 - self.decay) * n_i
        self.m_i = self.decay * self.m_i + (1 - self.decay) * sum_e_i
        setattr(vqvae, self.codebook_key, self.m_i / torch.maximum(self.N_i.unsqueeze(-1), torch.tensor([1.0])))
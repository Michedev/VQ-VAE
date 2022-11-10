import torch
from pytorch_lightning.callbacks import Callback


class EMAEmbedding(Callback):

    def __init__(self, decay: float = 0.999):
        super().__init__()
        self.decay = decay
        self.ema_w_embedding = None
        self.ema_

    @torch.no_grad()
    def on_train_batch_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        w_embedding = pl_module.w_embedding
        if self.ema_w_embedding is None:
            self.ema_w_embedding = w_embedding.clone()
        else:
            self.ema_w_embedding = self.decay * self.ema_w_embedding + (1 - self.decay) * w_embedding
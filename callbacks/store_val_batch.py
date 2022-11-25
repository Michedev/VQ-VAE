import pytorch_lightning as pl
import torch
import torchvision


class StoreFirstValBatch(pl.Callback):

    def __init__(self, batch_size=16):
        self.batch_size = batch_size

    def on_validation_batch_end(
            self,
            trainer: "pl.Trainer",
            pl_module: "pl.LightningModule",
            outputs: dict,
            batch: tuple,
            batch_idx: int,
            dataloader_idx: int,
    ) -> None:
        if batch_idx == 0:
            self.store_batch(batch, outputs)

    def store_batch(self, batch, outputs):
        assert isinstance(batch, (tuple, list)), f'{type(batch)}'
        x = batch[0]
        x_hat = outputs['x_hat_logit'].sigmoid()
        bs = x.shape[0]
        if self.batch_size > bs:
            print(
                f'Warning: expected batch size {self.batch_size} > actual batch size {bs}, setting expected batch size to {bs}')
            self.batch_size = bs
        x = x[:self.batch_size]
        x_hat = x_hat[:self.batch_size]
        # make grid
        x_xhat = torch.stack([x, x_hat], dim=1).flatten(0, 1)
        grid_x_xhat = torchvision.utils.make_grid(x_xhat, nrow=self.batch_size // 2)
        torchvision.utils.save_image(grid_x_xhat, 'val_true_recon_batch.png')

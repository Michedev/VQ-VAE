import hydra
import pkg_resources
from omegaconf import DictConfig, OmegaConf
from path import Path

from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl
import yaml


@hydra.main(pkg_resources.resource_filename("vq_vae", 'config'), 'test.yaml')
def test(config: DictConfig):
    ckpt_folder = Path(config.checkpoint_path)
    assert ckpt_folder.exists() and ckpt_folder.isdir(), f"Checkpoint {ckpt_folder} does not exist or is not a directory"

    print('Set seed to {}'.format(config.seed))
    pl.seed_everything(config.seed)

    ckpt_config = OmegaConf.load(ckpt_folder / 'config.yaml')

    trainer = pl.Trainer(resume_from_checkpoint=ckpt_folder / 'best.ckpt')

    model: pl.LightningModule = hydra.utils.instantiate(ckpt_config.model)

    test_dataset: Dataset = hydra.utils.instantiate(ckpt_config.dataset.instance, train=False)

    test_dl = DataLoader(test_dataset, batch_size=config.batch_size)

    test_metrics = trainer.test(model, test_dl)

    yaml_test_metrics = yaml.safe_dump(test_metrics)

    print(yaml_test_metrics)
    with open(ckpt_folder / 'test_metrics.yaml', 'w') as f:
        f.write(yaml_test_metrics)

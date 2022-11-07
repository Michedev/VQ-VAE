from typing import Iterator, Tuple
from path import Path
from utils.paths import SAVED_MODELS
from omegaconf import DictConfig, OmegaConf
import pytorch_lightning as pl
import hydra

def iter_experiments() -> Iterator[Tuple[Path, DictConfig]]:
    for experiment in SAVED_MODELS.dirs():
        config_path = experiment / 'config.yaml'
        config = OmegaConf.load(config_path)
        yield experiment, config


def iter_experiments_with_checkpoint() -> Iterator[Tuple[Path, DictConfig, pl.LightningModule]]:
    for experiment_path, config in iter_experiments():
        model = hydra.utils.instantiate(config.model)
        model.load_from_checkpoint(experiment_path / 'best.ckpt')
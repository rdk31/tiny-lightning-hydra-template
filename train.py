#!/usr/bin/env python


import hydra
import lightning as L
import rootutils
import wandb
from lightning.pytorch.loggers import WandbLogger
from omegaconf import DictConfig, OmegaConf

from src.utils import RankedLogger

rootutils.setup_root(__file__, pythonpath=True)


log = RankedLogger(__name__, rank_zero_only=True)


@hydra.main(version_base=None, config_path="config", config_name="default")
def main(cfg: DictConfig):
    if wandb.run:
        wandb.finish()

    L.seed_everything(cfg.core.seed, workers=True)

    log.info("Instantiating wandb")
    wandb_logger: WandbLogger = hydra.utils.instantiate(cfg.wandb)

    log.info("Logging hyperparameters!")
    wandb_logger.log_hyperparams(
        OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True)  # type: ignore
    )

    log.info(f"Instantiating datamodule <{cfg.data._target_}>")
    datamodule: L.LightningDataModule = hydra.utils.instantiate(cfg.data)

    log.info(f"Instantiating model <{cfg.model._target_}>")
    model: L.LightningModule = hydra.utils.instantiate(cfg.model)

    log.info("Instantiating callbacks...")
    callbacks: list[L.Callback] = [
        hydra.utils.instantiate(c) for c in cfg.callbacks.values()
    ]

    log.info(f"Instantiating trainer <{cfg.trainer._target_}>")
    trainer: L.Trainer = hydra.utils.instantiate(
        cfg.trainer, logger=wandb_logger, callbacks=callbacks
    )

    if cfg.get("train"):
        log.info("Starting training!")
        trainer.fit(model, datamodule, ckpt_path=cfg.get("ckpt_path"))

    if cfg.get("test"):
        log.info("Starting testing!")
        ckpt_path = "best" if cfg.get("train") else cfg.get("ckpt_path", "best")
        trainer.test(model=model, datamodule=datamodule, ckpt_path=ckpt_path)


if __name__ == "__main__":
    main()

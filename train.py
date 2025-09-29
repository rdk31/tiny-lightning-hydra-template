from typing import Union

import hydra
import lightning as L
import rootutils
import wandb
from lightning.pytorch.loggers import Logger
from omegaconf import DictConfig, OmegaConf

from src.utils import RankedLogger

rootutils.setup_root(__file__, pythonpath=True)


log = RankedLogger(__name__, rank_zero_only=True)


@hydra.main(version_base=None, config_path="config", config_name="default")
def main(cfg: DictConfig) -> None:
    if wandb.run:
        wandb.finish()

    L.seed_everything(cfg.core.seed, workers=True)

    log.info(f"Instantiating datamodule <{cfg.data._target_}>")
    datamodule: L.LightningDataModule = hydra.utils.instantiate(cfg.data)

    log.info(f"Instantiating model <{cfg.model._target_}>")
    model: L.LightningModule = hydra.utils.instantiate(cfg.model)

    callbacks: list[L.Callback] = []
    if cfg.get("callbacks"):
        log.info("Instantiating callbacks...")
        callbacks.extend([hydra.utils.instantiate(c) for c in cfg.callbacks.values()])

    logger: Union[Logger, bool] = False
    if cfg.get("logger"):
        log.info(f"Instantiating logger <{cfg.logger._target_}>")
        logger = hydra.utils.instantiate(cfg.logger)

        log.info("Logging hyperparameters!")
        logger.log_hyperparams(  # type: ignore
            OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True)  # type: ignore
        )
    else:
        log.info("Running without logger!")

    log.info(f"Instantiating trainer <{cfg.trainer._target_}>")
    trainer: L.Trainer = hydra.utils.instantiate(
        cfg.trainer, logger=logger, callbacks=callbacks
    )

    if cfg.get("train"):
        log.info("Starting training!")
        trainer.fit(model=model, datamodule=datamodule, ckpt_path=cfg.get("ckpt_path"))

    if cfg.get("test"):
        log.info("Starting testing!")
        ckpt_path = "best" if cfg.get("train") else cfg.get("ckpt_path", "best")
        trainer.test(model=model, datamodule=datamodule, ckpt_path=ckpt_path)


if __name__ == "__main__":
    main()

#!/usr/bin/env python


import hydra
import lightning as L
import rootutils
import wandb
from omegaconf import DictConfig, OmegaConf

rootutils.setup_root(__file__, pythonpath=True)


@hydra.main(version_base=None, config_path="config", config_name="default")
def main(cfg: DictConfig):
    if wandb.run:
        wandb.finish()

    L.seed_everything(cfg.core.seed)

    logger = hydra.utils.instantiate(cfg.logger)
    logger.log_hyperparams(OmegaConf.to_container(cfg, resolve=True))

    datamodule = hydra.utils.instantiate(cfg.data)
    model = hydra.utils.instantiate(cfg.model)
    callbacks = [hydra.utils.instantiate(c) for c in cfg.callbacks.values()]

    trainer = hydra.utils.instantiate(cfg.trainer, logger=logger, callbacks=callbacks)

    trainer.fit(model, datamodule)


if __name__ == "__main__":
    main()

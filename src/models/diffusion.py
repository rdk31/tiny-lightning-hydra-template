from collections.abc import Callable
from typing import Any

import lightning as L
import torch
from lightning.pytorch.utilities.types import OptimizerLRScheduler
from torch import Tensor
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler

from src.nets.diffusion import DiffusionEngine


class DiffusionLightningModule(L.LightningModule):
    def __init__(
        self,
        class_conditioning: bool,
        unet: torch.nn.Module,
        diffusion: DiffusionEngine,
        optimizer: Callable[..., Optimizer],
        lr_scheduler: Callable[..., LRScheduler] | None = None,
    ):
        super().__init__()
        self.save_hyperparameters(logger=False)

        self.class_conditioning = class_conditioning

        self.unet = unet
        self.diffusion = diffusion
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler

    def training_step(self, batch: dict[str, Any]) -> Tensor:
        x_0 = batch["image"]

        model_kwargs = {}
        if self.class_conditioning:
            model_kwargs["class_labels"] = batch["target"]

        output = self.diffusion.training_losses(
            self.unet, x_0, model_kwargs=model_kwargs
        )
        loss = output["loss"].mean()

        self.log_dict({"train/loss": loss}, sync_dist=True)

        return loss

    def validation_step(self, batch: dict[str, Any]) -> dict[str, Any] | None:
        x_0 = batch["image"]

        model_kwargs = {}
        if self.class_conditioning:
            model_kwargs["class_labels"] = batch["target"]

        pred_x_0 = self.diffusion.p_sample_loop(
            self.unet, noise=torch.randn_like(x_0), model_kwargs=model_kwargs
        )

        x_log = torch.cat([pred_x_0, x_0], dim=2)
        if self.class_conditioning:
            captions = [
                self.trainer.datamodule.val_dataset.idx_to_class[y]  # type: ignore
                for y in batch["target"].detach().cpu().numpy().tolist()
            ]
        else:
            captions = None

        return {
            "wandb_image_logger": {
                "val/samples": {
                    "images": x_log,
                    "denormalize_from": "standard",
                    "captions": captions,
                }
            }
        }

    def test_step(self, batch: dict[str, Any]) -> dict[str, Any] | None:
        x_0 = batch["image"]

        model_kwargs = {}
        if self.class_conditioning:
            model_kwargs["class_labels"] = batch["target"]

        pred_x_0 = self.diffusion.p_sample_loop(
            self.unet, noise=torch.randn_like(x_0), model_kwargs=model_kwargs
        )

        x_log = torch.cat([pred_x_0, x_0], dim=2)
        if self.class_conditioning:
            captions = [
                self.trainer.datamodule.test_dataset.idx_to_class[y]  # type: ignore
                for y in batch["target"].detach().cpu().numpy().tolist()
            ]
        else:
            captions = None

        return {
            "wandb_image_logger": {
                "test/samples": {
                    "images": x_log,
                    "denormalize_from": "standard",
                    "captions": captions,
                }
            }
        }

    def configure_optimizers(self) -> OptimizerLRScheduler:
        optimizer = self.optimizer(params=self.parameters())

        if self.lr_scheduler is None:
            return optimizer

        lr_scheduler = self.lr_scheduler(
            optimizer=optimizer,
            T_max=self.trainer.estimated_stepping_batches,
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": lr_scheduler,
                "interval": "step",
                "frequency": 1,
            },
        }

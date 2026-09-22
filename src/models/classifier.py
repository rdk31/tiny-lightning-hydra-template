from collections.abc import Callable
from typing import Any

import lightning as L
import torch
import torch.nn.functional as F
from lightning.pytorch.utilities.types import OptimizerLRScheduler
from torch import Tensor
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler
from torchmetrics import Accuracy


class ClassifierLightningModule(L.LightningModule):
    def __init__(
        self,
        num_classes: int,
        clf: torch.nn.Module,
        optimizer: Callable[..., Optimizer],
        lr_scheduler: Callable[..., LRScheduler] | None = None,
    ):
        super().__init__()
        self.save_hyperparameters(logger=False)

        self.clf = clf
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler

        self.train_acc = Accuracy(task="multiclass", num_classes=num_classes)
        self.val_acc = Accuracy(task="multiclass", num_classes=num_classes)
        self.test_acc = Accuracy(task="multiclass", num_classes=num_classes)

    def training_step(self, batch: dict[str, Any]) -> Tensor:
        x, y = batch["image"], batch["target"]

        logits = self.clf(x)

        loss = F.cross_entropy(logits, y)
        self.train_acc(logits, y)

        self.log_dict({"train/loss": loss, "train/acc": self.train_acc}, sync_dist=True)

        return loss

    def validation_step(self, batch: dict[str, Any]) -> dict[str, Any] | None:
        x, y = batch["image"], batch["target"]

        logits = self.clf(x)

        loss = F.cross_entropy(logits, y)
        self.val_acc(logits, y)

        self.log_dict({"val/loss": loss, "val/acc": self.val_acc}, sync_dist=True)

        preds = torch.argmax(logits, dim=1)
        captions = []
        for pred, label in zip(preds, y):
            idx_to_class = self.trainer.datamodule.val_dataset.idx_to_class  # type: ignore
            captions.append(
                f"predicted: {idx_to_class[pred.item()]} true: {idx_to_class[label.item()]}"
            )

        return {
            "wandb_image_logger": {
                "val/samples": {
                    "images": x,
                    "captions": captions,
                    "denormalize_from": "imagenet",
                }
            }
        }

    def test_step(self, batch: dict[str, Any]) -> dict[str, Any] | None:
        x, y = batch["image"], batch["target"]

        logits = self.clf(x)

        loss = F.cross_entropy(logits, y)
        self.test_acc(logits, y)

        self.log_dict({"test/loss": loss, "test/acc": self.test_acc}, sync_dist=True)

        preds = torch.argmax(logits, dim=1)
        captions = []
        for pred, label in zip(preds, y):
            idx_to_class = self.trainer.datamodule.test_dataset.idx_to_class  # type: ignore
            captions.append(
                f"predicted: {idx_to_class[pred.item()]} true: {idx_to_class[label.item()]}"
            )

        return {
            "wandb_image_logger": {
                "test/samples": {
                    "images": x,
                    "captions": captions,
                    "denormalize_from": "imagenet",
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

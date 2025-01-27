import lightning as L
import torch
import torch.nn as nn
import torchmetrics


class ClassifierLightningModule(L.LightningModule):
    def __init__(
        self,
        num_classes: int,
        clf: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        lr_scheduler: torch.optim.lr_scheduler.LRScheduler | None = None,
    ):
        super().__init__()
        self.save_hyperparameters(logger=False)

        self.clf = clf
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler

        self.criterion = nn.CrossEntropyLoss()
        self.train_acc = torchmetrics.Accuracy(
            task="multiclass", num_classes=num_classes
        )
        self.val_acc = torchmetrics.Accuracy(task="multiclass", num_classes=num_classes)
        self.test_acc = torchmetrics.Accuracy(
            task="multiclass", num_classes=num_classes
        )

    def training_step(self, batch, batch_idx: int):
        x, y = batch["image"], batch["target"]

        logits = self.clf(x)

        loss = self.criterion(logits, y)
        self.train_acc(logits, y)

        self.log_dict(
            {"train/loss": loss, "train/acc": self.train_acc},
            on_step=True,
        )

        return loss

    def validation_step(self, batch, batch_idx: int):
        x, y = batch["image"], batch["target"]

        logits = self.clf(x)

        loss = self.criterion(logits, y)
        self.val_acc(logits, y)

        self.log_dict(
            {"val/loss": loss, "val/acc": self.val_acc},
            on_epoch=True,
        )

        preds = torch.argmax(logits, dim=1)
        captions = []
        for pred, label in zip(preds, y):
            captions.append(f"pred: {pred} true: {label}")

        return {
            "wandb_image_logger": {
                "val/samples": {
                    "images": x,
                    "captions": captions,
                    "denormalize_from": "imagenet",
                }
            }
        }

    def test_step(self, batch, batch_idx: int):
        x, y = batch["image"], batch["target"]

        logits = self.clf(x)

        loss = self.criterion(logits, y)
        self.test_acc(logits, y)

        self.log_dict(
            {"test/loss": loss, "test/acc": self.test_acc},
            on_epoch=True,
        )

        preds = torch.argmax(logits, dim=1)
        captions = []
        for pred, label in zip(preds, y):
            captions.append(f"pred: {pred} true: {label}")

        return {
            "wandb_image_logger": {
                "test/samples": {
                    "images": x,
                    "captions": captions,
                    "denormalize_from": "imagenet",
                }
            }
        }

    def configure_optimizers(self):
        optimizer = self.optimizer(params=self.parameters())
        out = {"optimizer": optimizer}
        if self.lr_scheduler is not None:
            lr_scheduler = self.lr_scheduler(
                optimizer=optimizer,
                T_max=self.trainer.estimated_stepping_batches,
            )
            out["lr_scheduler"] = {
                "scheduler": lr_scheduler,
                "interval": "step",
                "frequency": 1,
            }

        return out

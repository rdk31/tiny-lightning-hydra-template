import lightning as L
import torch
import torch.nn as nn
import torchmetrics


class LightningModule(L.LightningModule):
    def __init__(
        self,
        num_classes: int,
        net: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        lr_scheduler: torch.optim.lr_scheduler.LRScheduler = None,
    ):
        super().__init__()
        self.save_hyperparameters(logger=False)

        self.net = net
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
        x, y = batch

        logits = self.net(x)

        loss = self.criterion(logits, y)
        self.train_acc(logits, y)

        self.log_dict(
            {"train/loss": loss, "train/acc": self.train_acc},
            on_step=True,
        )

        return loss

    def validation_step(self, batch, batch_idx: int):
        x, y = batch

        logits = self.net(x)

        loss = self.criterion(logits, y)
        self.val_acc(logits, y)

        self.log_dict(
            {"val/loss": loss, "val/acc": self.val_acc},
            on_epoch=True,
        )

    def test_step(self, batch, batch_idx: int):
        x, y = batch

        logits = self.net(x)

        loss = self.criterion(logits, y)
        self.test_acc(logits, y)

        self.log_dict(
            {"test/loss": loss, "test/acc": self.test_acc},
            on_epoch=True,
        )

    def configure_optimizers(self):
        optimizer = self.optimizer(params=self.parameters())
        if self.lr_scheduler is not None:
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
        return {"optimizer": optimizer}

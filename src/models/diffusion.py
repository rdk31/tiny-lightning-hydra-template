import lightning as L
import torch

from src.nets.diffusion import DiffusionEngine


class DiffusionLightningModule(L.LightningModule):
    def __init__(
        self,
        class_conditioning: bool,
        unet: torch.nn.Module,
        diffusion: DiffusionEngine,
        optimizer: torch.optim.Optimizer,
        lr_scheduler: torch.optim.lr_scheduler.LRScheduler | None = None,
    ):
        super().__init__()
        self.save_hyperparameters(logger=False)

        self.class_conditioning = class_conditioning

        self.unet = unet
        self.diffusion = diffusion
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler

    def training_step(self, batch, batch_idx: int):
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

    def validation_step(self, batch, batch_idx: int):
        x_0 = batch["image"]

        model_kwargs = {}
        if self.class_conditioning:
            model_kwargs["class_labels"] = batch["target"]

        pred_x_0 = self.diffusion.p_sample_loop(
            self.unet, shape=x_0.shape, model_kwargs=model_kwargs
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

    def test_step(self, batch, batch_idx: int):
        x_0 = batch["image"]

        model_kwargs = {}
        if self.class_conditioning:
            model_kwargs["class_labels"] = batch["target"]

        pred_x_0 = self.diffusion.p_sample_loop(
            self.unet, shape=x_0.shape, model_kwargs=model_kwargs
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

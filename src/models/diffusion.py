import lightning as L
import torch

from src.nets.diffusion import DiffusionEngine


class DiffusionLightningModule(L.LightningModule):
    def __init__(
        self,
        unet: torch.nn.Module,
        diffusion: DiffusionEngine,
        optimizer: torch.optim.Optimizer,
        lr_scheduler: torch.optim.lr_scheduler.LRScheduler | None = None,
    ):
        super().__init__()
        self.save_hyperparameters(logger=False)

        self.unet = unet
        self.diffusion = diffusion
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler

    def training_step(self, batch, batch_idx: int):
        x_0, y = batch["image"], batch["target"]

        output = self.diffusion.training_losses(
            self.unet, x_0, model_kwargs={"class_labels": y}
        )
        loss = output["loss"].mean()

        self.log_dict({"train/loss": loss}, sync_dist=True)

        return loss

    @torch.no_grad()
    def validation_step(self, batch, batch_idx: int):
        x_0, y = batch["image"], batch["target"]

        pred_x_0 = self.diffusion.p_sample_loop(
            self.unet, shape=x_0.shape, model_kwargs={"class_labels": y}
        )

        x_log = torch.cat([pred_x_0, x_0], dim=2)

        return {
            "wandb_image_logger": {
                "val/samples": {
                    "images": x_log,
                    "denormalize_from": "standard",
                }
            }
        }

    @torch.no_grad()
    def test_step(self, batch, batch_idx: int):
        x_0, y = batch["image"], batch["target"]

        pred_x_0 = self.diffusion.p_sample_loop(
            self.unet, shape=x_0.shape, model_kwargs={"class_labels": y}
        )

        x_log = torch.cat([pred_x_0, x_0], dim=2)

        return {
            "wandb_image_logger": {
                "test/samples": {
                    "images": x_log,
                    "denormalize_from": "standard",
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

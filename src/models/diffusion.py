import lightning as L
import torch
from torchmetrics.image import PeakSignalNoiseRatio as PSNR
from torchmetrics.image import StructuralSimilarityIndexMeasure as SSIM

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

        self.val_psnr = PSNR()
        self.val_ssim = SSIM()

        self.test_ssim = SSIM()
        self.test_psnr = PSNR()

    def training_step(self, batch, batch_idx: int):
        x_T, x_0 = batch["corrupted"], batch["image"]

        output = self.diffusion.training_losses(self.unet, x_0, x_T)

        self.log_dict({"train/loss": output["loss"]}, sync_dist=True)

        return output["loss"]

    @torch.no_grad()
    def validation_step(self, batch, batch_idx: int):
        x_T, x_0 = batch["corrupted"], batch["image"]

        pred_x_0 = self.diffusion.p_sample_loop(self.unet, shape=x_0.shape)
        pred_x_0 = pred_x_0.clamp(-1, 1)

        psnr = self.val_psnr(pred_x_0, x_0)
        ssim = self.val_ssim(pred_x_0, x_0)

        self.log_dict({"val/psnr": psnr, "val/ssim": ssim}, sync_dist=True)

        x_log = torch.cat([x_T, pred_x_0, x_0], dim=2)

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
        x_T, x_0 = batch["corrupted"], batch["image"]

        pred_x_0 = self.diffusion.p_sample_loop(self.unet, shape=x_0.shape)
        pred_x_0 = pred_x_0.clamp(-1, 1)

        psnr = self.test_psnr(pred_x_0, x_0)
        ssim = self.test_ssim(pred_x_0, x_0)

        self.log_dict({"test/psnr": psnr, "test/ssim": ssim}, sync_dist=True)

        x_log = torch.cat([x_T, pred_x_0, x_0], dim=2)

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

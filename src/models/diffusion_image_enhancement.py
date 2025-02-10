import lightning as L
import torch
from torchmetrics.image import PeakSignalNoiseRatio as PSNR
from torchmetrics.image import StructuralSimilarityIndexMeasure as SSIM
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity as LPIPS

from src.nets.diffusion import DiffusionEngine


class ImageEnhancementLightningModule(L.LightningModule):
    def __init__(
        self,
        unet: torch.nn.Module,
        diffusion: DiffusionEngine,
        optimizer: torch.optim.Optimizer,
        vae: torch.nn.Module | None = None,
        lr_scheduler: torch.optim.lr_scheduler.LRScheduler | None = None,
    ):
        super().__init__()
        self.save_hyperparameters(logger=False)

        self.unet = unet
        self.diffusion = diffusion

        if vae:
            self.vae = vae.eval()
        else:
            self.vae = None  # type: ignore

        self.val_psnr = PSNR()
        self.val_ssim = SSIM()
        self.val_lpips = LPIPS().eval()

        self.test_psnr = PSNR()
        self.test_ssim = SSIM()
        self.test_lpips = LPIPS().eval()

        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler

    def forward(self, x_T):
        x_T = 2 * x_T - 1

        z_T = self.vae.encode(x_T) if self.vae else x_T

        pred_z_0 = self.diffusion.p_sample_loop(self.unet, z_T)

        return self.vae.decode(pred_z_0) if self.vae else pred_z_0.clamp(-1, 1)

    def training_step(self, batch, batch_idx: int):
        x_T, x_0 = batch["corrupted"], batch["image"]

        x_T = 2 * x_T - 1
        x_0 = 2 * x_0 - 1

        if self.vae:
            z_T = self.vae.encode(x_T)
            z_0 = self.vae.encode(x_0)
        else:
            z_T = x_T
            z_0 = x_0

        output = self.diffusion.training_losses(self.unet, z_0, z_T)

        self.log("train/loss", output["loss"], on_epoch=False, on_step=True)

        return output["loss"]

    def validation_step(self, batch, batch_idx: int):
        x_T, x_0 = batch["corrupted"], batch["image"]

        norm_x_0 = 2 * x_0 - 1

        pred_x_0 = self.forward(x_T)

        psnr = self.val_psnr(pred_x_0, norm_x_0)
        ssim = self.val_ssim(pred_x_0, norm_x_0)
        lpips = self.val_lpips(pred_x_0, norm_x_0)

        self.log("val/psnr", psnr, on_epoch=True, on_step=False)
        self.log("val/ssim", ssim, on_epoch=True, on_step=False)
        self.log("val/lpips", lpips, on_epoch=True, on_step=False)

        pred_x_0 = (pred_x_0 + 1) / 2

        x_log = torch.cat([x_T, pred_x_0, x_0], dim=2)  # input, output, target

        return {"wandb_image_logger": {"val/samples": {"images": x_log}}}

    def test_step(self, batch, batch_idx: int):
        x_T, x_0 = batch["corrupted"], batch["image"]

        norm_x_0 = 2 * x_0 - 1

        pred_x_0 = self.forward(x_T)

        psnr = self.test_psnr(pred_x_0, norm_x_0)
        ssim = self.test_ssim(pred_x_0, norm_x_0)
        lpips = self.test_lpips(pred_x_0, norm_x_0)

        self.log("test/psnr", psnr, on_epoch=True, on_step=False)
        self.log("test/ssim", ssim, on_epoch=True, on_step=False)
        self.log("test/lpips", lpips, on_epoch=True, on_step=False)

        pred_x_0 = (pred_x_0 + 1) / 2

        x_log = torch.cat([x_T, pred_x_0, x_0], dim=2)  # input, output, target

        return {"wandb_image_logger": {"test/samples": {"images": x_log}}}

    def configure_optimizers(self):
        optimizer = self.optimizer(params=self.unet.parameters())
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

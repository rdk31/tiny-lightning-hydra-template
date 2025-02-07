import torch
from torch import nn


class DiffusionEngine(nn.Module):
    def __init__(
        self,
        train_diffusion,
        val_diffusion,
    ):
        super().__init__()

        self.train_diffusion = train_diffusion
        self.val_diffusion = val_diffusion

    @torch.no_grad()
    def p_sample_loop(self, unet, *args, **kwargs):
        return self.val_diffusion.p_sample_loop(unet, *args, **kwargs)

    def training_losses(self, unet, x0, *args, **kwargs):
        return self.train_diffusion.training_losses(unet, x0, *args, **kwargs)

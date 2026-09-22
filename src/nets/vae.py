import torch
from diffusers.models.autoencoders.autoencoder_kl import AutoencoderKL
from torch import nn


class VAE(nn.Module):
    def __init__(self, model_id: str = "stabilityai/stable-diffusion-x4-upscaler"):
        super().__init__()
        self.backbone = AutoencoderKL.from_pretrained(model_id, subfolder="vae")
        self.scaling_factor = self.backbone.config.scaling_factor
        self.latent_channels = self.backbone.config.latent_channels

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        output = self.backbone.encode(x).latent_dist.mode()
        return output * self.scaling_factor

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        out = self.backbone.decode(z / self.scaling_factor).sample
        return out.clamp(-1, 1)

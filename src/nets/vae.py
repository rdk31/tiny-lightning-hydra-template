import torch.nn as nn
from diffusers.models.autoencoders import AutoencoderKL


class VAE(nn.Module):
    def __init__(self, model_id="stabilityai/stable-diffusion-x4-upscaler"):
        super(VAE, self).__init__()
        self.backbone = AutoencoderKL.from_pretrained(model_id, subfolder="vae")
        self.scaling_factor = self.backbone.config.scaling_factor
        self.latent_channels = self.backbone.config.latent_channels

    def encode(self, x):
        output = self.backbone.encode(x).latent_dist.mode()
        return output * self.scaling_factor

    def decode(self, z):
        out = self.backbone.decode(z / self.scaling_factor).sample
        return out.clamp(-1, 1)

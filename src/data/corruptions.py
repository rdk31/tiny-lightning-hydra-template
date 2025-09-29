import torch
from torchvision.transforms.v2 import functional as F


class LowResolution:
    def __init__(self, factor=4):
        self.factor = factor

    def __call__(self, img: torch.Tensor, return_small=False):
        _, h, w = img.shape
        small = F.resize(img, (h // self.factor, w // self.factor))
        upscaled = F.resize(small, (h, w))

        if return_small:
            return upscaled, small
        return upscaled

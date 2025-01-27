from typing import Callable

from torch.utils.data import Dataset
from torchvision.transforms.v2 import functional as F


class LowResolution:
    def __init__(self, factor=4):
        self.factor = factor

    def __call__(self, img, return_small=False):
        _, h, w = img.shape
        small = F.resize(img, (h // self.factor, w // self.factor))
        upscaled = F.resize(small, (h, w))

        if return_small:
            return upscaled, small
        return upscaled


class CorruptedDataset(Dataset):
    def __init__(self, dataset: Dataset, corruption: Callable):
        self.dataset = dataset
        self.corruption = corruption

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        outs = self.dataset[idx]

        outs["corrupted"] = self.corruption(outs["image"])

        return outs

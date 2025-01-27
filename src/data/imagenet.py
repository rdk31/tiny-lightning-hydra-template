from typing import Callable

from torchvision import datasets


class ImageNet(datasets.ImageNet):
    def __init__(self, root: str, split: str, transform: Callable):
        super().__init__(root, split, transform=transform)

    def __getitem__(self, index: int):
        sample, target = super().__getitem__(index)

        return {
            "image": sample,
            "target": target,
        }

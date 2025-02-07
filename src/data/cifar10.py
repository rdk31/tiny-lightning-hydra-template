from typing import Callable

from torchvision import datasets


class CIFAR10(datasets.CIFAR10):
    def __init__(self, root: str, train: bool, transform: Callable):
        super().__init__(root, train, transform=transform, download=True)

    def __getitem__(self, index: int):
        sample, target = super().__getitem__(index)

        return {
            "image": sample,
            "target": target,
        }

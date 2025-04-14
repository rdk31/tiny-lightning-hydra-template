from typing import Any, Callable

from torchvision import datasets


class CIFAR10(datasets.CIFAR10):
    def __init__(
        self,
        root: str,
        train: bool,
        transform: Callable,
    ):
        super().__init__(root, train, transform=transform, download=True)
        self.idx_to_class = {idx: name for name, idx in self.class_to_idx.items()}

    def __getitem__(self, index: int) -> dict[str, Any]:
        sample, target = super().__getitem__(index)

        return {
            "image": sample,
            "target": target,
        }

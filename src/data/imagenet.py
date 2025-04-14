from typing import Any, Callable

from torchvision import datasets


class ImageNet(datasets.ImageNet):
    def __init__(
        self,
        root: str,
        split: str,
        transform: Callable,
    ):
        super().__init__(root, split, transform=transform)
        self.idx_to_class = {idx: name for name, idx in self.class_to_idx.items()}

    def __getitem__(self, index: int) -> dict[str, Any]:
        sample, target = super().__getitem__(index)

        return {
            "image": sample,
            "target": target,
        }

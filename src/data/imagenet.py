from typing import Any, Callable, Optional

import lightning as L
from torch.utils.data import DataLoader
from torchvision import datasets


class ImageNet(datasets.ImageNet):
    def __init__(
        self,
        root: str,
        split: str,
        transform: Callable,
        corruption: Optional[Callable] = None,
    ):
        super().__init__(root, split, transform=transform)
        self.idx_to_class = {idx: name for name, idx in self.class_to_idx.items()}
        self.corruption = corruption

    def __getitem__(self, index: int) -> dict[str, Any]:
        sample, target = super().__getitem__(index)

        out = {
            "image": sample,
            "target": target,
        }

        if self.corruption is not None:
            out["corrupted"] = self.corruption(sample)

        return out


class ImageNetDataModule(L.LightningDataModule):
    def __init__(
        self,
        root: str,
        train_transform: Callable,
        val_transform: Callable,
        corruption: Optional[Callable] = None,
        batch_size: int = 32,
        num_workers: int = 8,
        pin_memory: bool = True,
    ):
        super().__init__()

        self.root = root
        self.corruption = corruption
        self.train_transform = train_transform
        self.val_transform = val_transform

        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory

    def setup(self, stage=None) -> None:
        if stage == "fit" or stage is None:
            self.train_dataset = ImageNet(
                root=self.root,
                split="train",
                transform=self.train_transform,
                corruption=self.corruption,
            )
            self.val_dataset = ImageNet(
                root=self.root,
                split="val",
                transform=self.val_transform,
                corruption=self.corruption,
            )

        if stage == "test" or stage is None:
            self.test_dataset = ImageNet(
                root=self.root,
                split="val",
                transform=self.val_transform,
                corruption=self.corruption,
            )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            shuffle=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
        )

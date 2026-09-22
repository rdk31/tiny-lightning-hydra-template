from collections.abc import Callable
from typing import Any

import lightning as L
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets


class ImageNet(Dataset[dict[str, Any]]):
    def __init__(
        self,
        root: str,
        split: str,
        transform: Callable,
        corruption: Callable | None = None,
    ) -> None:
        self.dataset = datasets.ImageNet(root, split, transform=transform)

        self.class_to_idx = self.dataset.class_to_idx
        self.idx_to_class = {idx: name for name, idx in self.class_to_idx.items()}
        self.corruption = corruption

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int) -> dict[str, Any]:
        sample, target = self.dataset[index]

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
        num_classes: int,
        train_transform: Callable,
        val_transform: Callable,
        corruption: Callable | None = None,
        batch_size: int = 32,
        num_workers: int = 8,
        pin_memory: bool = True,
    ):
        super().__init__()

        self.root = root
        self.num_classes = num_classes
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

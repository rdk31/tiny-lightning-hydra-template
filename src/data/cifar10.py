from typing import Any, Callable

import lightning as L
from torch.utils.data import DataLoader
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


class CIFAR10DataModule(L.LightningDataModule):
    def __init__(
        self,
        num_classes: int,
        root: str,
        train_transform: Callable,
        val_transform: Callable,
        batch_size: int = 32,
        num_workers: int = 8,
        pin_memory: bool = True,
    ):
        super().__init__()

        self.root = root
        self.train_transform = train_transform
        self.val_transform = val_transform

        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory

    def setup(self, stage=None) -> None:
        self.train_dataset = CIFAR10(
            root=self.root,
            train=True,
            transform=self.train_transform,
        )
        self.val_dataset = CIFAR10(
            root=self.root,
            train=False,
            transform=self.val_transform,
        )
        self.test_dataset = CIFAR10(
            root=self.root,
            train=False,
            transform=self.val_transform,
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

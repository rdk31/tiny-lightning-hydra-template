import lightning as L
import torch
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import v2


class CIFAR10DataModule(L.LightningDataModule):
    def __init__(
        self,
        data_dir: str,
        num_classes: int,
        train_transform: v2.Compose,
        val_transform: v2.Compose | None = None,
        batch_size: int = 512,
        num_workers: int = 8,
    ):
        super().__init__()
        self.train_transform = train_transform
        if val_transform:
            self.val_transform = val_transform
        else:
            self.val_transform = v2.Compose(
                [
                    v2.ToImage(),
                    v2.ToDtype(torch.float32, scale=True),
                    v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ]
            )
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers

    def prepare_data(self):
        datasets.CIFAR10(root=self.data_dir, train=True, download=True)
        datasets.CIFAR10(root=self.data_dir, train=False, download=True)

    def setup(self, stage=None):
        if stage == "fit" or stage is None:
            self.train_dataset = datasets.CIFAR10(
                root=self.data_dir, train=True, transform=self.train_transform
            )
            self.val_dataset = datasets.CIFAR10(
                root=self.data_dir, train=False, transform=self.val_transform
            )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=True,
        )

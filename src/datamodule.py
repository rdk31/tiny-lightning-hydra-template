import lightning as L
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


class CIFAR10DataModule(L.LightningDataModule):
    def __init__(
        self,
        data_dir: str,
        train_transform: torchvision.transforms.Compose,
        batch_size: int = 32,
        num_workers: int = 8,
    ):
        super().__init__()
        self.train_transform = train_transform
        self.val_transform = transforms.ToTensor()
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

import pathlib
from typing import Callable, Literal

import lightning as L
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets

# tested for torchvision.models.ResNet101_Weights.DEFAULT
IMAGENETTE_TO_IMAGE_NET = {
    0: 0,
    1: 217,
    2: 482,
    3: 491,
    4: 497,
    5: 566,
    6: 569,
    7: 571,
    8: 574,
    9: 701,
}


class Imagenette(Dataset):
    def __init__(
        self,
        root: str,
        split: Literal["train", "val"],
        transform: Callable,
        map_targets_to_image_net: bool = False,
    ):
        self.map_targets_to_image_net = map_targets_to_image_net
        self.dataset = datasets.Imagenette(root=root, split=split, transform=transform)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index: int):
        x, y = self.dataset[index]

        if self.map_targets_to_image_net:
            y = IMAGENETTE_TO_IMAGE_NET[y]

        return x, y


class ImagenetteDataModule(L.LightningDataModule):
    def __init__(
        self,
        data_dir: str,
        train_transform: Callable,
        val_transform: Callable,
        map_targets_to_image_net: bool = False,
        batch_size: int = 32,
        num_workers: int = 8,
    ):
        super().__init__()
        self.data_dir = data_dir
        self.train_transform = train_transform
        self.val_transform = val_transform
        self.map_targets_to_image_net = map_targets_to_image_net
        self.batch_size = batch_size
        self.num_workers = num_workers

    def prepare_data(self):
        if not pathlib.Path(self.data_dir, "imagenette2").exists():
            datasets.Imagenette(root=self.data_dir, split="train", download=True)

    def setup(self, stage=None):
        if stage == "fit" or stage is None:
            self.train_dataset = Imagenette(
                root=self.data_dir,
                split="train",
                transform=self.train_transform,
                map_targets_to_image_net=self.map_targets_to_image_net,
            )
            self.val_dataset = Imagenette(
                root=self.data_dir,
                split="val",
                transform=self.val_transform,
                map_targets_to_image_net=self.map_targets_to_image_net,
            )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=True,
            pin_memory=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=True,
        )

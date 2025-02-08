import lightning as L
from torch.utils.data import DataLoader, Dataset


class DataModule(L.LightningDataModule):
    def __init__(
        self,
        num_classes: int,
        train_dataset: Dataset,
        val_dataset: Dataset,
        test_dataset: Dataset | None = None,
        batch_size: int = 32,
        num_workers: int = 8,
        pin_memory: bool = True,
    ):
        super().__init__()

        self.train_dataset_factory = train_dataset
        self.val_dataset_factory = val_dataset
        if test_dataset is not None:
            self.test_dataset_factory = test_dataset
        else:
            self.test_dataset_factory = val_dataset

        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory

    def setup(self, stage=None):
        if stage == "fit" or stage is None:
            self.train_dataset = self.train_dataset_factory()
            self.val_dataset = self.val_dataset_factory()
        elif stage == "validate":
            self.val_dataset = self.val_dataset_factory()

        if stage == "test" or stage is None:
            self.test_dataset = self.test_dataset_factory()

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

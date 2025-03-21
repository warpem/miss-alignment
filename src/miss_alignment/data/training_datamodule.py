from os import PathLike
from pathlib import Path
from copy import deepcopy

import pytorch_lightning as pl
from torch.utils.data import DataLoader, random_split

from .training_dataset import EMDBDataset


class EMDBDataModule(pl.LightningDataModule):
    """
    PyTorch Lightning DataModule for MRC data.

    Parameters
    ----------
    dataset_directory : PathLike
        Directory containing MRC files.
    rng : torch.Generator
        Random seed for reproducibility.
    batch_size : int
        Batch size for dataloaders.
    target_size : int or tuple
        Target size to pad/crop volumes to.
    train_val_split : tuple
        Fractions for train, validation, and test splits.
    num_workers : int
        Number of workers for DataLoader.
    """

    def __init__(
        self,
        dataset_directory: PathLike,
        batch_size: int = 4,
        target_size: int = 64,
        train_val_split: tuple = (
            0.7,
            0.3,
        ),
        num_workers: int = 4,
    ):
        super().__init__()
        self.dataset_directory = Path(dataset_directory)
        self.batch_size = batch_size
        self.target_size = target_size
        self.train_val_split = train_val_split
        self.num_workers = num_workers

        # Verify split fractions sum to 1
        if sum(train_val_split) != 1.0:
            raise ValueError(
                "Train, validation, and test split fractions must sum to 1"
            )

        self.train_dataset, self.val_dataset, self.test_dataset = None, None, None

    def prepare_data(self) -> None:
        EMDBDataset(self.dataset_directory, self.target_size)

    def setup(self, stage: str | None = None):
        """
        Setup datasets for training, validation, and testing.

        Parameters
        ----------
        stage : str, optional
            Stage to setup ('fit', 'validate', 'test', or 'predict').
        """
        if stage == "fit":
            full_dataset = EMDBDataset(self.dataset_directory)
            n_total = len(full_dataset)
            n_train = int(self.train_val_split[0] * n_total)
            n_val = n_total - n_train
            self.train_dataset, self.val_dataset = random_split(
                full_dataset,
                [n_train, n_val],  # generator=self.rng
            )
            self.val_dataset = deepcopy(self.val_dataset)
            self.train_dataset.dataset.train()
            self.val_dataset.dataset.eval()

    def train_dataloader(self):
        """
        Return DataLoader for training data.
        """
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            drop_last=True,
            num_workers=self.num_workers,
            pin_memory=True,
        )

    def val_dataloader(self):
        """
        Return DataLoader for validation data.
        """
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            drop_last=False,
            num_workers=self.num_workers,
            pin_memory=True,
        )

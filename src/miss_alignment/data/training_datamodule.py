import os
import glob
import numpy as np
import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader
from typing import Optional

from training_dataset import MRCDataset


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)


class MRCDataModule(pl.LightningDataModule):
    """
    PyTorch Lightning DataModule for MRC data.

    Parameters
    ----------
    data_dir : str
        Directory containing MRC files.
    rng : torch.Generator
        Random seed for reproducibility.
    batch_size : int
        Batch size for dataloaders.
    target_size : int or tuple
        Target size to pad/crop volumes to.
    max_rotation_angle : float
        Maximum rotation angle in degrees for random rotations.
    translation_std : float
        Standard deviation for Gaussian random translations.
    train_val_test_split : tuple
        Fractions for train, validation, and test splits.
    num_workers : int
        Number of workers for DataLoader.
    """

    def __init__(
        self,
        data_dir: str,
        rng: torch.Generator,
        batch_size: int = 8,
        target_size: int = 64,
        train_val_test_split: tuple = (0.8, 0.1, 0.1),
        num_workers: int = 4,
    ):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.target_size = target_size
        self.train_val_test_split = train_val_test_split
        self.num_workers = num_workers
        self.rng = rng

        # Verify split fractions sum to 1
        if sum(train_val_test_split) != 1.0:
            raise ValueError(
                "Train, validation, and test split fractions must sum to 1"
            )

        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None

    def prepare_data(self):
        """
        Check if data exists. This method is called only on 1 GPU.
        """
        if not os.path.exists(self.data_dir):
            raise ValueError(f"Data directory {self.data_dir} does not exist")

    def setup(self, stage: Optional[str] = None):
        """
        Setup datasets for training, validation, and testing.

        Parameters
        ----------
        stage : str, optional
            Stage to setup ('fit', 'validate', 'test', or 'predict').
        """
        # Find all MRC files
        mrc_files = sorted(glob.glob(os.path.join(self.data_dir, "*.mrc")))

        if len(mrc_files) == 0:
            raise ValueError(f"No MRC files found in {self.data_dir}")

        # Calculate split sizes
        n_train = int(len(mrc_files) * self.train_val_test_split[0])
        n_val = int(len(mrc_files) * self.train_val_test_split[1])

        # Split files
        train_files = mrc_files[:n_train]
        val_files = mrc_files[n_train : n_train + n_val]
        test_files = mrc_files[n_train + n_val :]

        # Create datasets
        if stage == "fit" or stage is None:
            self.train_dataset = MRCDataset(
                train_files,
                target_size=self.target_size,
            )

            self.val_dataset = MRCDataset(
                val_files,
                target_size=self.target_size,
            )

        if stage == "test" or stage is None:
            self.test_dataset = MRCDataset(
                test_files,
                target_size=self.target_size,
            )

    def train_dataloader(self):
        """
        Return DataLoader for training data.
        """
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
            worker_init_fn=seed_worker,
            generator=self.rng,
        )

    def val_dataloader(self):
        """
        Return DataLoader for validation data.
        """
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            worker_init_fn=seed_worker,
            generator=self.rng,
        )

    def test_dataloader(self):
        """
        Return DataLoader for test data.
        """
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            worker_init_fn=seed_worker,
            generator=self.rng,
        )

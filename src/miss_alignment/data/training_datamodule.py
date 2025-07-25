from os import PathLike
from pathlib import Path
from collections.abc import Callable

import pytorch_lightning as pl
from torch.utils.data import DataLoader

from miss_alignment.data.training_dataset import SHRECDataset


class SHRECDataModule(pl.LightningDataModule):
    """
    Training datamodule assumes the following directory layout:

    dataset_directory/
    +- iter0/  # this is the initial state
    ¦  +- model_0.pickle  # stores a Tomogram object as a pickle
    ¦  +- model_1.pickle
    ¦  +- ...
    +- iter1/  # this is where the updated alignments are stored
    ¦  +- model_0.pickle
    etc...

    Parameters
    ----------
    dataset_directory
    batch_size
    num_workers
    target_size
    patches_per_tomogram
    training_iteration
    shift_generator
    """
    def __init__(
        self,
        dataset_directory: PathLike,
        shift_generator: Callable,
        batch_size: int = 4,
        num_workers: int = 4,
        target_size: int = 64,
        patches_per_tomogram: int = 1000,
        training_iteration: int = 0,
    ):

        super().__init__()
        # data module controls
        self.batch_size = batch_size
        self.num_workers = num_workers

        # dataset controls
        self.dataset_directory = Path(dataset_directory)
        self.training_iteration = training_iteration
        self.target_size = target_size
        self.patches_per_tomogram = patches_per_tomogram
        # function that generates shifts in 3D
        self.shift_generator = shift_generator

        self.train_dataset, self.predict_dataset = (None, None)

    @property
    def dataset_directory_at_iteration(self):
        return self.dataset_directory / f"iter{self.training_iteration}"

    def prepare_data(self):
        # this is not done in the setup() because lightning specifically
        # calls this function with a single process to prevent race conditions
        SHRECDataset(
            self.dataset_directory, self.shift_generator, download=True
        )

    def setup(self, stage=None):
        """No stage for setup because this is used just for training."""
        if stage is not None:
            raise ValueError(
                "stage parameter is not supported by MissAlignment"
            )
        self.train_dataset = SHRECDataset(
            self.dataset_directory_at_iteration,
            self.shift_generator,
            target_size=self.target_size,
            patches_per_tomogram=self.patches_per_tomogram,
            train=True,
        )

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
            persistent_workers=True,
            pin_memory=True,
        )

    def align_dataset(self, model, reinitialize: bool = True):
        """
        Return DataLoader for prediction
        """
        for tilt_series in self.train_dataset.tomograms:
            # run alignment
            # store alignment to next iteration folder
            continue
        # update iteration
        # store state?
        # if reinitialize: rerun setup?

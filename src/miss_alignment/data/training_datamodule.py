from os import PathLike
from pathlib import Path
from collections.abc import Callable

import lightning.pytorch as pl
from torch.utils.data import DataLoader

from miss_alignment.data.training_dataset import SHRECDataset
from miss_alignment.alignment import evaluate_tilt_series
from miss_alignment.data.io import read_tomogram_from_pickle


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
        num_workers: int = 4,
        batch_size: int = 4,
        target_size: int = 64,
        loss_metric_steps: int = 1000,
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
        self.samples_per_epoch = loss_metric_steps * batch_size
        # function that generates shifts in 3D
        self.shift_generator = shift_generator

        self.train_dataset, self.predict_dataset = (None, None)

    @property
    def dataset_directory_at_iteration(self):
        return self.dataset_directory / f"iter{self.training_iteration}"

    def prepare_data(self):
        # this is not done in the setup() because lightning specifically
        # calls this function with a single process to prevent race conditions
        if self.training_iteration == 0:
            SHRECDataset(
                self.dataset_directory_at_iteration, self.shift_generator, download=True
            )

    def setup(self, stage: str = "fit"):
        """No stage for setup because this is used just for training."""
        if stage != "fit":
            raise ValueError(f"{stage} is not a valid stage")
        self.train_dataset = SHRECDataset(
            self.dataset_directory_at_iteration,
            self.shift_generator,
            target_size=self.target_size,
            samples_per_epoch=self.samples_per_epoch,
            train=True,
        )

    def train_dataloader(self):
        """Return DataLoader for training data."""
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            drop_last=True,
            num_workers=self.num_workers,
            persistent_workers=True,
            pin_memory=False,
        )

    def align_dataset(self, model, patches_per_dim, patch_size, ground_truth_dir):
        """Align the tomograms in the dataset with the model."""
        output_directory = self.dataset_directory / f"iter{self.training_iteration + 1}"
        output_directory.mkdir(parents=True, exist_ok=True)
        for file_path, tilt_series in self.train_dataset.tomos:
            tilt_series_name = file_path.stem
            tilt_series_ground_truth = read_tomogram_from_pickle(
                ground_truth_dir / f"{tilt_series_name}.pickle"
            )
            # results are written to the output_directory
            evaluate_tilt_series(
                model,
                tilt_series_name,
                tilt_series,
                patches_per_dim,
                patch_size,
                (180, 512, 512),
                output_directory,
                tilt_series_ground_truth=tilt_series_ground_truth,
                device="cpu",
            )
        self.training_iteration += 1
        self.setup(stage="fit")

# standard libraries
import tempfile
import time
import multiprocessing as mp
import warnings
import subprocess
import os
import shutil
from pathlib import Path
from collections.abc import Callable
from typing import Optional

# deep learning packages
import lightning.pytorch as pl
from torch.utils.data import DataLoader

from miss_alignment.alignment import evaluate_tilt_series
from miss_alignment.data.io import read_tomogram_from_pickle
from ._reconstruction_worker import reconstruction_worker
from .training_dataset import ReconstructionPoolDataset


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
    zenodo_archive = "16574872"

    def __init__(
        self,
        dataset_directory: os.PathLike,
        shift_generator: Callable,
        reconstruction_workers: int = 4,
        dataloader_workers: int = 4,
        batch_size: int = 4,
        steps_per_epoch: int = 1000,
        patch_size: int = 64,
        download_data: bool = False,
    ):
        super().__init__()
        # data module controls
        self.batch_size = batch_size
        self.pool_size = batch_size * steps_per_epoch
        self.reconstruction_workers = reconstruction_workers
        self.dataloader_workers = dataloader_workers

        # reconstruction controls
        self.patch_size = patch_size
        self.shift_generator = shift_generator

        # initialize the dataset and pool attributes
        self.dataset_directory = Path(dataset_directory)
        self.allow_download = download_data
        self.train_dataset = None
        self.pool_dir = None
        self.pool_processes: list = []
        self.stop_event: Optional[mp.Event] = None

    def __enter__(self):
        """Enter context manager and setup pool."""
        self.setup()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit context manager and cleanup."""
        self.teardown()
        return False

    @property
    def data_directory_is_empty(self):
        return len(list(self.dataset_directory.iterdir())) == 0

    def prepare_data(self):
        # this is not done in the setup() because lightning specifically
        # calls this function with a single process to prevent race conditions
        if self.allow_download:
            self.dataset_directory.mkdir(exist_ok=True, parents=True)
            if self.data_directory_is_empty:
                subprocess.run(
                    [
                        "zenodo_get",
                        self.zenodo_archive,  # update value
                        "--output-dir",
                        str(self.dataset_directory),
                    ]
                )
                for archive in ("torch_tiltxcorr.zip", "ground_truth.zip"):
                    zipped_archive = self.dataset_directory / archive
                    shutil.unpack_archive(
                        zipped_archive,
                        extract_dir=self.dataset_directory,
                    )
                    os.remove(zipped_archive)
            else:
                warnings.warn(
                    "dataset directory is non-empty, download is skipped"
                )

    def setup(self, stage: str = "fit"):
        """No stage for setup because this is used just for training."""
        if stage != "fit":
            raise ValueError(f"{stage} is not a valid stage")

        # Create temporary directory for pool
        self.pool_dir = Path(tempfile.mkdtemp(prefix='recon_pool_'))
        print(f"Created pool directory: {self.pool_dir}")

        # Partition indices among workers
        num_workers = self.reconstruction_workers
        indices_per_worker = self.pool_size // num_workers
        partitions = []
        for i in range(num_workers):
            start_idx = i * indices_per_worker
            if i == num_workers - 1:
                # Last worker gets any remaining indices
                end_idx = self.pool_size
            else:
                end_idx = start_idx + indices_per_worker
            partitions.append(list(range(start_idx, end_idx)))

        # Start worker processes
        self.stop_event = mp.Event()
        ready_flag = mp.Value('i', 0)
        source_counter = mp.Value('i', 0)

        for worker_id, indices in enumerate(partitions):
            p = mp.Process(
                target=reconstruction_worker,
                args=(
                    worker_id,
                    indices,
                    self.pool_dir,
                    ready_flag,
                    self.stop_event,
                    source_counter
                )
            )
            p.start()
            self.pool_processes.append(p)

        # Wait for all workers to complete initial fill
        print("Waiting for pool to be filled...")
        while ready_flag.value < num_workers:
            time.sleep(0.1)
        print("Pool ready!")

        # Create dataset
        self.train_dataset = ReconstructionPoolDataset(
            self.pool_dir, self.pool_size
        )

    def train_dataloader(self):
        """Return DataLoader for training data."""
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            drop_last=True,
            num_workers=self.dataloader_workers,
            persistent_workers=True,
            pin_memory=True,
        )

    def teardown(self, stage: Optional[str] = None):
        """Clean up pool and worker processes."""
        if self.stop_event:
            print("Stopping reconstruction workers...")
            self.stop_event.set()

            for p in self.pool_processes:
                p.join(timeout=5.0)
                if p.is_alive():
                    p.terminate()
                    p.join()

        if self.pool_dir and self.pool_dir.exists():
            shutil.rmtree(self.pool_dir)
            print(f"Cleaned up pool directory: {self.pool_dir}")

        print("Cleanup complete")

    #TODO this should happen in another place now!
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

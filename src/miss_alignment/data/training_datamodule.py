# standard libraries
import tempfile
import multiprocessing as mp
import os
import shutil
from pathlib import Path
from collections.abc import Callable
from typing import Optional
import numpy as np
import time

# deep learning packages
import lightning.pytorch as pl
from torch.utils.data import DataLoader

from ._reconstruction_worker import reconstruction_worker
from ._pool_monitor import SimplePoolMonitor
from .training_dataset import ReconstructionPoolDataset

if "MISS_ALIGNMENT_RECON_POOL_SIZE" in os.environ:
    RECON_POOL_SIZE = int(os.environ["MISS_ALIGNMENT_RECON_POOL_SIZE"])
else:
    RECON_POOL_SIZE = 1000


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
        dataset_directory: os.PathLike,
        shift_generator: Callable,
        reconstruction_workers: int = 4,
        dataloader_workers: int = 4,
        batch_size: int = 4,
        steps_per_epoch: int = 1000,
        patch_size: int = 64,
        tomogram_shape: tuple[int, int, int] = (180, 512, 512),
        monitor: Optional[SimplePoolMonitor] = None,
    ):
        super().__init__()
        # data module controls
        self.batch_size = batch_size
        self.epoch_size = steps_per_epoch * batch_size
        self.reconstruction_workers = reconstruction_workers
        self.dataloader_workers = dataloader_workers

        # reconstruction controls
        self.tomogram_shape = tomogram_shape
        self.patch_size = patch_size
        self.shift_generator = shift_generator

        # initialize the dataset and pool attributes
        self.dataset_directory = Path(dataset_directory)
        self.train_dataset = None
        self.pool_dir = None
        self.pool_processes: list = []
        self.stop_event: Optional[mp.Event] = None
        self.ready_flag: Optional[mp.Event] = None

        # timing production and consumption rates
        self.monitor = monitor

    def __enter__(self):
        """Enter context manager and setup pool."""
        self.setup()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit context manager and cleanup."""
        self.teardown()
        return False

    def prepare_data(self):
        pass

    def setup(self, stage: str = "fit"):
        """No stage for setup because this is used just for training."""
        if stage != "fit":
            raise ValueError(f"{stage} is not a valid stage")

        mp.set_start_method("spawn", force=True)

        # Create temporary directory for pool
        self.pool_dir = Path(tempfile.mkdtemp(prefix='recon_pool_'))
        print(f"Created pool directory: {self.pool_dir}")

        # Partition indices among workers
        partitions = np.array_split(range(RECON_POOL_SIZE), self.reconstruction_workers)

        # Start worker processes
        self.stop_event = mp.Event()
        self.ready_flag = mp.Value('i', 0)

        # get a list of all the tilt series pickle files
        tilt_series_pickles = (
            [x for x in self.dataset_directory.iterdir() if x.suffix =='.pickle']
        )

        for worker_id, indices in enumerate(partitions):
            p = mp.Process(
                target=reconstruction_worker,
                args=(
                    worker_id,
                    indices.tolist(),
                    self.pool_dir,
                    tilt_series_pickles,
                    self.tomogram_shape,
                    self.patch_size,
                    self.shift_generator,
                    self.ready_flag,
                    self.stop_event,
                    self.monitor
                )
            )
            p.start()
            self.pool_processes.append(p)

        # Create dataset
        self.train_dataset = ReconstructionPoolDataset(
            self.pool_dir, RECON_POOL_SIZE, self.epoch_size
        )

    def wait_for_pool_to_fill(self):
        print("Waiting for pool to be filled...")
        while self.ready_flag.value < self.reconstruction_workers:
            time.sleep(0.1)
        print("Pool ready!")

    def train_dataloader(self):
        """Return DataLoader for training data."""
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=False,  # data is already randomized by recon workers
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

        if self.monitor is not None:
            # Print final statistics
            print("\n" + "=" * 50)
            print("FINAL STATISTICS")
            print("=" * 50)
            self.monitor.print_stats()

# standard libraries
import tempfile
import multiprocessing as mp
import shutil
from pathlib import Path
from collections.abc import Callable
from typing import Optional

# deep learning packages
import torch
import lightning.pytorch as pl
from torch.utils.data import DataLoader

from ._reconstruction_worker import reconstruction_worker
from .training_dataset import ReconstructionPoolDataset

# Default pool size
DEFAULT_POOL_SIZE = 1000


def _worker_init_fn(worker_id: int):
    """Initialize DataLoader worker with its partition ID."""
    worker_info = torch.utils.data.get_worker_info()
    if worker_info is not None:
        dataset = worker_info.dataset
        dataset.partition_id = worker_id


class MissAlignmentDataModule(pl.LightningDataModule):
    """
    Training datamodule with coupled reconstruction and data loading workers.

    Each reconstruction worker is paired 1:1 with a DataLoader worker, sharing
    a partition of the pool. Reconstruction workers pause when their partition
    is full, and DataLoader workers pause when their partition is too empty.

    Directory layout:

    dataset_directory/
    +- iter0/  # this is the initial state
    ¦  +- model_0.json  # stores tilt series metadata
    ¦  +- model_1.json
    ¦  +- ...
    +- iter1/  # this is where the updated alignments are stored
    ¦  +- model_0.json
    etc...

    Parameters
    ----------
    dataset_directory : os.PathLike
        Directory containing tilt series JSON files
    shift_generator : Callable
        Function that generates synthetic alignment shifts
    n_workers : int, default 4
        Number of reconstruction/DataLoader worker pairs
    reconstruction_accelerators : Optional[list[int]], default None
        GPU device IDs for reconstruction workers
    batch_size : int, default 4
        Batch size for training
    steps_per_epoch : int, default 1000
        Number of steps per training epoch
    patch_size : int, default 64
        Size of 3D patches (patch_size^3)
    apply_ctf : bool, default False
        Apply CTF correction during reconstruction
    downsample : int, default 1
        Downsampling factor for reconstructions
    pool_size : int, default 1000
        Total size of reconstruction pool (divided among partitions)
    """

    def __init__(
        self,
        dataset_directory,
        shift_generator: Callable,
        n_workers: int = 4,
        reconstruction_accelerators: Optional[list[int]] = None,
        batch_size: int = 4,
        steps_per_epoch: int = 1000,
        patch_size: int = 64,
        apply_ctf: bool = False,
        downsample: int = 1,
        pool_size: int = DEFAULT_POOL_SIZE,
    ):
        super().__init__()
        # data module controls
        self.batch_size = batch_size
        self.epoch_size = steps_per_epoch * batch_size
        self.n_workers = n_workers
        self.pool_size = pool_size

        # Validate pool size
        self.partition_size = pool_size // n_workers
        if self.partition_size < 2 * batch_size:
            raise ValueError(
                f"Pool size {pool_size} with {n_workers} workers gives "
                f"partition_size={self.partition_size}, which is less than "
                f"2 * batch_size ({2 * batch_size}). Increase pool_size or "
                f"decrease n_workers/batch_size."
            )

        # Set up reconstruction devices
        if reconstruction_accelerators in [None, []]:
            self.reconstruction_devices = ["cpu"] * n_workers
        else:
            if len(reconstruction_accelerators) != n_workers:
                raise ValueError(
                    "Number of reconstruction accelerators must match n_workers"
                )
            self.reconstruction_devices = [
                f"cuda:{x}" for x in reconstruction_accelerators
            ]

        # reconstruction controls
        self.patch_size = patch_size
        self.apply_ctf = apply_ctf
        self.downsample = downsample
        self.shift_generator = shift_generator

        # initialize the dataset and pool attributes
        self.dataset_directory = Path(dataset_directory)
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

    def prepare_data(self):
        pass

    def setup(self, stage: str = "fit"):
        """No stage for setup because this is used just for training."""
        if stage != "fit":
            raise ValueError(f"{stage} is not a valid stage")

        mp.set_start_method("spawn", force=True)

        # Create temporary directory for pool
        self.pool_dir = Path(tempfile.mkdtemp(prefix="recon_pool_"))
        print(f"Created pool directory: {self.pool_dir}")

        # Start worker processes
        self.stop_event = mp.Event()

        # get a list of all the tilt series json files
        tilt_series_xmls = list(self.dataset_directory.glob("*.xml"))

        for partition_id in range(self.n_workers):
            p = mp.Process(
                target=reconstruction_worker,
                kwargs={
                    "partition_id": partition_id,
                    "partition_size": self.partition_size,
                    "pool_dir": self.pool_dir,
                    "tilt_series_xmls": tilt_series_xmls,
                    "patch_size": self.patch_size,
                    "apply_ctf": self.apply_ctf,
                    "downsample": self.downsample,
                    "shift_generator": self.shift_generator,
                    "stop_event": self.stop_event,
                    "tilt_series_refresh_rate": 10,
                    "device": self.reconstruction_devices[partition_id],
                },
            )
            p.start()
            self.pool_processes.append(p)

        # Create dataset
        self.train_dataset = ReconstructionPoolDataset(
            self.pool_dir, self.batch_size, self.epoch_size
        )

    def train_dataloader(self):
        """Return DataLoader for training data."""
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=False,  # data is already randomized by recon workers
            drop_last=True,
            num_workers=self.n_workers,
            worker_init_fn=_worker_init_fn,
            multiprocessing_context="fork",
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
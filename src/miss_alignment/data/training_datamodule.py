# standard libraries
import hashlib
import multiprocessing as mp
import shutil
import tempfile
from pathlib import Path
from collections.abc import Callable
from typing import Optional

# deep learning packages
import torch
import lightning.pytorch as pl
from torch.utils.data import DataLoader

from ._reconstruction_worker import reconstruction_worker
from .training_dataset import ReconstructionPoolDataset
from ..utils import is_rank_zero


# Default pool size
DEFAULT_POOL_SIZE = 1000


def _worker_init_fn(worker_id: int):
    """Initialize DataLoader worker with its partition ID.

    Each dataloader worker across all DDP ranks gets its own dedicated partition:
    - partition_id = global_rank * num_local_workers + worker_id

    With n_partitions = n_training_devices * dataloader_workers, each worker
    gets exactly one partition (no sharing).

    Note: global_rank is obtained directly from torch.distributed here, not from
    the dataset. This is critical because the dataloader may be created before
    DDP is initialized, but worker_init_fn runs after DDP setup (workers are
    forked from DDP processes).
    """
    worker_info = torch.utils.data.get_worker_info()
    if worker_info is not None:
        dataset = worker_info.dataset
        # Get global_rank directly from distributed - this works because
        # DataLoader workers are forked AFTER DDP is initialized, so they
        # inherit the distributed state from their parent DDP process.
        if torch.distributed.is_initialized():
            global_rank = torch.distributed.get_rank()
        else:
            global_rank = 0
        num_local_workers = worker_info.num_workers

        # Each dataloader worker gets its own partition
        # partition_id = global_rank * num_local_workers + worker_id
        dataset.partition_id = global_rank * num_local_workers + worker_id


class MissAlignmentDataModule(pl.LightningDataModule):
    """
    Training datamodule with reconstruction workers and DataLoader consumers.

    Reconstruction workers cycle through all partitions, writing to each in
    round-robin order. Each DataLoader worker (across all DDP ranks) gets
    its own dedicated partition.

    Partitioning scheme:
        n_partitions = n_training_devices * dataloader_workers
        partition_id = global_rank * dataloader_workers + local_worker_id

    File naming:
        partition_{partition_id}_worker_{worker_id}_seq_{sequential_id}.pickle

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
    reconstruction_accelerators : list[int]
        GPU device IDs for reconstruction workers. The number of workers is
        determined by the length of this list. Devices can be repeated to run
        multiple workers per GPU (e.g., [0, 0, 1, 1]).
    n_training_devices : int
        Number of training devices (GPUs) used for DDP training
    batch_size : int, default 4
        Batch size for training (per device in DDP)
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
    dataloader_workers : int, default 2
        Number of CPU DataLoader workers per training device.
    """

    def __init__(
        self,
        dataset_directory,
        shift_generator: Callable,
        reconstruction_accelerators: list[int],
        n_training_devices: int,
        batch_size: int = 4,
        steps_per_epoch: int = 1000,
        patch_size: int = 64,
        apply_ctf: bool = False,
        downsample: int = 1,
        pool_size: int = DEFAULT_POOL_SIZE,
        dataloader_workers: int = 2,
    ):
        super().__init__()

        # Validate reconstruction accelerators
        if not reconstruction_accelerators:
            raise ValueError("reconstruction_accelerators cannot be empty")

        # Number of workers is determined by the accelerators list
        self.n_workers = len(reconstruction_accelerators)
        self.reconstruction_devices = [f"cuda:{x}" for x in reconstruction_accelerators]

        # data module controls
        self.batch_size = batch_size
        self.epoch_size = steps_per_epoch * batch_size
        self.pool_size = pool_size
        self.dataloader_workers = dataloader_workers
        self.n_training_devices = n_training_devices

        # Calculate number of partitions: one per dataloader worker across all ranks
        self.n_partitions = n_training_devices * dataloader_workers

        # Validate pool size
        self.partition_size = pool_size // self.n_partitions
        if self.partition_size < 2 * batch_size:
            raise ValueError(
                f"Pool size {pool_size} with {self.n_partitions} partitions gives "
                f"partition_size={self.partition_size}, which is less than "
                f"2 * batch_size ({2 * batch_size}). Increase pool_size or "
                f"decrease n_training_devices/dataloader_workers/batch_size."
            )

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
        self._workers_spawned = False  # Guard against duplicate worker spawning

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
        """Set up the data module for training.

        This method spawns reconstruction workers (on rank 0 only) and creates
        the dataset. All ranks share the same pool directory.

        In DDP, only rank 0 spawns reconstruction workers. Other ranks skip
        worker spawning but still create their dataset pointing to the shared
        pool directory.
        """
        if stage != "fit":
            raise ValueError(f"{stage} is not a valid stage")

        # Guard: Only run setup once per process
        if self._workers_spawned:
            return

        # Use a deterministic pool directory in /tmp so all DDP ranks know the path.
        # The path is based on a hash of the training directory to avoid collisions.
        dir_hash = hashlib.md5(
            str(self.dataset_directory.resolve()).encode()
        ).hexdigest()[:12]
        self.pool_dir = Path(tempfile.gettempdir()) / f"recon_pool_{dir_hash}"

        # Only rank 0 spawns reconstruction workers
        if is_rank_zero():
            mp.set_start_method("spawn", force=True)

            # Clean up any existing pool directory from previous runs
            if self.pool_dir.exists():
                shutil.rmtree(self.pool_dir)
            self.pool_dir.mkdir(parents=True, exist_ok=True)
            print(f"Created pool directory: {self.pool_dir}")
            print(
                f"Pool configuration: {self.n_workers} workers -> "
                f"{self.n_partitions} partitions ({self.partition_size} files each)"
            )

            # Start worker processes
            self.stop_event = mp.Event()

            # get a list of all the tilt series json files
            tilt_series_xmls = list(self.dataset_directory.glob("*.xml"))

            for worker_id in range(self.n_workers):
                p = mp.Process(
                    target=reconstruction_worker,
                    kwargs={
                        "worker_id": worker_id,
                        "n_partitions": self.n_partitions,
                        "partition_size": self.partition_size,
                        "pool_dir": self.pool_dir,
                        "tilt_series_xmls": tilt_series_xmls,
                        "patch_size": self.patch_size,
                        "apply_ctf": self.apply_ctf,
                        "downsample": self.downsample,
                        "shift_generator": self.shift_generator,
                        "stop_event": self.stop_event,
                        "tilt_series_refresh_rate": 10,
                        "device": self.reconstruction_devices[worker_id],
                    },
                )
                p.start()
                self.pool_processes.append(p)

        # Mark setup as complete to prevent duplicate execution
        self._workers_spawned = True

        # Create dataset (global_rank is obtained in worker_init_fn
        # from torch.distributed). All ranks create their own dataset
        # pointing to the shared pool directory.
        self.train_dataset = ReconstructionPoolDataset(
            pool_dir=self.pool_dir,
            batch_size=self.batch_size,
            epoch_size=self.epoch_size,
            n_partitions=self.n_partitions,
        )

    def train_dataloader(self):
        """Return DataLoader for training data."""
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=False,  # data is already randomized by recon workers
            drop_last=True,
            num_workers=self.dataloader_workers,
            worker_init_fn=_worker_init_fn,
            multiprocessing_context="fork",
            persistent_workers=True,
            pin_memory=True,
        )

    def teardown(self, stage: Optional[str] = None):
        """Clean up pool and worker processes.

        Only rank 0 performs cleanup since only rank 0 spawns workers.
        Other ranks simply reset their local state.
        """
        # Only perform full teardown on rank 0 (where workers were spawned)
        if is_rank_zero():
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

        # Reset state for potential reuse (all ranks)
        self._workers_spawned = False
        self.pool_processes = []
        self.stop_event = None
        self.pool_dir = None

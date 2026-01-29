# standard libraries
import os
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


def _is_rank_zero() -> bool:
    """Check if we're on the main process (rank 0) in DDP.

    This checks both torch.distributed (if initialized) and the LOCAL_RANK
    environment variable (set by DDP launchers before script execution).
    """
    if torch.distributed.is_initialized():
        return torch.distributed.get_rank() == 0
    # Check LOCAL_RANK env var (set by DDP launcher before torch.distributed init)
    local_rank = os.environ.get("LOCAL_RANK")
    if local_rank is not None:
        return int(local_rank) == 0
    return True


# Default pool size
DEFAULT_POOL_SIZE = 1000


def _worker_init_fn(worker_id: int):
    """Initialize DataLoader worker with its partition ID.

    Calculates global partition assignment for multi-GPU DDP training:
    - global_worker_id = global_rank * num_local_workers + worker_id
    - partition_id = global_worker_id % n_partitions

    This allows multiple DataLoader workers to consume from the same partition
    when there are more consumers than producers.

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
        n_partitions = dataset.n_partitions

        global_worker_id = global_rank * num_local_workers + worker_id
        dataset.partition_id = global_worker_id % n_partitions


class MissAlignmentDataModule(pl.LightningDataModule):
    """
    Training datamodule with reconstruction workers and DataLoader consumers.

    Reconstruction workers write to partitioned pool directories. DataLoader
    workers consume from these partitions. Multiple DataLoader workers can
    share a partition (they race to grab files, with retry logic handling
    conflicts).

    For multi-GPU DDP training, partition assignment is global:
        global_worker_id = global_rank * dataloader_workers + local_worker_id
        partition_id = global_worker_id % n_workers

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
    n_workers : int
        Number of reconstruction workers (each writes to its own partition)
    reconstruction_accelerators : list[int]
        GPU device IDs for reconstruction workers (one per worker)
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
        Number of CPU DataLoader workers per training device. Can exceed
        n_workers; multiple DataLoader workers will share partitions.
    """

    def __init__(
        self,
        dataset_directory,
        shift_generator: Callable,
        n_workers: int,
        reconstruction_accelerators: list[int],
        batch_size: int = 4,
        steps_per_epoch: int = 1000,
        patch_size: int = 64,
        apply_ctf: bool = False,
        downsample: int = 1,
        pool_size: int = DEFAULT_POOL_SIZE,
        dataloader_workers: int = 2,
    ):
        super().__init__()
        # data module controls
        self.batch_size = batch_size
        self.epoch_size = steps_per_epoch * batch_size
        self.n_workers = n_workers
        self.pool_size = pool_size
        self.dataloader_workers = dataloader_workers

        # Validate pool size
        self.partition_size = pool_size // n_workers
        if self.partition_size < 2 * batch_size:
            raise ValueError(
                f"Pool size {pool_size} with {n_workers} workers gives "
                f"partition_size={self.partition_size}, which is less than "
                f"2 * batch_size ({2 * batch_size}). Increase pool_size or "
                f"decrease n_workers/batch_size."
            )

        # This is an old code path that used cpu's for reconstruction
        # we have abandonded it in favor of splitting the GPU's
        if reconstruction_accelerators in [None, []]:
            # self.reconstruction_devices = ["cpu"] * n_workers
            raise NotImplementedError

        # Set up reconstruction devices
        if len(reconstruction_accelerators) != n_workers:
            raise ValueError(
                "Number of reconstruction accelerators must match n_workers"
            )
        self.reconstruction_devices = [f"cuda:{x}" for x in reconstruction_accelerators]

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

        # Use a deterministic pool directory so all ranks know the path.
        # This is placed inside the training directory to ensure all ranks
        # can access it (important for distributed filesystems).
        self.pool_dir = self.dataset_directory / ".recon_pool"

        # Only rank 0 spawns reconstruction workers
        if _is_rank_zero():
            mp.set_start_method("spawn", force=True)

            # Clean up any existing pool directory from previous runs
            if self.pool_dir.exists():
                shutil.rmtree(self.pool_dir)
            self.pool_dir.mkdir(parents=True, exist_ok=True)
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

        # Mark setup as complete to prevent duplicate execution
        self._workers_spawned = True

        # Create dataset (global_rank is obtained in worker_init_fn
        # from torch.distributed). All ranks create their own dataset
        # pointing to the shared pool directory.
        self.train_dataset = ReconstructionPoolDataset(
            pool_dir=self.pool_dir,
            batch_size=self.batch_size,
            epoch_size=self.epoch_size,
            n_partitions=self.n_workers,
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
        if _is_rank_zero():
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

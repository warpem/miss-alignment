import time
import random
import pickle
from pathlib import Path

import einops
import torch
from torch.utils.data import Dataset

from ._augmentation import (
    random_contrast,
    random_edge_mask,
    random_cube_mask,
)

# pause polling interval in seconds
PAUSE_POLL_INTERVAL = 0.05  # 50ms


class ReconstructionPoolDataset(Dataset):
    """
    Dataset that reads pre-computed reconstructions from a partitioned pool.

    Each DataLoader worker is assigned a partition via worker_init_fn.
    The dataset reads files from its partition, converts fp16 to fp32,
    deletes the file after reading, and applies augmentations.

    Parameters
    ----------
    pool_dir : Path
        Directory containing the reconstruction pool files
    batch_size : int
        Minimum number of files required before reading (pause threshold)
    epoch_size : int
        Number of samples per epoch
    n_partitions : int
        Total number of partitions (reconstruction workers)
    """

    def __init__(
        self,
        pool_dir: Path,
        batch_size: int,
        epoch_size: int,
        n_partitions: int,
    ):
        self.pool_dir = pool_dir
        self.batch_size = batch_size
        self.epoch_size = epoch_size
        self.n_partitions = n_partitions
        # partition_id is set by worker_init_fn in the DataLoader
        self.partition_id: int | None = None

    def __len__(self) -> int:
        # we just set this to define the size of an epoch
        return self.epoch_size

    def _list_partition_files(self) -> list[Path]:
        """List all files in this worker's partition."""
        if self.partition_id is None:
            raise RuntimeError(
                "partition_id not set. Ensure worker_init_fn assigns it."
            )
        pattern = f"partition_{self.partition_id}_*.pickle"
        # Filter out temp files (they start with tmp_)
        return [f for f in self.pool_dir.glob(pattern) if not f.name.startswith("tmp_")]

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, ...]:
        """
        Fetch a reconstruction from this worker's partition.

        The idx parameter is ignored - we pick any available file from
        the partition. This method blocks if fewer than batch_size files
        are available.

        Parameters
        ----------
        idx : int
            Index (ignored, used only for epoch length)

        Returns
        -------
        tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
            Three volumes and their labels
        """
        # Wait until we have enough files in our partition
        files = self._list_partition_files()
        while len(files) < self.batch_size:
            time.sleep(PAUSE_POLL_INTERVAL)
            files = self._list_partition_files()

        # Pick a random file from available ones
        file_path = random.choice(files)

        # Read the file
        try:
            with open(file_path, "rb") as infile:
                examples = pickle.load(infile)
            # Delete the file after successful read
            file_path.unlink()
        except (FileNotFoundError, EOFError, pickle.UnpicklingError):
            # File was deleted by another worker or is corrupted
            # Retry with a different file
            return self.__getitem__(idx)

        # Convert fp16 to fp32 and extract volumes/labels
        volumes = []
        labels = []
        for vol, label in examples:
            volumes.append(vol.float())  # fp16 -> fp32
            labels.append(label)

        # Validate triplet structure
        if not (1 in labels and -1 in labels and all(i in [1, -1] for i in labels)):
            raise ValueError(
                "Training examples must contain positive and negative labels."
            )

        # Shuffle the triplet order
        combined = list(zip(volumes, labels))
        random.shuffle(combined)
        volumes, labels = zip(*combined)
        labels = torch.tensor(labels)

        # run normalization and augmentation (no mirroring -
        # done by reconstruction worker)
        volumes = self._prep_and_augment(list(volumes))
        # add empty channel dim to all volumes
        volumes = [einops.rearrange(v, "d h w -> 1 d h w") for v in volumes]

        return *volumes, labels

    def _prep_and_augment(self, volumes: list[torch.Tensor]) -> list[torch.Tensor]:
        volumes = [self._normalize(v) for v in volumes]
        volumes = [random_contrast(v) for v in volumes]
        volumes = [random_edge_mask(v, edge_width=(1, 5)) for v in volumes]
        volumes = [random_cube_mask(v) for v in volumes]
        return volumes

    def _normalize(self, volume: torch.Tensor) -> torch.Tensor:
        mean, std = torch.mean(volume), torch.std(volume)
        if std == 0.0:
            raise ValueError(
                "Cannot normalize patch because the standard deviation is 0."
            )
        return (volume - mean) / std

from torch.utils.data import Dataset
from pathlib import Path
import einops
import torch
import random
import pickle

from ._augmentation import (
    random_contrast,
    random_mirror,
    random_edge_mask,
    random_cube_mask,
)


class ReconstructionPoolDataset(Dataset):
    """
    Dataset that reads pre-computed reconstructions from a pool.

    The pool files are constantly being updated by background workers,
    but reads are always safe due to atomic rename operations.

    Parameters
    ----------
    pool_dir : Path
        Directory containing the reconstruction pool files
    pool_size : int
        Number of reconstructions in the pool
    """
    noise_augmentation = False

    def __init__(self, pool_dir: Path, pool_size: int, epoch_size: int):
        self.pool_dir = pool_dir
        self.pool_size = pool_size
        self.epoch_size = epoch_size

    def __len__(self) -> int:
        # we just set this to define the size of an epoch
        return self.epoch_size

    def __getitem__(self, idx: int) -> tuple[torch.Tensor]:
        """
        Fetch a reconstruction from the pool.

        Parameters
        ----------
        idx : int
            Index of the reconstruction to fetch

        Returns
        -------
        tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
            Dictionary containing the reconstruction data
        """
        file_path = self.pool_dir / f"recon_{idx % self.pool_size}.pickle"

        # This should always succeed due to atomic rename
        with open(file_path, "rb") as infile:
            examples = pickle.load(infile)

        # shuffle the triplet
        random.shuffle(examples)
        volumes, labels = zip(*examples)
        labels = torch.tensor(labels)

        # volumes, labels = data_and_labels[:-1], data_and_labels[-1]
        # augment and add empty channel dim to all volumes
        volumes = self._augment(volumes)
        volumes = [einops.rearrange(v, "d h w -> 1 d h w") for v in volumes]

        return *volumes, labels

    def _augment(self, volumes: list[torch.Tensor]) -> list[torch.Tensor]:
        if self.noise_augmentation and random.random() > 0.5:
            noise_std = random.random() * 1.5
            volumes = [v + torch.normal(
                mean=0.0,
                std=noise_std,
                size=v.shape,
            ) for v in volumes]
        volumes = [random_edge_mask(v, edge_width=(1, 5))
                   for v in volumes]
        volumes = [random_cube_mask(v) for v in volumes]
        volumes = [self._normalize(v) for v in volumes]
        volumes = [random_contrast(v) for v in volumes]
        volumes = random_mirror(volumes)
        return volumes

    def _normalize(self, volume: torch.Tensor) -> torch.Tensor:
        mean, std = torch.mean(volume), torch.std(volume)
        volume = torch.nan_to_num(volume, nan=float(mean))
        return (volume - mean) / std

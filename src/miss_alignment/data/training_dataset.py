from torch.utils.data import Dataset
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

    def __init__(self, pool_dir: Path, pool_size: int):
        self.pool_dir = pool_dir
        self.pool_size = pool_size

    def __len__(self) -> int:
        return self.pool_size

    def __getitem__(self, idx: int) -> tuple[torch.Tensor]:
        """
        Fetch a reconstruction from the pool.

        Parameters
        ----------
        idx : int
            Index of the reconstruction to fetch

        Returns
        -------
        tuple[torch.Tensor]
            Dictionary containing the reconstruction data
        """
        file_path = self.pool_dir / f"recon_{idx}.pickle"

        # This should always succeed due to atomic rename
        with open(file_path, "rb") as infile:
            data_and_labels = pickle.load(infile)

        _, labels = data_and_labels
        # augment all volumes but skip the labels (last elem of list)
        volumes = [self._augment(x) for x in data_and_labels[:-1]]
        volumes = [einops.rearrange(v, "d h w -> 1 d h w") for v in volumes]

        return *volumes, labels

    def _augment(self, volume: torch.Tensor) -> torch.Tensor:
        if self.noise_augmentation and random.random() > 0.5:
            noise_std = random.random() * 1.5
            volume = volume + torch.normal(
                mean=0.0,
                std=noise_std,
                size=volume.shape,
            )
        volume = random_edge_mask(volume, edge_width=(1, 5))
        volume = random_cube_mask(volume)
        volume = self._normalize(volume)
        volume = random_contrast(volume)
        volume = random_mirror(volume)
        return volume

    def _normalize(self, volume: torch.Tensor) -> torch.Tensor:
        mean, std = torch.mean(volume), torch.std(volume)
        volume = torch.nan_to_num(volume, nan=float(mean))
        return (volume - mean) / std

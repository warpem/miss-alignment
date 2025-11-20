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
        if not (1 in labels and -1 in labels and all(i in [1, -1] for i in labels)):
            raise ValueError(
                "Training examples must contain positive and negative labels."
            )
        labels = torch.tensor(labels)

        # run normalization and augmentation
        volumes = self._prep_and_augment(volumes)
        # add empty channel dim to all volumes
        volumes = [einops.rearrange(v, "d h w -> 1 d h w") for v in volumes]

        return *volumes, labels

    def _prep_and_augment(self, volumes: list[torch.Tensor]) -> list[torch.Tensor]:
        volumes = [self._normalize(v) for v in volumes]
        volumes = [random_contrast(v) for v in volumes]
        volumes = [random_edge_mask(v, edge_width=(1, 5)) for v in volumes]
        volumes = [random_cube_mask(v) for v in volumes]
        # random mirror works on list to ensure consistency between triplets
        volumes = random_mirror(volumes)
        return volumes

    def _normalize(self, volume: torch.Tensor) -> torch.Tensor:
        mean, std = torch.mean(volume), torch.std(volume)
        if std == 0.0:
            raise ValueError(
                "Cannot normalize patch because the standard deviation is 0."
            )
        return (volume - mean) / std

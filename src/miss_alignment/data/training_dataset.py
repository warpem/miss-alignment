from os import PathLike
from pathlib import Path
from typing import List

import random
import einops
import numpy as np
import torch
from torchvision.transforms import functional as TF
from torch.utils.data import Dataset
import mrcfile
from scipy.spatial.transform import Rotation
from torch_fourier_slice import project_3d_to_2d, backproject_2d_to_3d
from torch_fourier_shift import fourier_shift_image_2d
import torch.nn.functional as F


def random_contrast(*images):
    std_change, mean_change = (
        torch.normal(1, 0.3, (1,)), torch.normal(0, 0.3, (1,))
    )
    return [i * std_change + mean_change for i in images]


def random_flip(*images: torch.Tensor, p=0.5) -> List[torch.Tensor]:
    """Apply the same random flip to multiple images."""
    if random.random() > 1 - p:
        images = [TF.hflip(image) for image in images]
    if random.random() > 1 - p:
        images = [TF.vflip(image) for image in images]
    return list(images)


def random_cube_mask(*volumes: torch.Tensor, p=0.5, size_range=(0.1, 0.3)) -> \
List[torch.Tensor]:
    """Apply the same random cube mask to multiple volumes.

    Args:
        *volumes: 3D volumes of shape [D, H, W] without channel dimension
        p: Probability of applying the mask
        size_range: Range for the size of the cube as a fraction of the volume dimensions

    Returns:
        List of masked volumes
    """
    if random.random() > 1 - p:
        # Get the dimensions of the first volume (assuming all have same dimensions)
        d, h, w = volumes[0].shape

        # Determine mask size as fraction of volume dimensions
        mask_fraction = random.uniform(size_range[0], size_range[1])
        mask_d = max(1, int(d * mask_fraction))
        mask_h = max(1, int(h * mask_fraction))
        mask_w = max(1, int(w * mask_fraction))

        # Random starting positions for the mask
        start_d = random.randint(0, d - mask_d)
        start_h = random.randint(0, h - mask_h)
        start_w = random.randint(0, w - mask_w)

        # Apply the same mask to all volumes
        masked_volumes = []
        for volume in volumes:
            masked = volume.clone()
            masked[start_d:start_d + mask_d,
            start_h:start_h + mask_h,
            start_w:start_w + mask_w] = 0
            masked_volumes.append(masked)

        return masked_volumes

    return list(volumes)


class EMDBDataset(Dataset):
    """
    Dataset for loading MRC files and applying transformations.

    Parameters
    ----------
    directory : PathLike
        Directory with MRC files for training.
    target_size : int or tuple
        Target size to pad/crop images to. If an integer, same size is used for all dimensions.
    """

    def __init__(
        self,
        directory: PathLike,
        train: bool = True,
        target_size: int = 64,
    ):
        self.dataset_directory = Path(directory)

        if len(self.mrc_files) == 0:
            raise FileNotFoundError("No MRC files found in the dataset directory.")

        self.target_size = (target_size, target_size, target_size)

        self.train() if train else self.eval()

    def train(self):
        self._is_training = True

    def eval(self):
        self._is_training = False

    @property
    def mrc_files(self):
        return sorted(self.dataset_directory.glob("*.mrc"))

    def __len__(self):
        return len(self.mrc_files)

    def __getitem__(self, idx):
        # Load MRC file
        mrc_path = self.mrc_files[idx]
        with mrcfile.open(mrc_path) as mrc:
            # Convert to torch tensor and move to device
            volume = torch.from_numpy(mrc.data.astype(np.float32))

        volume = self._preprocess(volume, random_affine=True)

        tilt_angles = Rotation.from_euler(
            seq="Y", angles=np.arange(-51, 54, 3), degrees=True
        )
        rotations = torch.tensor(tilt_angles.as_matrix()).float()
        aligned, misaligned = self._generate_reconstructions(volume, rotations)

        volume = volume.float()
        aligned = aligned.float()
        misaligned = misaligned.float()

        return {  # add channel dimension to all output
            "volume": einops.rearrange(volume, "d h w -> 1 d h w"),
            "aligned": einops.rearrange(aligned, "d h w -> 1 d h w"),
            "misaligned": einops.rearrange(misaligned, "d h w -> 1 d h w"),
            "file_path": mrc_path.__str__(),
        }

    def _generate_reconstructions(
        self, volume: torch.Tensor, matrices: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        tilts = project_3d_to_2d(volume, rotation_matrices=matrices, pad=True)
        tilts = tilts + torch.normal(
            mean=0.0,
            std=float(torch.normal(3, .5, (1,))),
            size=tilts.shape,
            device=tilts.device,
        )

        # generate shift magnitude
        misaligned_std = torch.rand(1) * 2.0  # number between 0 a 2
        aligned_std = torch.rand(1) * misaligned_std  # number between 0 and misaligned_std

        # create aligned reconstruction
        aligned = self._reconstruct(tilts, aligned_std, matrices)
        aligned = self._normalize(aligned)

        # create misaligned
        misaligned = self._reconstruct(tilts, misaligned_std, matrices)
        misaligned = self._normalize(misaligned)

        if self._is_training:  # contrast adjustment
            aligned, misaligned = random_contrast(aligned, misaligned)
            aligned, misaligned = random_flip(aligned, misaligned)
            aligned, misaligned = random_cube_mask(aligned, misaligned)

        return aligned, misaligned

    def _reconstruct(self, tilts, shift_std, matrices):
        translations = torch.normal(
            mean=0.0,
            std=float(shift_std),
            size=(matrices.shape[0], 2),  # batch of number of tilts
            device=tilts.device,
        )
        shifted = fourier_shift_image_2d(
            tilts, translations
        )
        reconstruction = backproject_2d_to_3d(
            shifted,
            rotation_matrices=matrices,
            pad=True,
        )
        return reconstruction

    def _preprocess(
        self, volume: torch.Tensor, random_affine: bool = True
    ) -> torch.Tensor:
        volume = einops.rearrange(volume, "d h w -> 1 1 d h w")
        volume = self._pad_to_target_size(volume)
        volume = self._normalize(volume)
        if random_affine:
            volume = self._apply_random_affine_transform(volume)
        return einops.rearrange(volume, "1 1 d h w -> d h w")

    def _normalize(self, volume: torch.Tensor) -> torch.Tensor:
        mean, std = torch.mean(volume), torch.std(volume)
        torch.nan_to_num(volume, nan=mean)
        return (volume - mean) / std

    def _pad_to_target_size(self, volume: torch.Tensor) -> torch.Tensor:
        """Pad volume to target size, centered."""
        current_size = volume.shape
        pad_size = []

        for i in range(-1, -4, -1):
            diff = self.target_size[i] - current_size[i]
            pad_left = diff // 2
            pad_right = diff - pad_left
            pad_size += [pad_left, pad_right]

        # Apply padding if needed
        padded_volume = F.pad(
            volume,
            pad_size,
            mode="constant",
        )
        return padded_volume

    def _apply_random_affine_transform(self, volume: torch.Tensor) -> torch.Tensor:
        """
        Apply a random 3D affine transformation (rotation + translation) to a volume.

        Parameters:
        -----------
        volume : torch.Tensor
            Input volume tensor of shape [batch_size, channels, depth, height, width]
        box_size : float or None
            Reference size for the translation. If None, it will be computed as the
            maximum dimension of the volume.

        Returns:
        --------
        torch.Tensor
            Transformed volume with the same shape as the input
        """
        # Create identity matrix as starting point
        affine_matrix = torch.eye(4, device=volume.device)

        # random rotation matrix from uniform distribution
        rot_matrix = torch.tensor(
            Rotation.random(1).as_matrix(),  # used numpy seed set for the worker
            dtype=torch.float32,
            device=volume.device,
        )

        # random translation sampled from Gaussian
        translation = torch.normal(
            mean=0.0,
            std=0.1,  # standard deviation assuming box coords from -1 to 1
            size=(3,),
            device=volume.device,
        )

        # Construct affine matrix:
        # [ R  t ]
        # [ 0  1 ]
        affine_matrix[:3, :3] = rot_matrix
        affine_matrix[:3, 3] = translation

        # Create the grid for sampling
        # We need to adjust the grid to account for the center being at the middle of the volume
        grid = F.affine_grid(
            einops.rearrange(affine_matrix[:3, :], "h w -> 1 h w"),
            volume.shape,
            align_corners=True,
        )

        # Apply the transformation using grid_sample
        transformed_volume = F.grid_sample(
            volume, grid, mode="bilinear", padding_mode="border", align_corners=True
        )

        # Return the volume in its original shape format
        return transformed_volume

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
from scipy.spatial.transform import Rotation as R
from torch_fourier_slice import (
    extract_central_slices_rfft_3d, insert_central_slices_rfft_3d
)
from torch_fourier_shift import fourier_shift_dft_2d,  fourier_shift_dft_3d
from torch_grid_utils import fftfreq_grid
import torch.nn.functional as F


def random_contrast(*images):
    std_change, mean_change = (
        torch.normal(1, 0.1, (1,)), torch.normal(0, 0.1, (1,))
    )
    return [i * std_change + mean_change for i in images]


def random_flip(*images: torch.Tensor, p=.3) -> List[torch.Tensor]:
    """Apply the same random flip to multiple images."""
    if random.random() > 1 - p:
        images = [TF.hflip(image) for image in images]
    if random.random() > 1 - p:
        images = [TF.vflip(image) for image in images]
    return list(images)


def random_cube_mask(*volumes: torch.Tensor, p=.3, size_range=(0.1, 0.3)) -> \
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

        volume = self._preprocess(volume)

        tilt_angles = R.from_euler(
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
        # generate shift magnitude
        misaligned_std = torch.rand(1) * 2.0  # number between 0 a 2
        aligned_std = torch.rand(1) * misaligned_std  # number between 0 and misaligned_std
        aligned_translations = torch.normal(
            mean=0.0,
            std=float(aligned_std),
            size=(matrices.shape[0], 2),  # batch of number of tilts
        )
        misaligned_translations = torch.normal(
            mean=0.0,
            std=float(misaligned_std),
            size=(matrices.shape[0], 2),  # batch of number of tilts
        )

        aligned, misaligned = self._reconstruct(
            volume,
            matrices,
            aligned_translations,
            misaligned_translations
        )

        # set the noise_std
        noise_std = float(torch.rand(1) * 2 + 2)
        aligned = aligned + torch.normal(
            mean=0.0,
            std=noise_std,
            size=aligned.shape,
        )
        misaligned = misaligned + torch.normal(
            mean=0.0,
            std=noise_std,
            size=aligned.shape,
        )
        aligned = self._normalize(aligned)
        misaligned = self._normalize(misaligned)

        if self._is_training:  # contrast adjustment
            aligned, misaligned = random_contrast(aligned, misaligned)
            aligned, misaligned = random_flip(aligned, misaligned)
            aligned, misaligned = random_cube_mask(aligned, misaligned)

        return aligned, misaligned

    def _reconstruct(
        self,
        volume: torch.Tensor,  # (d, h, w)
        tilt_rotation_matrices: torch.Tensor,  # (b, 3, 3)
        tilt_shifts_small: torch.Tensor,  # (b, 2)
        tilt_shifts_large: torch.Tensor,  # (b, 2)
    ):
        ## make volume dft
        # premultiply by sinc2
        grid = fftfreq_grid(
            image_shape=volume.shape,
            rfft=False,
            fftshift=True,
            norm=True,
            device=volume.device
        )

        volume = volume * torch.sinc(grid) ** 2

        # calculate DFT
        volume_dft = torch.fft.fftshift(volume, dim=(-3, -2, -1))  # volume center to array origin
        volume_dft = torch.fft.rfftn(volume_dft, dim=(-3, -2, -1))
        volume_dft = torch.fft.fftshift(volume_dft, dim=(-3, -2,))  # actual fftshift of 3D rfft

        # apply random 3d shift
        random_shift = torch.tensor(
            np.random.normal(
                loc=0, scale=.1 * volume.shape[-1], size=(3, )
            ),
            dtype=torch.float32
        )
        volume_dft = fourier_shift_dft_3d(
            dft=volume_dft,
            image_shape=volume.shape,
            shifts=random_shift,
            rfft=True,
            fftshifted=True
        )

        # get random rotation offset for slice extraction
        random_rotation = torch.tensor(R.random().as_matrix(), dtype=torch.float32)

        # extract tilt dfts
        tilt_dfts = extract_central_slices_rfft_3d(
            volume_rfft=volume_dft,
            image_shape=volume.shape,
            rotation_matrices=tilt_rotation_matrices @ random_rotation,
            fftfreq_max=0.5,  # ~2x less coords to rotate
        )

        # phase shift to apply translations
        tilt_dfts_small_shifts = fourier_shift_dft_2d(
            dft=tilt_dfts,
            image_shape=volume.shape[-2:],
            shifts=tilt_shifts_small,
            rfft=True,
            fftshifted=True
        )

        tilt_dfts_large_shifts = fourier_shift_dft_2d(
            dft=tilt_dfts,
            image_shape=volume.shape[-2:],
            shifts=tilt_shifts_large,
            rfft=True,
            fftshifted=True
        )

        # reconstruct
        volume_dft_small_shifts, weights = insert_central_slices_rfft_3d(
            image_rfft=tilt_dfts_small_shifts,
            volume_shape=volume.shape,
            rotation_matrices=tilt_rotation_matrices,  # no rotation offset
            fftfreq_max=0.5
        )

        volume_dft_large_shifts, weights = insert_central_slices_rfft_3d(
            image_rfft=tilt_dfts_large_shifts,
            volume_shape=volume.shape,
            rotation_matrices=tilt_rotation_matrices,  # no rotation offset
            fftfreq_max=0.5
        )

        # reweight reconstructions
        valid_weights = weights > 1e-3
        volume_dft_small_shifts[valid_weights] /= weights[valid_weights]
        volume_dft_large_shifts[valid_weights] /= weights[valid_weights]

        # back to real space
        volume_dft_small_shifts = torch.fft.ifftshift(volume_dft_small_shifts, dim=(-3, -2,))  # actual ifftshift
        volume_dft_small_shifts = torch.fft.irfftn(volume_dft_small_shifts, dim=(-3, -2, -1))
        volume_dft_small_shifts = torch.fft.ifftshift(volume_dft_small_shifts, dim=(-3, -2, -1))  # center in real space

        volume_dft_large_shifts = torch.fft.ifftshift(volume_dft_large_shifts, dim=(-3, -2,))  # actual ifftshift
        volume_dft_large_shifts = torch.fft.irfftn(volume_dft_large_shifts, dim=(-3, -2, -1))
        volume_dft_large_shifts = torch.fft.ifftshift(volume_dft_large_shifts, dim=(-3, -2, -1))  # center in real space

        return torch.real(volume_dft_small_shifts), torch.real(volume_dft_large_shifts)

    def _preprocess(
        self, volume: torch.Tensor
    ) -> torch.Tensor:
        volume = einops.rearrange(volume, "d h w -> 1 1 d h w")
        volume = self._pad_to_target_size(volume)
        volume = self._normalize(volume)
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

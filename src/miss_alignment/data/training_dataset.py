import einops
import numpy as np
import torch
from torch.utils.data import Dataset
import mrcfile
from scipy.spatial.transform import Rotation
from torch_fourier_slice import project_3d_to_2d, backproject_2d_to_3d
from torch_fourier_shift import fourier_shift_image_2d
import torch.nn.functional as F


class MRCDataset(Dataset):
    """
    Dataset for loading MRC files and applying transformations.

    Parameters
    ----------
    mrc_files : list
        List of paths to MRC files.
    target_size : int or tuple
        Target size to pad/crop images to. If an integer, same size is used for all dimensions.
    translation_std : float
        Standard deviation for Gaussian random translations.
    device : str
        Device to use for transformations ('cpu' or 'cuda').
    """

    def __init__(self, mrc_files, target_size=64, device="cpu"):
        self.mrc_files = mrc_files

        if isinstance(target_size, int):
            self.target_size = (target_size, target_size, target_size)
        else:
            self.target_size = tuple(target_size)
        self.device = device

    def __len__(self):
        return len(self.mrc_files)

    def __getitem__(self, idx):
        # Load MRC file
        mrc_path = self.mrc_files[idx]
        with mrcfile.open(mrc_path) as mrc:
            # Convert to torch tensor and move to device
            volume = torch.from_numpy(mrc.data.astype(np.float32)).to(self.device)

        volume = self._preprocess(volume, random_affine=True)

        tilt_angles = Rotation.from_euler("Y", np.arange(-51, 54, 3), degrees=True)
        rotations = torch.tensor(tilt_angles.as_matrix()).float()
        aligned, misaligned = self._generate_reconstructions(volume, rotations)

        return {  # add channel dimension to all output
            "volume": einops.rearrange(volume, "d h w -> 1 d h w"),
            "aligned": einops.rearrange(aligned, "d h w -> 1 d h w"),
            "misaligned": einops.rearrange(misaligned, "d h w -> 1 d h w"),
            "file_path": mrc_path,
        }

    def _generate_reconstructions(
        self, volume: torch.Tensor, matrices: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        tilts = project_3d_to_2d(volume, rotation_matrices=matrices, pad=True)
        smoll_translations = torch.normal(
            mean=0.0,
            std=0.5,
            size=(matrices.shape[0], 2),  # batch of number of tilts
            device=volume.device,
        )
        aligned = backproject_2d_to_3d(
            fourier_shift_image_2d(tilts, smoll_translations),
            rotation_matrices=matrices,
            pad=True,
        )
        big_translations = torch.normal(
            mean=0.0,
            std=1.5,
            size=(matrices.shape[0], 2),  # batch of number of tilts
            device=volume.device,
        )
        misaligned = backproject_2d_to_3d(
            fourier_shift_image_2d(tilts, big_translations),
            rotation_matrices=matrices,
            pad=True,
        )
        return aligned, misaligned

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
            mode="replicate",
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
            std=0.05,  # standard deviation assuming box coords from -1 to 1
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

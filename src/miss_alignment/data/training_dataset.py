from os import PathLike
from pathlib import Path

import random
import einops
import numpy as np
import torch
import tqdm
from torch.utils.data import Dataset
import mrcfile
from scipy.spatial.transform import Rotation as R
from torch_fourier_slice import (
    extract_central_slices_rfft_3d,
    insert_central_slices_rfft_3d,
)
from torch_fourier_shift import fourier_shift_dft_2d, fourier_shift_dft_3d
from torch_grid_utils import fftfreq_grid, sphere
import torch.nn.functional as F

from .shift_generation import generate_shifts, project_shifts_3d_to_2d
from .augmentation import (
    random_contrast,
    random_mirror,
    random_edge_mask,
)


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
        self.target_size = (target_size, target_size, target_size)

        ## make volume dft
        # premultiply by sinc2
        self.sinc2 = (
            torch.sinc(
                fftfreq_grid(
                    image_shape=self.target_size,
                    rfft=False,
                    fftshift=True,
                    norm=True,
                    device="cpu",
                )
            )
            ** 2
        )

        mrc_files = list(self.dataset_directory.glob("*.mrc"))
        self.volumes = []
        for mrc_path in tqdm.tqdm(mrc_files):
            with mrcfile.open(mrc_path) as mrc:
                # Convert to torch tensor and move to device
                volume = torch.from_numpy(mrc.data.astype(np.float32))
            volume = self._preprocess(volume)
            volume = volume * self.sinc2

            # calculate DFT
            volume_dft = torch.fft.fftshift(
                volume, dim=(-3, -2, -1)
            )  # volume center to array origin
            volume_dft = torch.fft.rfftn(volume_dft, dim=(-3, -2, -1))
            volume_dft = torch.fft.fftshift(
                volume_dft,
                dim=(
                    -3,
                    -2,
                ),
            )  # actual fftshift of 3D rfft
            self.volumes += [(mrc_path.stem, volume_dft)]

        if len(self.volumes) == 0:
            raise FileNotFoundError("No MRC files found in the dataset directory.")

        self.train() if train else self.eval()

    def train(self):
        self._is_training = True

    def eval(self):
        self._is_training = False

    def __len__(self):
        return len(self.volumes)

    def __getitem__(self, idx):
        # grab box
        _, volume_dft = self.volumes[idx]

        tilt_angles = R.from_euler(  # TODO differ tilt range, increment
            seq="Y",
            angles=np.linspace(
                -random.randint(40, 70),  # min tilt
                random.randint(40, 70),  # max tilt
                random.randint(30, 50),  # number of increments
                endpoint=True,
            ),
            degrees=True,
        )
        rotations = torch.tensor(tilt_angles.as_matrix()).float()
        aligned, misaligned, aligned_translations, misaligned_translations = (
            self._generate_reconstructions(volume_dft, rotations)
        )

        # randomly create either two positive or two negative examples
        example1 = {
            "volume": self._augment(aligned) if self._is_training else aligned,
            "target": 1,
        }
        example2 = {
            "volume": self._augment(misaligned) if self._is_training else misaligned,
            "target": -1,
        }

        # Create a third example that also goes through _reconstruct
        # Generate a new reconstruction with random 3D shift and rotation
        if random.random() > 0.5:
            # Create a new reconstruction from aligned volume
            example3_volume = self._reconstruct(
                volume_dft,
                rotations,
                aligned_translations,
            )
            example3_volume = self._normalize(example3_volume)
            example3 = {
                "volume": self._augment(example3_volume)
                if self._is_training
                else example3_volume,
                "target": 1,
            }
        else:
            # Create a new reconstruction from misaligned volume
            example3_volume = self._reconstruct(
                volume_dft,
                rotations,
                misaligned_translations,
            )
            example3_volume = self._normalize(example3_volume)
            example3 = {
                "volume": self._augment(example3_volume)
                if self._is_training
                else example3_volume,
                "target": -1,
            }

        data = [example1.values(), example2.values(), example3.values()]
        random.shuffle(data)
        volumes, targets = zip(*data)
        return (
            *(einops.rearrange(v, "d h w -> 1 d h w") for v in volumes),
            torch.tensor(targets),
        )

    def prepare_test_boxes(self, shift_fraction: float = 0.25):
        test_data = []
        for map_name, volume in self.volumes:
            tilt_angles = R.from_euler(
                seq="Y", angles=np.arange(-51, 54, 3), degrees=True
            )
            rotations = torch.tensor(tilt_angles.as_matrix()).float()
            # between 0 and misaligned_std
            misaligned_translations = generate_shifts(
                rotations.shape[0],
                volume.shape[0] * shift_fraction,
            )
            misaligned_translations = project_shifts_3d_to_2d(
                misaligned_translations,
                rotations,
            )
            test_data.append(
                {
                    "map_name": map_name,
                    "volume": volume,
                    "rotations": rotations,
                    "translations": misaligned_translations,
                }
            )
        return test_data

    def _generate_reconstructions(
        self, volume_dft: torch.Tensor, matrices: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        # generate two sets of shifts
        shifts_1 = generate_shifts(  # 3d shifts zyx
            matrices.shape[0],
            self.target_size[0] * 0.25,
        )
        shifts_2 = generate_shifts(  # 3d shifts zyx
            matrices.shape[0],
            self.target_size[0] * 0.25,
        )

        # project the shifts to 2d
        shifts_1 = project_shifts_3d_to_2d(
            shifts_1,
            matrices,
        )
        shifts_2 = project_shifts_3d_to_2d(
            shifts_2,
            matrices,
        )

        # randomly remove x or y shifts
        die_roll = random.random()
        if die_roll < 0.25:  # set y to 0
            shifts_1[:, 0] = 0.0
            shifts_2[:, 0] = 0.0
        if 0.25 <= die_roll < 0.5:  # set x to 0
            shifts_1[:, 1] = 0.0
            shifts_2[:, 1] = 0.0

        # set (mis)alignment based on the \sigma of the generated shifts
        if torch.sum(torch.abs(shifts_1)) > torch.sum(torch.abs(shifts_2)):
            # if torch.pow(shifts_1, 2).sum() > torch.pow(shifts_2, 2).sum():
            misaligned_translations = shifts_1
            aligned_translations = shifts_2
        else:
            misaligned_translations = shifts_2
            aligned_translations = shifts_1

        aligned = self._reconstruct(
            volume_dft,
            matrices,
            aligned_translations,
        )
        misaligned = self._reconstruct(
            volume_dft,
            matrices,
            misaligned_translations,
        )

        aligned = self._normalize(aligned)
        misaligned = self._normalize(misaligned)

        return aligned, misaligned, aligned_translations, misaligned_translations

    def _augment(self, volume: torch.Tensor) -> torch.Tensor:
        if random.random() > 0.5:
            noise_std = random.random() * 1.5
            volume = volume + torch.normal(
                mean=0.0,
                std=noise_std,
                size=volume.shape,
            )
        volume = random_edge_mask(volume)
        # TODO temporarily removed for EMDB dataset
        # volume = random_cube_mask(volume)
        volume = self._normalize(volume)
        volume = random_contrast(volume)
        volume = random_mirror(volume)
        return volume

    def _reconstruct(
        self,
        volume_dft: torch.Tensor,  # (d, h, w)
        tilt_rotation_matrices: torch.Tensor,  # (b, 3, 3)
        tilt_shifts: torch.Tensor,  # (b, 2)
    ):
        # apply random 3d shift
        random_shift = torch.tensor(
            np.random.normal(loc=0, scale=0.1 * self.target_size[-1], size=(3,)),
            dtype=torch.float32,
        )
        volume_dft = fourier_shift_dft_3d(
            dft=volume_dft,
            image_shape=self.target_size,
            shifts=random_shift,
            rfft=True,
            fftshifted=True,
        )

        # get random rotation offset for slice extraction
        random_rotation = torch.tensor(R.random().as_matrix(), dtype=torch.float32)

        # extract tilt dfts
        tilt_dfts = extract_central_slices_rfft_3d(
            volume_rfft=volume_dft,
            image_shape=self.target_size,
            rotation_matrices=random_rotation @ tilt_rotation_matrices,
            fftfreq_max=0.5,  # ~2x less coords to rotate
        )

        # phase shift to apply translations
        tilt_dfts = fourier_shift_dft_2d(
            dft=tilt_dfts,
            image_shape=self.target_size[-2:],
            shifts=tilt_shifts,
            rfft=True,
            fftshifted=True,
        )

        # reconstruct
        volume_dft, weights = insert_central_slices_rfft_3d(
            image_rfft=tilt_dfts,
            volume_shape=self.target_size,
            rotation_matrices=tilt_rotation_matrices,  # no rotation offset
            fftfreq_max=0.5,
        )

        # reweight reconstructions
        valid_weights = weights > 1e-3
        volume_dft[valid_weights] /= weights[valid_weights]

        # back to real space
        volume_dft = torch.fft.ifftshift(
            volume_dft,
            dim=(
                -3,
                -2,
            ),
        )  # actual ifftshift
        volume_dft = torch.fft.irfftn(volume_dft, dim=(-3, -2, -1))
        volume_dft = torch.fft.ifftshift(
            volume_dft, dim=(-3, -2, -1)
        )  # center in real space

        volume_dft = volume_dft / self.sinc2

        return torch.real(volume_dft)

    def _preprocess(self, volume: torch.Tensor) -> torch.Tensor:
        volume = self._mask(volume)  # apply spherical mask
        volume = self._pad_to_target_size(volume)
        volume = self._normalize(volume)
        return volume

    def _mask(self, volume: torch.Tensor) -> torch.Tensor:
        mask = sphere(volume.shape[-1] - 3, tuple(volume.shape), smoothing_radius=3)
        volume = volume * mask
        return volume

    def _normalize(self, volume: torch.Tensor) -> torch.Tensor:
        mean, std = torch.mean(volume), torch.std(volume)
        torch.nan_to_num(volume, nan=float(mean))
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

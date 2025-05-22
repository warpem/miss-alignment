import random
import torch
import einops
import numpy as np
import tqdm
from scipy.spatial.transform import Rotation as R
from torch_fourier_slice import (
    extract_central_slices_rfft_3d,
    insert_central_slices_rfft_3d_multichannel,
)
from torch_fourier_shift import fourier_shift_dft_2d,  fourier_shift_dft_3d
from torch_grid_utils import fftfreq_grid, coordinate_grid

from miss_alignment.data import EMDBDataset


def prep_tilts(
        volume_dft: torch.Tensor,
        tilt_rotation_matrices: torch.Tensor,
        tilt_image_shifts: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    size = volume_dft.shape[-3]
    shape = (size, ) * 3
    # random shift not needed for evaluation
    random_shift = torch.tensor(
        np.random.normal(
            loc=0, scale=.1 * size, size=(3,)
        ),
        dtype=torch.float32
    )
    volume_dft = fourier_shift_dft_3d(
        dft=volume_dft,
        image_shape=shape,
        shifts=random_shift,
        rfft=True,
        fftshifted=True
    )

    # get random rotation offset for slice extraction
    random_rotation = torch.tensor(
        R.random().as_matrix(),
        dtype=torch.float32
    )

    # extract tilt dfts
    tilt_dfts = extract_central_slices_rfft_3d(
        volume_rfft=volume_dft,
        image_shape=shape,
        rotation_matrices=random_rotation @ tilt_rotation_matrices,
        fftfreq_max=0.5,  # ~2x less coords to rotate
    )

    # phase shift to apply translations
    tilt_dfts_shifted = fourier_shift_dft_2d(
        dft=tilt_dfts,
        image_shape=shape[-2:],
        shifts=tilt_image_shifts,
        rfft=True,
        fftshifted=True
    )

    return (
        tilt_dfts,
        tilt_dfts_shifted,
        random_shift,
        random_rotation
    )


def batch_reconstruct(
        tilt_image_dfts: torch.Tensor,
        predicted_shifts: torch.Tensor,
        tilt_rotation_matrices: torch.Tensor,
        image_shape: tuple[int, int],
        volume_shape: tuple[int, int, int],
) -> torch.Tensor:

    predicted_shifts = einops.rearrange(
        predicted_shifts, "n yx -> n 1 yx"
    )
    tilt_dfts_shifted = fourier_shift_dft_2d(
        dft=tilt_image_dfts,
        image_shape=image_shape,
        shifts=predicted_shifts,
        rfft=True,
        fftshifted=True
    )

    # Rest of computation...
    volume_dft, weights = insert_central_slices_rfft_3d_multichannel(
        image_rfft=tilt_dfts_shifted,
        volume_shape=volume_shape,
        rotation_matrices=tilt_rotation_matrices,
        fftfreq_max=0.5
    )

    valid_weights = weights > 1e-3
    volume_dft[..., valid_weights] /= weights[..., valid_weights]

    volume_dft = torch.fft.ifftshift(volume_dft, dim=(-3, -2))
    volume = torch.fft.irfftn(volume_dft, dim=(-3, -2, -1))
    volume = torch.fft.ifftshift(volume, dim=(-3, -2, -1))

    grid = fftfreq_grid(
        image_shape=volume.shape[-3:],
        rfft=False,
        fftshift=True,
        norm=True,
        device=volume.device,
    )
    volume = volume / torch.sinc(grid) ** 2

    volume = torch.real(volume).to(torch.float32)
    return volume


def center_of_mass(volume: torch.Tensor) -> torch.Tensor:
    """Calculate the center of mass of a 3D tensor.

    Parameters
    ----------
    volume : torch.Tensor
        Input 3D tensor representing mass distribution

    Returns
    -------
    torch.Tensor
        Center of mass coordinates [z, y, x]
    """
    device = volume.device
    volume = volume ** 2
    grid = coordinate_grid(volume.shape[-3:], device=device)
    volume = einops.rearrange(volume, "... d h w -> ... d h w 1")
    mass = torch.sum(volume, dim=(-4, -3, -2))
    center_of_mass = torch.sum(grid * volume, dim=(-4, -3, -2)) / mass
    return center_of_mass


def optimize_shifts(
        model,
        tilt_image_dfts: torch.Tensor,
        tilt_rotation_matrices: torch.Tensor,
        gt_com: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, list]:
    """Find shifts to optimize model score.

    Parameters
    ----------
    model: torch.nn.Module
        Model that produces the optimization target.
    tilt_image_dfts: torch.Tensor
        DFTs of tilt images to use for reconstruction.
    tilt_rotation_matrices: torch.Tensor:
        Tilt rotation matrices to use for back projection.

    Returns
    -------
    tuple[torch.Tensor, torch.Tensor, list]
        - Detected translational alignment.
        - List of loss values for each optimization iteration.
    """
    n_tilts = tilt_rotation_matrices.shape[0]
    box_size = tilt_image_dfts.shape[-2]
    image_shape = (box_size, ) * 2
    volume_shape = (box_size, ) * 3
    device = tilt_image_dfts.device

    predicted_shifts = torch.zeros(
        size=(n_tilts, 2),
        dtype=torch.float32,
        device=device,
        requires_grad=True,
    )

    alignment_optimizer = torch.optim.LBFGS(
        [predicted_shifts,],
        line_search_fn="strong_wolfe",
    )

    # Initialize list to store loss values
    loss_values = []

    def closure():
        alignment_optimizer.zero_grad()
        volumes = batch_reconstruct(
            tilt_image_dfts,
            predicted_shifts,
            tilt_rotation_matrices,
            image_shape,
            volume_shape,
        )

        new_com = center_of_mass(volumes)
        distance = torch.sum((new_com - gt_com) ** 2)

        # change channel to batch dimension
        volumes = einops.rearrange(volumes, 'c d h w -> c 1 d h w')

        # Get loss and compute backward pass
        loss = model(volumes) + distance
        loss = loss.mean()
        loss.backward()

        # Store the loss value
        loss_values.append(loss.item())

        return loss

    for _ in range(10):
        alignment_optimizer.step(closure)

    predicted_shifts = predicted_shifts.detach()  # center the shifts
    predicted_shifts = predicted_shifts - predicted_shifts.mean(axis=0)
    volumes = batch_reconstruct(
        tilt_image_dfts=tilt_image_dfts,
        predicted_shifts=predicted_shifts,
        tilt_rotation_matrices=tilt_rotation_matrices,
        image_shape=image_shape,
        volume_shape=volume_shape,
    )

    return volumes, predicted_shifts, loss_values


def get_alignment_optimization_metrics(
        model,
        test_data_directory,
        rotations,
        misalignments,  # this defines the number of iterations
        nboxes: int = 8,
):
    """Get a metric for the models performance on tilt series alignment.

    Parameters
    ----------
    model : Path
        Path to the trained model checkpoint.
    test_data_directory : Path
        Directory containing test data.
    nboxes : int, optional
        Number of boxes to use for optimization, by default 1.
    seed : int, optional
        Random seed for reproducibility, by default 45132.
    iterations : int, optional
        Number of iterations for optimization alignment, by default 50.
        Controls how many sets of maps are optimized for better statistics.
    """
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # initialize the dataset
    dataset = EMDBDataset(test_data_directory, train=False)
    dataset_size = len(dataset)

    n_tilts = rotations.shape[0]
    n_iterations = len(misalignments)

    mean_x_shift_misaligned = 0
    mean_y_shift_misaligned = 0
    mean_x_shift_aligned = 0
    mean_y_shift_aligned = 0
    ground_truth_loss, aligned_loss, misaligned_loss = 0, 0, 0

    for i, misaligned_translations in tqdm.tqdm(enumerate(misalignments)):
        misalignment = (
                misaligned_translations - misaligned_translations.mean(axis=0)
        )

        # Store map data for the JSON report
        ground_truth_tilts, misaligned_tilts = (
            [], []
        )

        # Store map names and indices for the current iteration
        map_indices = [
            random.randint(0, dataset_size - 1) for _ in range(nboxes)
        ]

        for idx in map_indices:
            ground_truth, misaligned, random_shift, random_rotation = (
                prep_tilts(
                    dataset.volumes[idx][1],
                    rotations,
                    misalignment,
                )
            )
            misaligned_tilts += [misaligned]
            ground_truth_tilts += [ground_truth]

            tilts = torch.fft.ifftshift(misaligned, dim=(-2))
            tilts = torch.fft.irfftn(tilts, dim=(-2, -1))
            tilts = torch.fft.ifftshift(tilts, dim=(-2, -1))

        misaligned_tilts = torch.stack(misaligned_tilts)
        misaligned_tilts = einops.rearrange(
            misaligned_tilts, "b n h w -> n b h w"
        )
        misaligned_reconstruction = batch_reconstruct(
            misaligned_tilts,
            torch.zeros((n_tilts, 2)),
            rotations,
            dataset.target_size[-2:],
            dataset.target_size,
        )
        ground_truth_tilts = torch.stack(ground_truth_tilts)
        ground_truth_tilts = einops.rearrange(
            ground_truth_tilts, "b n h w -> n b h w"
        )
        ground_truth_reconstruction = batch_reconstruct(
            ground_truth_tilts,
            torch.zeros((n_tilts, 2)),
            rotations,
            dataset.target_size[-2:],
            dataset.target_size,
        )

        coms = center_of_mass(ground_truth_reconstruction)
        # Call optimize_shifts and get both shifts and loss values
        volumes, shifts, loss_values = optimize_shifts(
            model=model.to(device),  # Use the user-specified device
            tilt_image_dfts=misaligned_tilts.to(device),  # Use the user-specified device
            tilt_rotation_matrices=rotations.to(device),  # Use the user-specified device
            gt_com=coms.to(device),  # Use the user-specified device
        )
        shifts = shifts.cpu()
        volumes = volumes.cpu()

        mag_ma = torch.sum(torch.abs(misalignment), dim=0).tolist()
        mag_a = torch.sum(torch.abs(misalignment + shifts), dim=0).tolist()

        mean_x_shift_misaligned += mag_ma[1]
        mean_y_shift_misaligned += mag_ma[0]
        mean_x_shift_aligned += mag_a[1]
        mean_y_shift_aligned += mag_a[0]
        ground_truth_loss += (
            model(
                einops.rearrange(ground_truth_reconstruction.to(device), 'c d h w -> c 1 d h w')
            ).mean().item()
        )
        misaligned_loss += (
            model(
                einops.rearrange(misaligned_reconstruction.to(device),
                                 'c d h w -> c 1 d h w')
            ).mean().item()
        )
        aligned_loss += (
            model(
                einops.rearrange(volumes.to(device), 'c d h w -> c 1 d h w')
            ).mean().item()
        )

    mean_x_shift_misaligned /= n_iterations
    mean_y_shift_misaligned /= n_iterations
    mean_x_shift_aligned /= n_iterations
    mean_y_shift_aligned /= n_iterations

    ground_truth_loss /= n_iterations
    misaligned_loss /= n_iterations
    aligned_loss /= n_iterations

    return {
        "ground_truth_loss": ground_truth_loss,
        "misaligned_loss": misaligned_loss,
        "aligned_loss": aligned_loss,
        "mean_x_shift_misaligned": mean_x_shift_misaligned,
        "mean_y_shift_misaligned": mean_y_shift_misaligned,
        "mean_x_shift_aligned": mean_x_shift_aligned,
        "mean_y_shift_aligned": mean_y_shift_aligned,
    }
import typer
from pathlib import Path
from typing import Optional

from ._cli import OPTION_PROMPT_KWARGS, cli

import mrcfile
import torch
import einops
import numpy as np
from pytorch_lightning import seed_everything
from torch_tomogram import Tomogram
import matplotlib.pyplot as plt

from miss_alignment.data.shift_generation import project_shifts_3d_to_2d
from miss_alignment.models import MissAlignment
from miss_alignment.data import SHRECDataModule
from miss_alignment.data.io import save_tomogram


def calculate_cross_correlation(
    a: torch.Tensor,
    b: torch.Tensor,
) -> torch.Tensor:
    """
    Calculate the 3D cross correlation between volumes of the same size.

    The position of the maximum relative to the center of the volume gives a shift.
    This is the shift that when applied to `b` best aligns it to `a`.

    Parameters
    ----------
    a : torch.Tensor
        First 3D volume with shape (..., D, H, W)
    b : torch.Tensor
        Second 3D volume with shape (..., D, H, W)

    Returns
    -------
    torch.Tensor
        3D cross-correlation volume
    """
    a = (a - a.mean()) / a.std()
    b = (b - b.mean()) / b.std()
    d, h, w = a.shape[-3:]
    fta = torch.fft.rfftn(a, dim=(-3, -2, -1))
    ftb = torch.fft.rfftn(b, dim=(-3, -2, -1))
    result = fta * torch.conj(ftb)
    result = torch.fft.irfftn(result, dim=(-3, -2, -1), s=(d, h, w))
    result = torch.fft.ifftshift(result, dim=(-3, -2, -1))
    result /= d * h * w  # normalize the result
    return result


def get_shift_from_correlation_image(correlation_image: torch.Tensor) -> torch.Tensor:
    """
    Extract shift from 3D correlation volume.

    The shift should be applied to img2 to align with img1.
    Uses parabolic interpolation for sub-voxel accuracy.

    Parameters
    ----------
    correlation_image : torch.Tensor
        3D correlation volume

    Returns
    -------
    torch.Tensor
        3D shift vector [z, y, x]
    """
    d, h, w = correlation_image.shape
    dtype, device = correlation_image.dtype, correlation_image.device

    image_shape = torch.as_tensor(correlation_image.shape, device=device, dtype=dtype)
    center = torch.divide(image_shape, 2, rounding_mode="floor")

    # Find peak location
    flat_idx = torch.argmax(correlation_image)
    peak_z = flat_idx // (h * w)
    peak_y = (flat_idx % (h * w)) // w
    peak_x = flat_idx % w

    # Check if peak is on border
    if (
        peak_z == 0
        or peak_z == d - 1
        or peak_y == 0
        or peak_y == h - 1
        or peak_x == 0
        or peak_x == w - 1
    ):
        shift = (
            torch.tensor([peak_z, peak_y, peak_x], device=device, dtype=dtype) - center
        )
        return shift

    # Parabolic interpolation in z direction
    f_z0 = correlation_image[peak_z - 1, peak_y, peak_x]
    f_z1 = correlation_image[peak_z, peak_y, peak_x]
    f_z2 = correlation_image[peak_z + 1, peak_y, peak_x]
    subpixel_peak_z = peak_z + 0.5 * (f_z0 - f_z2) / (f_z0 - 2 * f_z1 + f_z2)

    # Parabolic interpolation in y direction
    f_y0 = correlation_image[peak_z, peak_y - 1, peak_x]
    f_y1 = correlation_image[peak_z, peak_y, peak_x]
    f_y2 = correlation_image[peak_z, peak_y + 1, peak_x]
    subpixel_peak_y = peak_y + 0.5 * (f_y0 - f_y2) / (f_y0 - 2 * f_y1 + f_y2)

    # Parabolic interpolation in x direction
    f_x0 = correlation_image[peak_z, peak_y, peak_x - 1]
    f_x1 = correlation_image[peak_z, peak_y, peak_x]
    f_x2 = correlation_image[peak_z, peak_y, peak_x + 1]
    subpixel_peak_x = peak_x + 0.5 * (f_x0 - f_x2) / (f_x0 - 2 * f_x1 + f_x2)

    subpixel_shift = (
        torch.tensor(
            [subpixel_peak_z, subpixel_peak_y, subpixel_peak_x],
            device=device,
            dtype=dtype,
        )
        - center
    )

    return subpixel_shift


def optimize_shifts_shrec(
    model,
    tilt_series,
    positions,
    patch_size,
    device,
):
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
    tilt_series.to(device)
    model.to(device)
    tilt_series.sample_translations.requires_grad_()

    alignment_optimizer = torch.optim.LBFGS(
        [
            tilt_series.sample_translations,
        ],
        line_search_fn="strong_wolfe",
    )

    # Initialize list to store loss values
    loss_values = []

    def closure():
        alignment_optimizer.zero_grad()

        volumes = []
        for zyx in positions:
            subtomo = tilt_series.reconstruct_subvolume(zyx, patch_size)
            mean, std = torch.mean(subtomo), torch.std(subtomo)
            subtomo = (subtomo - mean) / std
            volumes.append(subtomo)
        volumes = torch.stack(volumes)

        # new_com = center_of_mass(volumes)
        # distance = torch.sum((new_com - gt_com) ** 2)

        # change channel to batch dimension
        volumes = einops.rearrange(volumes, "c d h w -> c 1 d h w")

        # Get loss and compute backward pass
        loss = model(volumes)
        loss = loss.mean()
        loss.backward()

        # Store the loss value
        print(loss.item())
        loss_values.append(loss.item())

        return loss

    n_iters = 1  # 5 iterations should give convergence
    for x in range(n_iters):
        print(f"Optimizing... {x + 1}/{n_iters}")
        alignment_optimizer.step(closure)

    # remove gradients
    tilt_series.sample_translations.detach_()
    tilt_series.to("cpu")
    model.to("cpu")

    return tilt_series, loss_values


def evaluate_tilt_series(
    model: MissAlignment,
    tilt_series_name: str,
    tilt_series_raw: Tomogram,
    patches_per_dim: tuple[int, int, int],
    patch_size: int,
    tomogram_shape: tuple[int, int, int],
    output_directory: Path,
    tilt_series_ground_truth: Optional[Tomogram] = None,
    device: str = "cpu",
) -> Tomogram:
    d, h, w = tomogram_shape
    dc, hc, wc = d // 2, h // 2, w // 2

    offset = patch_size // 2

    position_grid = []
    zs = np.linspace(offset, d - offset, patches_per_dim[0]) - dc
    ys = np.linspace(offset, h - offset, patches_per_dim[1]) - hc
    xs = np.linspace(offset, w - offset, patches_per_dim[2]) - wc
    for z in zs:
        for y in ys:
            for x in xs:
                position_grid.append((z, y, x))

    ground_truth_alignment = tilt_series_ground_truth.sample_translations
    n_tilts, _, _ = tilt_series_ground_truth.images.shape

    ground_truth_reconstruction = tilt_series_ground_truth.reconstruct_tomogram(
        tomogram_shape, 128
    )

    mean_diff_initial = torch.abs(
        ground_truth_alignment - tilt_series_raw.sample_translations
    )
    initial_reconstruction = tilt_series_raw.reconstruct_tomogram(tomogram_shape, 128)

    tilt_series, loss = optimize_shifts_shrec(
        model,
        tilt_series_raw,  # is modified in-place
        position_grid,
        patch_size,
        device,
    )

    aligned_reconstruction = tilt_series.reconstruct_tomogram(tomogram_shape, 128)

    correlation = calculate_cross_correlation(
        ground_truth_reconstruction, aligned_reconstruction
    )
    shift_3d = get_shift_from_correlation_image(correlation)
    shift_3d = -1 * shift_3d  # get the forward shift for the imaging model
    print(shift_3d)
    shift_3d = shift_3d.repeat(n_tilts, 1)

    shifts_2d = project_shifts_3d_to_2d(
        shift_3d, tilt_series.projection_matrices[..., :3, :3]
    )
    tilt_series.sample_translations += shifts_2d

    mean_diff_final = torch.abs(
        ground_truth_alignment - tilt_series.sample_translations
    )

    fig, ax = plt.subplots(1, 2)
    ax[0].plot(mean_diff_initial[:, 0], label="initial y")
    ax[0].plot(mean_diff_final[:, 0], label="final y")
    ax[1].plot(mean_diff_initial[:, 1], label="initial x")
    ax[1].plot(mean_diff_final[:, 1], label="final x")
    ax[0].legend()
    ax[1].legend()

    plt.savefig(output_directory / f"{tilt_series_name}_yx_diff_per_tilt.png")

    aligned_reconstruction = tilt_series.reconstruct_tomogram(tomogram_shape, 128)

    mrcfile.write(
        output_directory / f"{tilt_series_name}_before.mrc",
        initial_reconstruction.detach().numpy(),
        voxel_size=10,
        overwrite=True,
    )
    mrcfile.write(
        output_directory / f"{tilt_series_name}_after.mrc",
        aligned_reconstruction.detach().numpy(),
        voxel_size=10,
        overwrite=True,
    )

    save_tomogram(tilt_series, output_directory / f"{tilt_series_name}.pickle")

    return tilt_series


@cli.command(name="align_shrec", no_args_is_help=True)
def align_shrec(
    model_checkpoint: Path = typer.Option(..., **OPTION_PROMPT_KWARGS),
    test_data_directory: Path = typer.Option(..., **OPTION_PROMPT_KWARGS),
    test_data_ground_truth: Path = typer.Option(..., **OPTION_PROMPT_KWARGS),
    output_directory: Path = typer.Option(..., **OPTION_PROMPT_KWARGS),
    nboxes: int = 1,
    seed: int = 45132,
    patch_size: int = 64,
    device: str = typer.Option(
        "cuda:0", help="Device to run the model on (e.g., 'cuda:0', 'cpu')"
    ),
    iterations: int = typer.Option(
        50, help="Number of iterations for optimization alignment"
    ),
    misalignments_file: Path = typer.Option(
        None, help="Path to a previously generated misalignments JSON file"
    ),
) -> None:
    """Optimize alignment of tilt series using a trained model.

    Parameters
    ----------
    model_checkpoint : Path
        Path to the trained model checkpoint.
    test_data_directory : Path
        Directory containing test data.
    test_data_ground_truth : Path
        Directory containing the ground truth alignments.
    output_directory : Path
        Directory where results will be saved.
    nboxes : int, optional
        Number of boxes to use for optimization, by default 1.
    seed : int, optional
        Random seed for reproducibility, by default 45132.
    device : str, optional
        Device to run the model on (e.g., 'cuda:0', 'cpu'), by default "cuda:0".
    iterations : int, optional
        Number of iterations for optimization alignment, by default 50.
        Controls how many sets of maps are optimized for better statistics.
    misalignments_file : Path, optional
        Path to a previously generated misalignments JSON file, by default None.
        If provided, the misalignments from this file will be used instead of generating new ones.
    """
    seed_everything(seed, workers=True)
    torch.set_printoptions(precision=2, sci_mode=False)

    # get model
    model = MissAlignment.load_from_checkpoint(model_checkpoint, map_location="cpu")
    model.freeze()

    # initialize the dataset
    datamodule = SHRECDataModule(
        test_data_directory,
        dataset_type="SHREC",
    )
    datamodule.setup(stage="validate")
    ground_truth = SHRECDataModule(
        test_data_ground_truth,
        dataset_type="SHREC",
    )
    ground_truth.setup(stage="validate")
    tomograms = datamodule.test_dataset.tomos
    tomograms_ground_truth = ground_truth.test_dataset.tomos

    for test, ground_truth in zip(tomograms, tomograms_ground_truth):
        file_path, tilt_series = test
        file_name = file_path.stem
        _, tilt_series_ground_truth = ground_truth
        evaluate_tilt_series(
            model,
            file_name,
            tilt_series,
            (1, 4, 4),
            128,
            (180, 512, 512),
            output_directory,
            tilt_series_ground_truth=tilt_series_ground_truth,
            device="cpu",
        )

    return None

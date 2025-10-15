from pathlib import Path
from typing import Optional

import numpy as np
import mrcfile
import torch
import einops
import matplotlib.pyplot as plt
from torch_tomogram import Tomogram
from itertools import chain

from miss_alignment.data.io import save_tomogram_to_pickle, \
    read_tomogram_from_pickle
from miss_alignment.data.shift_generation import project_shifts_3d_to_2d
from miss_alignment.models import MissAlignment


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


def optimize_shifts_local(
    model,
    tilt_series,
    positions,
    patch_size,
    device,
    tomogram_shape,
    image_warp_grid,
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
    from torch_cubic_spline_grids import CubicBSplineGrid3d

    tilt_series.to(device)
    model.to(device)
    model.freeze()
    model.eval()

    d, h, w = tomogram_shape
    dc, hc, wc = d // 2, h // 2, w // 2
    n_tilts, _ = tilt_series.sample_translations.shape
    tilt_idx = torch.linspace(0, 1, n_tilts)

    # store the initial tilt_series alignment
    initial_alignment = tilt_series.sample_translations.clone()

    # create the alignment parameters
    shifts_y = CubicBSplineGrid3d(resolution=image_warp_grid)
    shifts_x = CubicBSplineGrid3d(resolution=image_warp_grid)
    alignment_optimizer = torch.optim.LBFGS(
        chain(shifts_y.parameters(), shifts_x.parameters()),
        line_search_fn="strong_wolfe",
    )

    # Initialize list to store loss values
    loss_values = []
    patches = [None,]

    def closure():
        alignment_optimizer.zero_grad()

        volumes = []
        for zyx in positions:
            # convert to [0, 1] for accessing the spline grids
            z, y, x = zyx
            z, y, x = (z + dc) / d, (y + hc) / h, (x + wc) / w
            shift_idx = torch.stack((tilt_idx, torch.tensor([y,] * n_tilts),
                         torch.tensor([x,] * n_tilts))).T
            sub_shifts = torch.cat((shifts_y(shift_idx), shifts_x(
                shift_idx)), dim=1)
            sub_shifts = sub_shifts.to(device)
            tilt_series.sample_translations = initial_alignment + sub_shifts

            subtomo = tilt_series.reconstruct_subvolume(zyx, patch_size)
            mean, std = torch.mean(subtomo), torch.std(subtomo)
            subtomo = (subtomo - mean) / std
            volumes.append(subtomo)
        volumes = torch.stack(volumes)

        # change channel to batch dimension
        volumes = einops.rearrange(volumes, "b d h w -> b 1 d h w")

        # Get loss and compute backward pass
        loss = model(volumes)
        loss = loss.mean()
        loss.backward()

        print(loss.item())

        # Store the loss value
        loss_values.append(loss.item())
        patches[0] = volumes.detach().cpu()

        return loss

    n_iters = 1  # 5 iterations should give convergence
    for x in range(n_iters):
        print(f"Optimizing... {x + 1}/{n_iters}")
        alignment_optimizer.step(closure)

    # remove gradients
    # tilt_series.sample_translations = initial_alignment + shifts.detach()
    tilt_series.to("cpu")
    model.to("cpu")

    return (shifts_y, shifts_x), patches[0]

    # return tilt_series, loss_values


def optimize_shifts(
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
    model.freeze()
    model.eval()

    # store the initial tilt_series alignment
    initial_alignment = tilt_series.sample_translations.clone()

    # create the alignment parameters
    shifts = torch.zeros_like(
        tilt_series.sample_translations,
        requires_grad=True,
        device=device,
    )
    alignment_optimizer = torch.optim.LBFGS(
        [shifts],
        line_search_fn="strong_wolfe",
    )

    # set the reference image index, the tilt image closest to 0 degrees
    # reference_idx = torch.abs(tilt_series.tilt_angles).argmin().item()
    # Create mask outside the closure
    # mask = torch.ones_like(shifts, device=device)
    # mask[reference_idx] = 0.0

    # Initialize list to store loss values
    loss_values = []

    def closure():
        alignment_optimizer.zero_grad()

        # update the alignments
        # masked_shifts = shifts * mask
        tilt_series.sample_translations = initial_alignment + shifts

        volumes = []
        for zyx in positions:
            subtomo = tilt_series.reconstruct_subvolume(zyx, patch_size)
            mean, std = torch.mean(subtomo), torch.std(subtomo)
            subtomo = (subtomo - mean) / std
            volumes.append(subtomo)
        volumes = torch.stack(volumes)

        # change channel to batch dimension
        volumes = einops.rearrange(volumes, "b d h w -> b 1 d h w")

        # Get loss and compute backward pass
        loss = model(volumes)
        loss = loss.mean()
        loss.backward()

        # Store the loss value
        loss_values.append(loss.item())

        return loss

    n_iters = 1  # 5 iterations should give convergence
    for x in range(n_iters):
        print(f"Optimizing... {x + 1}/{n_iters}")
        alignment_optimizer.step(closure)

    # remove gradients
    tilt_series.sample_translations = initial_alignment + shifts.detach()
    tilt_series.to("cpu")
    model.to("cpu")

    return tilt_series, loss_values


def evaluate_tilt_series(
    model_checkpoint: Path,
    tilt_series: Path,
    patches_per_dim: tuple[int, int, int],
    patch_size: int,
    tomogram_shape: tuple[int, int, int],
    output_directory: Path,
    tilt_series_ground_truth: Optional[Path] = None,
    device: str = "cpu",
) -> tuple[Tomogram, list[float]]:
    # load the best model and run alignment optimization
    model = MissAlignment.load_from_checkpoint(
        model_checkpoint, map_location='cpu',
    )

    # load tilt_series and set its name for output
    tilt_series_name = tilt_series.stem
    tilt_series = read_tomogram_from_pickle(tilt_series)

    d, h, w = tomogram_shape
    dc, hc, wc = d // 2, h // 2, w // 2  # reconstruction center
    pd, ph, pw = patches_per_dim

    # get the reconstruction positions to optimize over
    zs = ((torch.arange(pd) + .5) / pd) * d - dc
    ys = ((torch.arange(ph) + .5) / ph) * h - hc
    xs = ((torch.arange(pw) + .5) / pw) * w - wc

    position_grid = (
        torch.stack(torch.meshgrid(zs, ys, xs, indexing='ij'), dim=-1)
    )
    # ensure we get a list of zyx positions!!
    position_grid = einops.rearrange(position_grid, 'd h w zyx -> (d h w) zyx')
    
    # store initial translations, before tilt-series is modifed in-place
    initial_translations = tilt_series.sample_translations.clone()

    print(f"Aligning {tilt_series_name} on {device}")
    # return optimize_shifts_local(
    #     model,
    #     tilt_series,  # is modified in-place
    #     position_grid,
    #     patch_size,
    #     device,
    #     tomogram_shape,
    # )
    tilt_series, loss = optimize_shifts(
        model,
        tilt_series,  # is modified in-place
        position_grid,
        patch_size,
        device,
    )

    if tilt_series_ground_truth is not None:
        print(f"Centering alignment relative to ground truth...")
        
        # move back to device for faster reconstruction/cross-correlation
        tilt_series.to(device)
        aligned_reconstruction = tilt_series.reconstruct_tomogram(
            tomogram_shape, 128
        )

        tilt_series_ground_truth = read_tomogram_from_pickle(tilt_series_ground_truth)
        tilt_series_ground_truth.to(device)
        ground_truth_alignment = tilt_series_ground_truth.sample_translations
        n_tilts, _, _ = tilt_series_ground_truth.images.shape

        ground_truth_reconstruction = tilt_series_ground_truth.reconstruct_tomogram(
            tomogram_shape, 128
        )

        mean_diff_initial = torch.abs(
            ground_truth_alignment - initial_translations
        ).cpu()

        correlation = calculate_cross_correlation(
            ground_truth_reconstruction, aligned_reconstruction
        )
        shift_3d = get_shift_from_correlation_image(correlation)
        shift_3d = -1 * shift_3d  # get the forward shift for the imaging model
        shift_3d = shift_3d.repeat(n_tilts, 1)

        shifts_2d = project_shifts_3d_to_2d(
            shift_3d, tilt_series.projection_matrices[..., 1:3, :3]
        )
        tilt_series.sample_translations += shifts_2d

        mean_diff_final = torch.abs(
            ground_truth_alignment - tilt_series.sample_translations
        ).cpu()

        fig, ax = plt.subplots(1, 2)
        ax[0].plot(mean_diff_initial[:, 0], label="initial y")
        ax[0].plot(mean_diff_final[:, 0], label="final y")
        ax[1].plot(mean_diff_initial[:, 1], label="initial x")
        ax[1].plot(mean_diff_final[:, 1], label="final x")
        ax[0].legend()
        ax[1].legend()

        plt.savefig(output_directory / f"{tilt_series_name}_yx_diff_per_tilt.png")

    # mrcfile.write(
    #     output_directory / f"{tilt_series_name}_reconstruction.mrc",
    #     aligned_reconstruction.detach().numpy(),
    #     voxel_size=10,
    #     overwrite=True,
    # )
    tilt_series.to('cpu')
    save_tomogram_to_pickle(
        tilt_series, output_directory / f"{tilt_series_name}.pickle"
    )

    return tilt_series, loss

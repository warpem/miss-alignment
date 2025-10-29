from pathlib import Path
from typing import Optional

import functools
import torch
import einops
import matplotlib.pyplot as plt
from torch_tomogram import Tomogram
from itertools import chain
from warpylib.cubic_grid import CubicGrid
from warpylib import TiltSeries

from miss_alignment.data.io import TiltSeriesData
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


def optimize_shifts_image_warp(
        model: MissAlignment,
        tilt_series: TiltSeries,
        images: torch.Tensor,
        pixel_size: float,
        positions: torch.Tensor,
        patch_size: int,
        image_warp_grid: tuple[int, int, int],
        apply_ctf: bool,
        device: str | torch.device,
):
    """Find shifts with image warp to minimize model score.

    Parameters
    ----------


    Returns
    -------
    tuple[torch.Tensor, list]
        - Detected translational alignment.
        - List of loss values for each optimization iteration.
    """
    # move all modules to device in place
    tilt_series.to(device)
    model.to(device)
    model.freeze()
    model.eval()
    # move images to device
    images = images.to(device)

    # movement grids - these should receive gradients
    n_params = functools.reduce(lambda x, y: x * y, image_warp_grid)
    movement_x_leaf = torch.zeros(n_params, requires_grad=True)
    tilt_series.grid_movement_x = CubicGrid(image_warp_grid, movement_x_leaf)
    movement_y_leaf = torch.zeros(n_params, requires_grad=True)
    tilt_series.grid_movement_y = CubicGrid(image_warp_grid, movement_y_leaf)

    # set the optimizer and parameters
    alignment_optimizer = torch.optim.LBFGS(
        [movement_x_leaf, movement_y_leaf],
        line_search_fn="strong_wolfe",
    )

    # Initialize list to store loss values
    loss_values = []

    def closure():
        alignment_optimizer.zero_grad()

        # reconstruct subvolumes
        subvolumes = tilt_series.reconstruct_subvolumes_single(
            tilt_data=images,
            coords=positions,
            pixel_size=pixel_size,
            size=patch_size,
            apply_ctf=apply_ctf,
        )
        # ensure normalization per tilt
        subvolumes -= einops.reduce(
            subvolumes, "n d h w -> n 1 1 1", reduction="mean"
        )
        subvolumes /= torch.std(subvolumes, dim=(-3, -2, -1), keepdim=True)

        # change channel to batch dimension
        subvolumes = einops.rearrange(subvolumes, "b d h w -> b 1 d h w")

        # Get loss and compute backward pass
        loss = model(subvolumes)
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
    tilt_series.grid_movement_x.values = (
        tilt_series.grid_movement_x.values.detach()
    )
    tilt_series.grid_movement_y.values = (
        tilt_series.grid_movement_y.values.detach()
    )
    # move back because there were modified in-place
    tilt_series.to("cpu")
    model.to("cpu")

    return tilt_series, loss_values


def optimize_shifts_global(
        model: MissAlignment,
        tilt_series: TiltSeries,
        images: torch.Tensor,
        pixel_size: float,
        positions: torch.Tensor,
        patch_size: int,
        apply_ctf: bool,
        device: str | torch.device,
):
    """Find shifts to optimize model score.

    Parameters
    ----------


    Returns
    -------
    tuple[torch.Tensor, list]
        - Detected translational alignment.
        - List of loss values for each optimization iteration.
    """
    # move all modules to device in place
    tilt_series.to(device)
    model.to(device)
    model.freeze()
    model.eval()
    # move images to device
    images = images.to(device)

    # store the initial tilt_series alignment
    initial_tilt_axis_offset_y = tilt_series.tilt_axis_offset_y.clone()
    initial_tilt_axis_offset_x = tilt_series.tilt_axis_offset_x.clone()

    # create the alignment parameters
    shifts_y = torch.zeros_like(
        initial_tilt_axis_offset_x,
        requires_grad=True,
        device=device,
    )
    shifts_x = torch.zeros_like(
        initial_tilt_axis_offset_x,
        requires_grad=True,
        device=device,
    )
    alignment_optimizer = torch.optim.LBFGS(
        [shifts_y, shifts_x],
        line_search_fn="strong_wolfe",
    )

    # Initialize list to store loss values
    loss_values = []

    def closure():
        alignment_optimizer.zero_grad()

        # update the alignments
        tilt_series.tilt_axis_offset_y = (
            initial_tilt_axis_offset_y + shifts_y
        )
        tilt_series.tilt_axis_offset_x = (
            initial_tilt_axis_offset_x + shifts_x
        )

        # reconstruct subvolumes
        subvolumes = tilt_series.reconstruct_subvolumes_single(
            tilt_data=images,
            coords=positions,
            pixel_size=pixel_size,
            size=patch_size,
            apply_ctf=apply_ctf,
        )
        # ensure normalization per tilt
        subvolumes -= einops.reduce(
            subvolumes, "n d h w -> n 1 1 1", reduction="mean"
        )
        subvolumes /= torch.std(subvolumes, dim=(-3, -2, -1), keepdim=True)

        # change channel to batch dimension
        subvolumes = einops.rearrange(subvolumes, "b d h w -> b 1 d h w")

        # Get loss and compute backward pass
        loss = model(subvolumes)
        loss = loss.mean()
        loss.backward()

        # Store the loss value
        loss_values.append(loss.item())

        return loss

    n_iters = 1  # 5 iterations should give convergence
    for x in range(n_iters):
        alignment_optimizer.step(closure)

    # remove gradients
    tilt_series.tilt_axis_offset_y = (
        initial_tilt_axis_offset_y + shifts_y.detach()
    )
    tilt_series.tilt_axis_offset_x = (
        initial_tilt_axis_offset_x + shifts_x.detach()
    )
    # move back because there were modified in-place
    tilt_series.to("cpu")
    model.to("cpu")

    return tilt_series, loss_values


def evaluate_tilt_series(
        model_checkpoint_path: Path,
        tilt_series_path: Path,
        patches_per_dim: tuple[int, int, int],
        patch_size: int,
        output_directory: Path,
        image_warp_grid: Optional[tuple[int, int, int]] = None,
        apply_ctf: bool = True,
        ground_truth_path: Optional[Path] = None,
        device: str = "cpu",
) -> tuple[Path, list[float]]:
    # load the best model and run alignment optimization
    model = MissAlignment.load_from_checkpoint(
        model_checkpoint_path, map_location='cpu',
    )

    # load tilt_series and set its name for output
    tilt_series_name = tilt_series_path.stem
    tilt_series_data = TiltSeriesData.from_json(tilt_series_path)
    tilt_series, images, pixel_size = tilt_series_data.load_metadata_and_stack()

    x, y, z = tilt_series.volume_dimensions_physical
    px, py, pz = patches_per_dim

    # get the reconstruction positions to optimize over
    zs = ((torch.arange(pz) + .5) / pz) * z
    ys = ((torch.arange(py) + .5) / py) * y
    xs = ((torch.arange(px) + .5) / px) * x

    position_grid = (
        torch.stack(torch.meshgrid(xs, ys, zs, indexing='ij'), dim=-1)
    )
    # ensure we get a list of xyz positions!!
    position_grid = einops.rearrange(position_grid, 'd h w xyz -> (d h w) xyz')
    
    # store initial translations, before tilt-series is modified in-place
    #initial_translations = tilt_series.sample_translations.clone()

    # determine whether to run global or local alignment
    if image_warp_grid is not None:
        tilt_series, loss = optimize_shifts_image_warp(
            model,
            tilt_series,
            images,
            pixel_size,
            position_grid,
            patch_size,
            image_warp_grid,
            apply_ctf,
            device,
        )
    else:
        tilt_series, loss = optimize_shifts_global(
            model,
            tilt_series,
            images,
            pixel_size,
            position_grid,
            patch_size,
            apply_ctf,
            device,
        )

    if ground_truth_path is not None:
        print(f"Centering alignment relative to ground truth...")
        
        # move back to device for faster reconstruction/cross-correlation
        tilt_series.to(device)
        initial_translations = initial_translations.to(device)
        aligned_reconstruction = tilt_series.reconstruct_tomogram(
            tomogram_shape, 128
        )

        tilt_series_ground_truth = read_tomogram_from_pickle(ground_truth_path)
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

    xml_out = output_directory / (tilt_series_data.xml_filename + '.xml')
    json_out = output_directory / (tilt_series_data.xml_filename + '.json')
    new_tilt_series_data = (
        tilt_series_data.replace(xml_metadata_path=xml_out,)
    )
    new_tilt_series_data.save_metadata_to_xml(tilt_series)
    new_tilt_series_data.to_json(json_out)

    return json_out, loss

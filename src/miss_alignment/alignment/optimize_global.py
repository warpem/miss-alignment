"""Global shift optimization for tilt series alignment.

This module provides the core optimization function for per-tilt shifts
and warping grids (2D and 3D).
"""

import math

import einops
import torch
from warpylib import TiltSeries
from warpylib.cubic_grid import CubicGrid

from miss_alignment.models import MissAlignment


class AlignmentNanError(Exception):
    """Raised when NaN values are detected during alignment optimization."""

    pass


def optimize_shifts(
    model: MissAlignment,
    tilt_series: TiltSeries,
    images: torch.Tensor,
    pixel_size: float,
    positions: torch.Tensor,
    setting: str | tuple[int, int] | tuple[int, int, int, int] = "global",
    patch_size: int = 96,
    batch_size: int = 16,
    apply_ctf: bool = True,
    device: str | torch.device = "cpu",
    max_retries: int = 3,
):
    """Find shifts to optimize model score.

    Parameters
    ----------
    model : MissAlignment
        Trained model for scoring reconstructions.
    tilt_series : TiltSeries
        Tilt series to optimize.
    images : torch.Tensor
        Preprocessed tilt images.
    pixel_size : float
        Pixel size in Angstroms.
    positions : torch.Tensor
        3D positions to reconstruct and evaluate.
    setting : str | tuple
        Type of alignment to run:
        - 'global': optimizes a single shift per image
        - tuple(int, int) e.g. (3, 3): a single 2D field per tilt image
        - tuple(int, int, int, int) e.g. (3, 3, 2, 10): a volume warp grid
    patch_size : int
        Size of reconstruction patches.
    batch_size : int
        Batch size for reconstruction.
    apply_ctf : bool
        Whether to apply CTF correction.
    device : str | torch.device
        Device to run optimization on.
    max_retries : int
        Maximum number of retry attempts if NaN is encountered.

    Returns
    -------
    tuple[TiltSeries, list[float]]
        Optimized tilt series and list of loss values.
    """
    # Store original tilt series state in case all retries fail
    original_tilt_axis_offset_y = tilt_series.tilt_axis_offset_y.clone()
    original_tilt_axis_offset_x = tilt_series.tilt_axis_offset_x.clone()

    # Store original grid states if applicable
    # We need to store complete grid state (dimensions, values, margins)
    # because resize() changes the grid structure
    if setting != "global" and len(setting) == 2:
        has_grid_x = hasattr(tilt_series.grid_movement_x, "values")
        has_grid_y = hasattr(tilt_series.grid_movement_y, "values")
        original_grid_x = (
            {
                "dimensions": tilt_series.grid_movement_x.dimensions,
                "values": tilt_series.grid_movement_x.values.clone(),
                "margins": tilt_series.grid_movement_x.margins,
            }
            if has_grid_x
            else None
        )
        original_grid_y = (
            {
                "dimensions": tilt_series.grid_movement_y.dimensions,
                "values": tilt_series.grid_movement_y.values.clone(),
                "margins": tilt_series.grid_movement_y.margins,
            }
            if has_grid_y
            else None
        )
    elif setting != "global" and len(setting) == 4:
        has_grid_x = hasattr(tilt_series.grid_volume_warp_x, "values")
        has_grid_y = hasattr(tilt_series.grid_volume_warp_y, "values")
        has_grid_z = hasattr(tilt_series.grid_volume_warp_z, "values")
        original_grid_x = (
            {
                "dimensions": tilt_series.grid_volume_warp_x.dimensions,
                "values": tilt_series.grid_volume_warp_x.values.clone(),
                "margins": tilt_series.grid_volume_warp_x.margins,
            }
            if has_grid_x
            else None
        )
        original_grid_y = (
            {
                "dimensions": tilt_series.grid_volume_warp_y.dimensions,
                "values": tilt_series.grid_volume_warp_y.values.clone(),
                "margins": tilt_series.grid_volume_warp_y.margins,
            }
            if has_grid_y
            else None
        )
        original_grid_z = (
            {
                "dimensions": tilt_series.grid_volume_warp_z.dimensions,
                "values": tilt_series.grid_volume_warp_z.values.clone(),
                "margins": tilt_series.grid_volume_warp_z.margins,
            }
            if has_grid_z
            else None
        )

    # Use highest precision for optimization to avoid NaN issues
    # Training uses "medium" and 16-mixed, but optimization needs full precision
    original_precision = torch.get_float32_matmul_precision()
    torch.set_float32_matmul_precision("highest")

    # Retry loop
    retries_left = max_retries
    while retries_left > 0:
        try:
            return _optimize_shifts_inner(
                model=model,
                tilt_series=tilt_series,
                images=images,
                pixel_size=pixel_size,
                positions=positions,
                setting=setting,
                patch_size=patch_size,
                batch_size=batch_size,
                apply_ctf=apply_ctf,
                device=device,
                original_precision=original_precision,
            )
        except AlignmentNanError:
            retries_left -= 1
            if retries_left > 0:
                # Reset tilt series to original state before retry
                tilt_series.tilt_axis_offset_y = original_tilt_axis_offset_y.clone()
                tilt_series.tilt_axis_offset_x = original_tilt_axis_offset_x.clone()

                if setting != "global" and len(setting) == 2:
                    if original_grid_x is not None:
                        tilt_series.grid_movement_x = CubicGrid(
                            dimensions=original_grid_x["dimensions"],
                            values=original_grid_x["values"].clone(),
                            margins=original_grid_x["margins"],
                        )
                    if original_grid_y is not None:
                        tilt_series.grid_movement_y = CubicGrid(
                            dimensions=original_grid_y["dimensions"],
                            values=original_grid_y["values"].clone(),
                            margins=original_grid_y["margins"],
                        )
                elif setting != "global" and len(setting) == 4:
                    if original_grid_x is not None:
                        tilt_series.grid_volume_warp_x = CubicGrid(
                            dimensions=original_grid_x["dimensions"],
                            values=original_grid_x["values"].clone(),
                            margins=original_grid_x["margins"],
                        )
                    if original_grid_y is not None:
                        tilt_series.grid_volume_warp_y = CubicGrid(
                            dimensions=original_grid_y["dimensions"],
                            values=original_grid_y["values"].clone(),
                            margins=original_grid_y["margins"],
                        )
                    if original_grid_z is not None:
                        tilt_series.grid_volume_warp_z = CubicGrid(
                            dimensions=original_grid_z["dimensions"],
                            values=original_grid_z["values"].clone(),
                            margins=original_grid_z["margins"],
                        )
                print(f"Retrying optimization... (retries left: {retries_left})")

    # All retries failed, restore original state and return failure
    tilt_series.tilt_axis_offset_y = original_tilt_axis_offset_y
    tilt_series.tilt_axis_offset_x = original_tilt_axis_offset_x

    if setting != "global" and len(setting) == 2:
        if original_grid_x is not None:
            tilt_series.grid_movement_x = CubicGrid(
                dimensions=original_grid_x["dimensions"],
                values=original_grid_x["values"],
                margins=original_grid_x["margins"],
            )
        if original_grid_y is not None:
            tilt_series.grid_movement_y = CubicGrid(
                dimensions=original_grid_y["dimensions"],
                values=original_grid_y["values"],
                margins=original_grid_y["margins"],
            )
    elif setting != "global" and len(setting) == 4:
        if original_grid_x is not None:
            tilt_series.grid_volume_warp_x = CubicGrid(
                dimensions=original_grid_x["dimensions"],
                values=original_grid_x["values"],
                margins=original_grid_x["margins"],
            )
        if original_grid_y is not None:
            tilt_series.grid_volume_warp_y = CubicGrid(
                dimensions=original_grid_y["dimensions"],
                values=original_grid_y["values"],
                margins=original_grid_y["margins"],
            )
        if original_grid_z is not None:
            tilt_series.grid_volume_warp_z = CubicGrid(
                dimensions=original_grid_z["dimensions"],
                values=original_grid_z["values"],
                margins=original_grid_z["margins"],
            )

    # Restore original precision setting
    torch.set_float32_matmul_precision(original_precision)

    # Return original tilt series with failure loss
    return tilt_series, [float("inf")]


def _optimize_shifts_inner(
    model: MissAlignment,
    tilt_series: TiltSeries,
    images: torch.Tensor,
    pixel_size: float,
    positions: torch.Tensor,
    setting: str | tuple[int, int] | tuple[int, int, int, int],
    patch_size: int,
    batch_size: int,
    apply_ctf: bool,
    device: str | torch.device,
    original_precision: str,
):
    """Inner optimization function that can raise AlignmentNanError.

    Returns
    -------
    tuple[TiltSeries, list[float]]
        Optimized tilt series and list of loss values.
    """
    # move all modules to device in place
    tilt_series.to(device)
    model.to(device)
    model.freeze()
    model.eval()
    # move images to device
    images = images.to(device)

    parameters = None
    if setting == "global":
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
        parameters = [shifts_y, shifts_x]
    elif len(setting) == 2:  # TODO add case of starting from existent grid
        # movement grids - these should receive gradients
        grid_dims = [setting[0], setting[1], tilt_series.n_tilts]

        tilt_series.grid_movement_x = tilt_series.grid_movement_x.resize(
            new_size=grid_dims
        ).to(device)
        leaf_variable_x = tilt_series.grid_movement_x.values.requires_grad_(True)
        tilt_series.grid_movement_x = CubicGrid(grid_dims, leaf_variable_x)

        tilt_series.grid_movement_y = tilt_series.grid_movement_y.resize(
            new_size=grid_dims
        ).to(device)
        leaf_variable_y = tilt_series.grid_movement_y.values.requires_grad_(True)
        tilt_series.grid_movement_y = CubicGrid(grid_dims, leaf_variable_y)

        parameters = [leaf_variable_x, leaf_variable_y]
    elif len(setting) == 4:  # TODO add case of starting from existent grid
        tilt_series.grid_volume_warp_x = tilt_series.grid_volume_warp_x.resize(
            new_size=setting
        ).to(device)
        leaf_variable_x = tilt_series.grid_volume_warp_x.values.requires_grad_(True)
        tilt_series.grid_volume_warp_x = CubicGrid(setting, leaf_variable_x)

        tilt_series.grid_volume_warp_y = tilt_series.grid_volume_warp_y.resize(
            new_size=setting
        ).to(device)
        leaf_variable_y = tilt_series.grid_volume_warp_y.values.requires_grad_(True)
        tilt_series.grid_volume_warp_y = CubicGrid(setting, leaf_variable_y)

        tilt_series.grid_volume_warp_z = tilt_series.grid_volume_warp_z.resize(
            new_size=setting
        ).to(device)
        leaf_variable_z = tilt_series.grid_volume_warp_z.values.requires_grad_(True)
        tilt_series.grid_volume_warp_z = CubicGrid(setting, leaf_variable_z)

        parameters = [
            leaf_variable_x,
            leaf_variable_y,
            leaf_variable_z,
        ]
    else:
        raise ValueError(f"Invalid setting for alignment optimization: {setting}")

    alignment_optimizer = torch.optim.LBFGS(
        parameters,
        line_search_fn="strong_wolfe",
    )

    # Initialize list to store loss values
    loss_values = []

    # Determine device type for autocast
    device_type = "cuda" if str(device).startswith("cuda") else "cpu"

    def closure():
        alignment_optimizer.zero_grad()

        # Check for NaN in parameters before computing loss
        # If found, return large penalty to make line search reject this step
        nan_in_params = False
        if setting == "global":
            if torch.isnan(shifts_x).any() or torch.isnan(shifts_y).any():
                nan_in_params = True
        elif len(setting) == 2:
            if torch.isnan(leaf_variable_x).any() or torch.isnan(leaf_variable_y).any():
                nan_in_params = True
        elif len(setting) == 4:
            if (
                torch.isnan(leaf_variable_x).any()
                or torch.isnan(leaf_variable_y).any()
                or torch.isnan(leaf_variable_z).any()
            ):
                nan_in_params = True

        if nan_in_params:
            raise AlignmentNanError

        # update the alignments
        if setting == "global":
            tilt_series.tilt_axis_offset_y = initial_tilt_axis_offset_y + shifts_y
            tilt_series.tilt_axis_offset_x = initial_tilt_axis_offset_x + shifts_x

        batches = int(math.ceil(positions.shape[0] / batch_size))
        total_samples = positions.shape[0]
        total_weighted_score = 0.0
        total_precision = 0.0

        # Disable autocast to ensure full precision during optimization
        with torch.amp.autocast(device_type=device_type, enabled=False):
            # Use gradient accumulation: process each batch separately
            for b in range(batches):
                if b == batches - 1:
                    batch_positions = positions[b * batch_size :]
                else:
                    batch_positions = positions[b * batch_size : (b + 1) * batch_size]

                current_batch_size = batch_positions.shape[0]

                # reconstruct subvolumes for this batch
                subvolumes = tilt_series.reconstruct_subvolumes_single(
                    tilt_data=images,
                    coords=batch_positions.to(device),
                    pixel_size=pixel_size,
                    size=patch_size,
                    apply_ctf=apply_ctf,
                    oversampling=2.0,
                )

                # ensure normalization per subvolume
                mean = einops.reduce(subvolumes, "n d h w -> n 1 1 1", reduction="mean")
                std = torch.std(subvolumes, dim=(-3, -2, -1), keepdim=True)
                # Add epsilon to prevent division by zero (which causes NaN precision)
                eps = 1e-8
                subvolumes = (subvolumes - mean) / (std + eps)

                # change channel to batch dimension
                subvolumes = einops.rearrange(subvolumes, "b d h w -> b 1 d h w")

                # Get score and precision for this batch
                batch_scores, batch_log_precisions = model(subvolumes)

                batch_precisions = batch_log_precisions.exp()

                # Precision-weighted average score for this batch
                batch_weighted_score = (batch_scores * batch_precisions).sum()
                batch_precision_sum = batch_precisions.sum()

                # Weight by batch size for proper gradient accumulation
                weighted_loss = batch_weighted_score * (
                    current_batch_size / total_samples
                )

                # Backward pass for this batch (gradients accumulate)
                weighted_loss.backward()

                # Accumulate for precision-weighted average
                total_weighted_score += batch_weighted_score.item()
                total_precision += batch_precision_sum.item()

        # Precision-weighted average score
        if total_precision <= 0:
            raise ValueError(
                f"Total precision is {total_precision}, which is <= 0. "
                "This indicates a problem with the model precision outputs."
            )
        avg_score = total_weighted_score / total_precision

        # Check if loss is NaN and raise error
        if math.isnan(avg_score):
            raise AlignmentNanError("Loss value is NaN")

        loss_values.append(avg_score)

        return avg_score

    n_iters = 1  # 5 iterations should give convergence
    for x in range(n_iters):
        alignment_optimizer.step(closure)

    if setting == "global":
        # remove gradients and finalize global shifts
        tilt_series.tilt_axis_offset_y = initial_tilt_axis_offset_y + shifts_y.detach()
        tilt_series.tilt_axis_offset_x = initial_tilt_axis_offset_x + shifts_x.detach()
    elif len(setting) == 2:
        # remove gradients
        tilt_series.grid_movement_x.values = tilt_series.grid_movement_x.values.detach()
        tilt_series.grid_movement_y.values = tilt_series.grid_movement_y.values.detach()
    elif len(setting) == 4:
        # remove gradients
        tilt_series.grid_volume_warp_x.values = (
            tilt_series.grid_volume_warp_x.values.detach()
        )
        tilt_series.grid_volume_warp_y.values = (
            tilt_series.grid_volume_warp_y.values.detach()
        )
        tilt_series.grid_volume_warp_z.values = (
            tilt_series.grid_volume_warp_z.values.detach()
        )
    # move back because there were modified in-place
    tilt_series.to("cpu")
    model.to("cpu")

    # Restore original precision setting
    torch.set_float32_matmul_precision(original_precision)

    return tilt_series, loss_values

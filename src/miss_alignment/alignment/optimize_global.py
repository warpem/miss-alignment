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


def optimize_shifts(
    model: MissAlignment,
    tilt_series: TiltSeries,
    images: torch.Tensor,
    pixel_size: float,
    positions: torch.Tensor,
    setting: str | tuple[int, int, int] | tuple[int, int, int, int] = "global",
    patch_size: int = 96,
    batch_size: int = 16,
    apply_ctf: bool = True,
    device: str | torch.device = "cpu",
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
        - tuple(int, int, int) e.g. (3, 3, 41): a single 2D field per image
        - tuple(int, int, int, int) e.g. (3, 3, 2, 10): a volume warp grid
    patch_size : int
        Size of reconstruction patches.
    batch_size : int
        Batch size for reconstruction.
    apply_ctf : bool
        Whether to apply CTF correction.
    device : str | torch.device
        Device to run optimization on.

    Returns
    -------
    tuple[TiltSeries, list[float]]
        Optimized tilt series and list of loss values.
    """
    # Use highest precision for optimization to avoid NaN issues
    # Training uses "medium" and 16-mixed, but optimization needs full precision
    original_precision = torch.get_float32_matmul_precision()
    torch.set_float32_matmul_precision("highest")

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
    elif len(setting) == 3:  # TODO add case of starting from existent grid
        # movement grids - these should receive gradients
        tilt_series.grid_movement_x = tilt_series.grid_movement_x.resize(
            new_size=setting
        ).to(device)
        leaf_variable_x = tilt_series.grid_movement_x.values.requires_grad_(True)
        tilt_series.grid_movement_x = CubicGrid(setting, leaf_variable_x)

        tilt_series.grid_movement_y = tilt_series.grid_movement_y.resize(
            new_size=setting
        ).to(device)
        leaf_variable_y = tilt_series.grid_movement_y.values.requires_grad_(True)
        tilt_series.grid_movement_y = CubicGrid(setting, leaf_variable_y)

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
        elif len(setting) == 3:
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
            # Return large penalty to reject this step in line search
            penalty = torch.tensor(1e10, dtype=torch.float32, device=device)
            return penalty

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

        # Check if loss is NaN and return penalty if so
        if math.isnan(avg_score):
            penalty = torch.tensor(1e10, dtype=torch.float32, device=device)
            return penalty

        loss_values.append(avg_score)

        return avg_score

    n_iters = 1  # 5 iterations should give convergence
    for x in range(n_iters):
        alignment_optimizer.step(closure)

    if setting == "global":
        # remove gradients and finalize global shifts
        tilt_series.tilt_axis_offset_y = initial_tilt_axis_offset_y + shifts_y.detach()
        tilt_series.tilt_axis_offset_x = initial_tilt_axis_offset_x + shifts_x.detach()
    elif len(setting) == 3:
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

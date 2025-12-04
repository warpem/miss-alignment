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

        # update the alignments
        if setting == "global":
            tilt_series.tilt_axis_offset_y = initial_tilt_axis_offset_y + shifts_y
            tilt_series.tilt_axis_offset_x = initial_tilt_axis_offset_x + shifts_x

            # NaN check 1: Check shift parameters
            if torch.isnan(shifts_x).any() or torch.isnan(shifts_y).any():
                print("[NaN Check 1] NaN detected in shifts after update")
                print(f"  shifts_x has NaN: {torch.isnan(shifts_x).any()}")
                print(f"  shifts_y has NaN: {torch.isnan(shifts_y).any()}")
                print(
                    f"  shifts_x range: [{shifts_x.min().item():.2f}, "
                    f"{shifts_x.max().item():.2f}]"
                )
                print(
                    f"  shifts_y range: [{shifts_y.min().item():.2f}, "
                    f"{shifts_y.max().item():.2f}]"
                )

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

                # NaN check 2: Check reconstructions
                if torch.isnan(subvolumes).any():
                    print(f"[NaN Check 2] NaN detected in reconstructions (batch {b})")
                    print(
                        f"  Number of NaN values: "
                        f"{torch.isnan(subvolumes).sum().item()}"
                    )

                # ensure normalization per subvolume
                mean = einops.reduce(subvolumes, "n d h w -> n 1 1 1", reduction="mean")
                std = torch.std(subvolumes, dim=(-3, -2, -1), keepdim=True)
                # Add epsilon to prevent division by zero (which causes NaN precision)
                eps = 1e-8
                subvolumes = (subvolumes - mean) / (std + eps)

                # NaN check 3: Check after normalization
                if torch.isnan(subvolumes).any():
                    print(f"[NaN Check 3] NaN detected after normalization (batch {b})")
                    print(f"  mean has NaN: {torch.isnan(mean).any()}")
                    print(f"  std has NaN: {torch.isnan(std).any()}")

                # change channel to batch dimension
                subvolumes = einops.rearrange(subvolumes, "b d h w -> b 1 d h w")

                # Get score and precision for this batch
                batch_scores, batch_log_precisions = model(subvolumes)

                # NaN check 4: Check model outputs
                if (
                    torch.isnan(batch_scores).any()
                    or torch.isnan(batch_log_precisions).any()
                ):
                    print(f"[NaN Check 4] NaN detected in model outputs (batch {b})")
                    print(f"  batch_scores has NaN: {torch.isnan(batch_scores).any()}")
                    print(
                        f"  batch_log_precisions "
                        f"has NaN: {torch.isnan(batch_log_precisions).any()}"
                    )

                batch_precisions = batch_log_precisions.exp()

                # NaN check 5: Check after exp
                if (
                    torch.isnan(batch_precisions).any()
                    or torch.isinf(batch_precisions).any()
                ):
                    print(
                        f"[NaN Check 5] NaN/Inf detected in "
                        f"batch_precisions (batch {b})"
                    )
                    print(
                        f"  batch_precisions has NaN:"
                        f" {torch.isnan(batch_precisions).any()}"
                    )
                    print(
                        f"  batch_precisions has Inf:"
                        f" {torch.isinf(batch_precisions).any()}"
                    )
                    print(
                        f"  batch_log_precisions range:"
                        f" [{batch_log_precisions.min().item():.2f}, "
                        f"{batch_log_precisions.max().item():.2f}]"
                    )

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

        # NaN check 6: Check accumulated values
        if math.isnan(total_weighted_score) or math.isnan(total_precision):
            print("[NaN Check 6] NaN detected in accumulated values")
            print(f"  total_weighted_score: {total_weighted_score}")
            print(f"  total_precision: {total_precision}")

        # NaN check 7: Check gradients before optimizer uses them
        if setting == "global":
            if shifts_x.grad is not None and shifts_y.grad is not None:
                grad_x_has_nan = torch.isnan(shifts_x.grad).any()
                grad_y_has_nan = torch.isnan(shifts_y.grad).any()
                grad_x_has_inf = torch.isinf(shifts_x.grad).any()
                grad_y_has_inf = torch.isinf(shifts_y.grad).any()

                if grad_x_has_nan or grad_y_has_nan or grad_x_has_inf or grad_y_has_inf:
                    print("[NaN Check 7] NaN/Inf detected in gradients")
                    print(f"  shifts_x.grad has NaN: {grad_x_has_nan}")
                    print(f"  shifts_y.grad has NaN: {grad_y_has_nan}")
                    print(f"  shifts_x.grad has Inf: {grad_x_has_inf}")
                    print(f"  shifts_y.grad has Inf: {grad_y_has_inf}")

                grad_x_max = shifts_x.grad.abs().max().item()
                grad_y_max = shifts_y.grad.abs().max().item()
                print(
                    f"[Gradient Check] Max gradient magnitudes: "
                    f"shifts_x={grad_x_max:.6f}, shifts_y={grad_y_max:.6f}"
                )

        # Precision-weighted average score
        avg_score = (
            total_weighted_score / total_precision if total_precision > 0 else 0.0
        )
        if avg_score == 0.0:
            print(
                f"[Zero Score] Average score is exactly 0.0, "
                f"precision={total_precision}"
            )
        print(
            f"[Score] total_weighted={total_weighted_score}, "
            f"precision={total_precision}, avg={avg_score}"
        )
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

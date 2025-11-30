"""Spline-based optimization for tilt series alignment.

This module provides smooth spline parameterization for capturing systematic
drift while avoiding local minima, followed by per-tilt fine adjustment.
"""

import math

import einops
import torch
from warpylib import TiltSeries

from miss_alignment.models import MissAlignment

from .optimize_global import optimize_shifts


def evaluate_catmull_rom_spline(
    control_values: torch.Tensor,
    control_positions: torch.Tensor,
    query_positions: torch.Tensor,
) -> torch.Tensor:
    """Evaluate Catmull-Rom spline at query positions.

    Parameters
    ----------
    control_values : torch.Tensor
        Values at control points, shape (n_control_points,).
    control_positions : torch.Tensor
        Positions of control points (e.g., tilt angles), shape (n_control_points,).
        Must be sorted in ascending order.
    query_positions : torch.Tensor
        Positions to evaluate the spline at, shape (n_queries,).

    Returns
    -------
    torch.Tensor
        Interpolated values at query positions, shape (n_queries,).
    """
    n_cp = control_values.shape[0]

    # Normalize query positions to [0, n_cp-1] index space
    pos_min = control_positions[0]
    pos_max = control_positions[-1]
    normalized = (query_positions - pos_min) / (pos_max - pos_min) * (n_cp - 1)

    # Find which segment each query falls in
    segment_idx = torch.clamp(normalized.floor().long(), 0, n_cp - 2)
    t = normalized - segment_idx.float()  # local parameter in [0, 1]

    # Get the 4 control points for each query (with boundary clamping)
    idx_m1 = torch.clamp(segment_idx - 1, 0, n_cp - 1)
    idx_0 = segment_idx
    idx_1 = torch.clamp(segment_idx + 1, 0, n_cp - 1)
    idx_2 = torch.clamp(segment_idx + 2, 0, n_cp - 1)

    p0 = control_values[idx_m1]
    p1 = control_values[idx_0]
    p2 = control_values[idx_1]
    p3 = control_values[idx_2]

    # Catmull-Rom interpolation formula
    t2 = t * t
    t3 = t2 * t
    result = 0.5 * (
        (2 * p1)
        + (-p0 + p2) * t
        + (2 * p0 - 5 * p1 + 4 * p2 - p3) * t2
        + (-p0 + 3 * p1 - 3 * p2 + p3) * t3
    )

    return result


def optimize_shifts_spline(
    model: MissAlignment,
    tilt_series: TiltSeries,
    images: torch.Tensor,
    pixel_size: float,
    positions: torch.Tensor,
    patch_size: int = 96,
    batch_size: int = 16,
    apply_ctf: bool = True,
    device: str | torch.device = "cpu",
    n_control_points: int = 7,
):
    """Optimize shifts using a smooth spline parameterization.

    Instead of optimizing per-tilt shifts independently, this optimizes
    a smooth Catmull-Rom spline with uniformly spaced control points.
    This captures systematic drift while avoiding local minima.

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
    patch_size : int
        Size of reconstruction patches.
    batch_size : int
        Batch size for reconstruction.
    apply_ctf : bool
        Whether to apply CTF correction.
    device : str | torch.device
        Device to run optimization on.
    n_control_points : int
        Number of spline control points. More points = more flexibility.
        Default 7 gives smooth corrections.

    Returns
    -------
    tuple[TiltSeries, list[float]]
        Optimized tilt series and loss values.
    """
    # Move to device
    tilt_series.to(device)
    model.to(device)
    model.freeze()
    model.eval()
    images = images.to(device)

    # Get tilt angles and sort order
    angles = tilt_series.angles
    sorted_indices = tilt_series.indices_sorted_angle()
    sorted_angles = angles[sorted_indices]

    # Store initial offsets
    initial_offset_x = tilt_series.tilt_axis_offset_x.clone()
    initial_offset_y = tilt_series.tilt_axis_offset_y.clone()

    # Create uniformly spaced control point positions
    control_positions = torch.linspace(
        sorted_angles[0].item(),
        sorted_angles[-1].item(),
        n_control_points,
        device=device,
    )

    # Initialize control point deltas to zero (will be optimized)
    control_deltas_x = torch.zeros(n_control_points, device=device, requires_grad=True)
    control_deltas_y = torch.zeros(n_control_points, device=device, requires_grad=True)

    parameters = [control_deltas_x, control_deltas_y]

    alignment_optimizer = torch.optim.LBFGS(
        parameters,
        line_search_fn="strong_wolfe",
    )

    loss_values = []

    def closure():
        alignment_optimizer.zero_grad()

        # Evaluate spline at each tilt angle to get shift deltas
        shift_deltas_x = evaluate_catmull_rom_spline(
            control_deltas_x, control_positions, angles.to(device)
        )
        shift_deltas_y = evaluate_catmull_rom_spline(
            control_deltas_y, control_positions, angles.to(device)
        )

        # Apply deltas to initial offsets
        tilt_series.tilt_axis_offset_x = initial_offset_x + shift_deltas_x
        tilt_series.tilt_axis_offset_y = initial_offset_y + shift_deltas_y

        batches = int(math.ceil(positions.shape[0] / batch_size))
        total_samples = positions.shape[0]
        total_loss = 0.0

        for b in range(batches):
            if b == batches - 1:
                batch_positions = positions[b * batch_size :]
            else:
                batch_positions = positions[b * batch_size : (b + 1) * batch_size]

            current_batch_size = batch_positions.shape[0]

            subvolumes = tilt_series.reconstruct_subvolumes_single(
                tilt_data=images,
                coords=batch_positions.to(device),
                pixel_size=pixel_size,
                size=patch_size,
                apply_ctf=apply_ctf,
                oversampling=2.0,
            )

            mean = einops.reduce(subvolumes, "n d h w -> n 1 1 1", reduction="mean")
            std = torch.std(subvolumes, dim=(-3, -2, -1), keepdim=True)
            subvolumes = (subvolumes - mean) / std
            subvolumes = einops.rearrange(subvolumes, "b d h w -> b 1 d h w")

            batch_loss = model(subvolumes)
            batch_loss = batch_loss.mean()

            weighted_loss = batch_loss * (current_batch_size / total_samples)
            weighted_loss.backward()

            total_loss += batch_loss.item() * current_batch_size

        avg_loss = total_loss / total_samples
        loss_values.append(avg_loss)

        return avg_loss

    n_iters = 1
    for _ in range(n_iters):
        alignment_optimizer.step(closure)

    # Finalize: compute final shifts and detach
    with torch.no_grad():
        shift_deltas_x = evaluate_catmull_rom_spline(
            control_deltas_x, control_positions, angles.to(device)
        )
        shift_deltas_y = evaluate_catmull_rom_spline(
            control_deltas_y, control_positions, angles.to(device)
        )
        tilt_series.tilt_axis_offset_x = initial_offset_x + shift_deltas_x
        tilt_series.tilt_axis_offset_y = initial_offset_y + shift_deltas_y

    tilt_series.to("cpu")
    model.to("cpu")

    return tilt_series, loss_values


def optimize_shifts_coarse_to_fine(
    model: MissAlignment,
    tilt_series: TiltSeries,
    images: torch.Tensor,
    pixel_size: float,
    positions: torch.Tensor,
    patch_size: int = 96,
    batch_size: int = 16,
    apply_ctf: bool = True,
    device: str | torch.device = "cpu",
    n_control_points: int = 7,
):
    """Two-phase optimization: smooth spline followed by per-tilt fine adjustment.

    Phase 1: Optimize a smooth spline to capture systematic drift
    Phase 2: Optimize individual per-tilt shifts for fine jitter correction

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
    patch_size : int
        Size of reconstruction patches.
    batch_size : int
        Batch size for reconstruction.
    apply_ctf : bool
        Whether to apply CTF correction.
    device : str | torch.device
        Device to run optimization on.
    n_control_points : int
        Number of spline control points for coarse phase.

    Returns
    -------
    tuple[TiltSeries, list[float]]
        Optimized tilt series and loss values from both phases.
    """
    all_loss_values = []

    # Phase 1: Smooth spline optimization
    print("Phase 1: Spline optimization (coarse)")
    tilt_series, loss_values = optimize_shifts_spline(
        model=model,
        tilt_series=tilt_series,
        images=images,
        pixel_size=pixel_size,
        positions=positions,
        patch_size=patch_size,
        batch_size=batch_size,
        apply_ctf=apply_ctf,
        device=device,
        n_control_points=n_control_points,
    )
    all_loss_values.extend(loss_values)
    print(f"  Spline loss: {loss_values[0]:.4f} -> {loss_values[-1]:.4f}")

    # Phase 2: Per-tilt fine optimization
    print("Phase 2: Per-tilt optimization (fine)")
    tilt_series, loss_values = optimize_shifts(
        model=model,
        tilt_series=tilt_series,
        images=images,
        pixel_size=pixel_size,
        positions=positions,
        setting="global",
        patch_size=patch_size,
        batch_size=batch_size,
        apply_ctf=apply_ctf,
        device=device,
    )
    all_loss_values.extend(loss_values)
    print(f"  Per-tilt loss: {loss_values[0]:.4f} -> {loss_values[-1]:.4f}")

    return tilt_series, all_loss_values

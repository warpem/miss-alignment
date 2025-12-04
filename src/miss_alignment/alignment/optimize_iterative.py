"""Iterative anchoring optimization for tilt series alignment.

This module addresses high-tilt images getting stuck in local minima by
progressively expanding the set of reliable tilts while preserving
relative offsets of unreliable tilts.
"""

import math
from typing import Callable

import torch
from warpylib import TiltSeries

from miss_alignment.models import MissAlignment

from .optimize_global import optimize_shifts


def run_iterative_anchoring(
    tilt_series: TiltSeries,
    optimize_fn: Callable[[TiltSeries], tuple[TiltSeries, list[float]]],
    initial_reliable_fraction: float = 1 / 2,
) -> tuple[TiltSeries, list[float]]:
    """Core iterative anchoring logic for global shift optimization.

    Addresses high-tilt images getting stuck in local minima by:
    1. Treating only the central fraction of tilts as "reliable"
    2. Storing relative offsets of unreliable (high angle) tilts
    3. Running the optimizer
    4. Restoring relative structure of unreliable tilts while
       propagating the movement of boundary reliable tilts
    5. Gradually expanding the reliable set (1 tilt per side per iteration)

    Parameters
    ----------
    tilt_series : TiltSeries
        Tilt series to optimize.
    optimize_fn : Callable[[TiltSeries], tuple[TiltSeries, list[float]]]
        Optimization function that takes a TiltSeries and returns
        (optimized_tilt_series, loss_values).
    initial_reliable_fraction : float
        Initial fraction of tilts to consider reliable (0 to 1). Default 1/2.

    Returns
    -------
    tuple[TiltSeries, list[float]]
        Optimized tilt series and loss values from all iterations.
    """
    n_tilts = tilt_series.n_tilts
    sorted_indices = tilt_series.indices_sorted_angle()

    # Calculate initial number of unreliable tilts per side
    n_reliable = int(n_tilts * initial_reliable_fraction)
    n_reliable = max(1, min(n_reliable, n_tilts))
    n_unreliable_total = n_tilts - n_reliable
    n_unreliable_per_side = n_unreliable_total // 2

    all_loss_values = []

    # Track best solution - initialize with current state
    best_loss = float("inf")
    best_offsets_x = tilt_series.tilt_axis_offset_x.clone()
    best_offsets_y = tilt_series.tilt_axis_offset_y.clone()

    while n_unreliable_per_side > 0:
        print(f"n_unreliable_per_side: {n_unreliable_per_side}")
        # Define boundary in sorted index space
        # sorted_indices[0] is most negative angle, sorted_indices[-1] most positive
        pos_boundary_sorted_idx = n_tilts - 1 - n_unreliable_per_side

        # Store relative offsets for unreliable tilts (chain-like, toward center)
        # Negative side: sorted indices 0 to neg_boundary_sorted_idx-1
        neg_relative_offsets_x = []
        neg_relative_offsets_y = []
        for i in range(n_unreliable_per_side):
            curr_tilt_idx = sorted_indices[i]
            next_tilt_idx = sorted_indices[i + 1]  # neighbor toward center
            rel_x = (
                tilt_series.tilt_axis_offset_x[curr_tilt_idx]
                - tilt_series.tilt_axis_offset_x[next_tilt_idx]
            ).clone()
            rel_y = (
                tilt_series.tilt_axis_offset_y[curr_tilt_idx]
                - tilt_series.tilt_axis_offset_y[next_tilt_idx]
            ).clone()
            neg_relative_offsets_x.append(rel_x)
            neg_relative_offsets_y.append(rel_y)

        # Positive side: sorted indices pos_boundary_sorted_idx+1 to n_tilts-1
        pos_relative_offsets_x = []
        pos_relative_offsets_y = []
        for i in range(pos_boundary_sorted_idx + 1, n_tilts):
            curr_tilt_idx = sorted_indices[i]
            prev_tilt_idx = sorted_indices[i - 1]  # neighbor toward center
            rel_x = (
                tilt_series.tilt_axis_offset_x[curr_tilt_idx]
                - tilt_series.tilt_axis_offset_x[prev_tilt_idx]
            ).clone()
            rel_y = (
                tilt_series.tilt_axis_offset_y[curr_tilt_idx]
                - tilt_series.tilt_axis_offset_y[prev_tilt_idx]
            ).clone()
            pos_relative_offsets_x.append(rel_x)
            pos_relative_offsets_y.append(rel_y)

        # Run optimization
        tilt_series, loss_values = optimize_fn(tilt_series)
        all_loss_values.extend(loss_values)
        current_loss = loss_values[-1]
        print(f"Loss went from {loss_values[0]} to {current_loss}")

        # Check for NaN in tilt_axis_offsets
        has_nan_x = torch.isnan(tilt_series.tilt_axis_offset_x).any()
        has_nan_y = torch.isnan(tilt_series.tilt_axis_offset_y).any()
        if has_nan_x or has_nan_y:
            print(
                f"  -> NaN detected in tilt_axis_offsets "
                f"(x: {has_nan_x}, y: {has_nan_y}), loss: {current_loss}"
            )

        # Check for NaN or worse loss - immediately revert if so
        if math.isnan(current_loss) or current_loss >= best_loss:
            if math.isnan(current_loss):
                print(f"  -> NaN detected, reverting to best (loss={best_loss})")
            else:
                print(
                    f"  -> Loss worse ({current_loss:.4f} >= {best_loss:.4f}), "
                    "reverting"
                )
            # Restore best solution and skip anchoring restoration
            tilt_series.tilt_axis_offset_x = best_offsets_x.clone()
            tilt_series.tilt_axis_offset_y = best_offsets_y.clone()
        else:
            # New best - update and apply anchoring restoration
            best_loss = current_loss
            best_offsets_x = tilt_series.tilt_axis_offset_x.clone()
            best_offsets_y = tilt_series.tilt_axis_offset_y.clone()
            print(f"  -> New best loss: {best_loss:.4f}")

            # Restore unreliable tilts with chain-like relative offsets
            # Negative side: propagate from boundary outward (high sorted idx to low)
            for i in range(n_unreliable_per_side - 1, -1, -1):
                curr_tilt_idx = sorted_indices[i]
                next_tilt_idx = sorted_indices[i + 1]
                tilt_series.tilt_axis_offset_x[curr_tilt_idx] = (
                    tilt_series.tilt_axis_offset_x[next_tilt_idx]
                    + neg_relative_offsets_x[i]
                )
                tilt_series.tilt_axis_offset_y[curr_tilt_idx] = (
                    tilt_series.tilt_axis_offset_y[next_tilt_idx]
                    + neg_relative_offsets_y[i]
                )

            # Positive side: propagate from boundary outward (low sorted idx to high)
            for i, sorted_i in enumerate(range(pos_boundary_sorted_idx + 1, n_tilts)):
                curr_tilt_idx = sorted_indices[sorted_i]
                prev_tilt_idx = sorted_indices[sorted_i - 1]
                tilt_series.tilt_axis_offset_x[curr_tilt_idx] = (
                    tilt_series.tilt_axis_offset_x[prev_tilt_idx]
                    + pos_relative_offsets_x[i]
                )
                tilt_series.tilt_axis_offset_y[curr_tilt_idx] = (
                    tilt_series.tilt_axis_offset_y[prev_tilt_idx]
                    + pos_relative_offsets_y[i]
                )

        # Expand reliable set by 1 tilt per side
        n_unreliable_per_side -= 1

    # Final optimization with all tilts reliable
    tilt_series, loss_values = optimize_fn(tilt_series)
    all_loss_values.extend(loss_values)
    final_loss = loss_values[-1]
    print(f"Last one: loss went from {loss_values[0]} to {final_loss}")

    # Check for NaN in tilt_axis_offsets
    has_nan_x = torch.isnan(tilt_series.tilt_axis_offset_x).any()
    has_nan_y = torch.isnan(tilt_series.tilt_axis_offset_y).any()
    if has_nan_x or has_nan_y:
        print(
            f"  -> NaN detected in tilt_axis_offsets "
            f"(x: {has_nan_x}, y: {has_nan_y}), loss: {final_loss}"
        )

    # Check for NaN or worse than best
    if math.isnan(final_loss) or final_loss >= best_loss:
        if math.isnan(final_loss):
            print(f"Final produced NaN, restoring best (loss={best_loss:.4f})")
        else:
            print(f"Final worse ({final_loss:.4f} >= {best_loss:.4f}), restoring best")
        tilt_series.tilt_axis_offset_x = best_offsets_x
        tilt_series.tilt_axis_offset_y = best_offsets_y
    else:
        print(f"Final optimization achieved best loss: {final_loss:.4f}")

    return tilt_series, all_loss_values


def optimize_shifts_iterative(
    model: MissAlignment,
    tilt_series: TiltSeries,
    images: torch.Tensor,
    pixel_size: float,
    positions: torch.Tensor,
    patch_size: int = 96,
    batch_size: int = 16,
    apply_ctf: bool = True,
    device: str | torch.device = "cpu",
    initial_reliable_fraction: float = 1 / 2,
) -> tuple[TiltSeries, list[float]]:
    """Iterative anchored optimization for global shifts.

    Convenience wrapper around run_iterative_anchoring that uses
    optimize_shifts as the optimizer function.

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
    initial_reliable_fraction : float
        Initial fraction of tilts to consider reliable (0 to 1). Default 1/2.

    Returns
    -------
    tuple[TiltSeries, list[float]]
        Optimized tilt series and loss values from all iterations.
    """

    def optimize_fn(ts: TiltSeries) -> tuple[TiltSeries, list[float]]:
        return optimize_shifts(
            model=model,
            tilt_series=ts,
            images=images,
            pixel_size=pixel_size,
            positions=positions,
            setting="global",
            patch_size=patch_size,
            batch_size=batch_size,
            apply_ctf=apply_ctf,
            device=device,
        )

    return run_iterative_anchoring(
        tilt_series=tilt_series,
        optimize_fn=optimize_fn,
        initial_reliable_fraction=initial_reliable_fraction,
    )

"""Visualization tools for tracking alignment optimization progress.

This module provides utilities to capture and visualize intermediate states
during L-BFGS alignment optimization, including subvolumes, precision values,
and loss trajectories at each optimization step.
"""

from dataclasses import dataclass, field
from pathlib import Path

import einops
import torch
from warpylib import TiltSeries

from miss_alignment.models import MissAlignment


@dataclass
class OptimizationStepData:
    """Data captured at a single optimization step.

    Attributes
    ----------
    step : int
        Step number (L-BFGS closure call count).
    loss : float
        Precision-weighted average loss at this step.
    mean_precision : float
        Mean precision across all subvolumes.
    total_precision : float
        Sum of precisions across all subvolumes.
    subvolumes : torch.Tensor | None
        Subset of reconstruction patches, shape (n_samples, d, h, w).
        None if not captured this step.
    precisions : torch.Tensor | None
        Precision values for subvolumes, shape (n_samples,).
        None if not captured this step.
    scores : torch.Tensor | None
        Score values for subvolumes, shape (n_samples,).
        None if not captured this step.
    shifts_x : torch.Tensor | None
        Current X shifts (in Angstroms), shape (n_tilts,).
        For global alignment only.
    shifts_y : torch.Tensor | None
        Current Y shifts (in Angstroms), shape (n_tilts,).
        For global alignment only.
    positions : torch.Tensor | None
        3D positions of captured subvolumes, shape (n_samples, 3).
    alignment_setting : str | tuple | None
        Alignment setting used (e.g., "global", (3,3), (3,3,2,10)).
    grid_movement_x : torch.Tensor | None
        2D warping grid X values. For 2D warping modes only.
    grid_movement_y : torch.Tensor | None
        2D warping grid Y values. For 2D warping modes only.
    grid_volume_warp_x : torch.Tensor | None
        3D volume warp grid X values. For 3D warping modes only.
    grid_volume_warp_y : torch.Tensor | None
        3D volume warp grid Y values. For 3D warping modes only.
    grid_volume_warp_z : torch.Tensor | None
        3D volume warp grid Z values. For 3D warping modes only.
    """

    step: int
    loss: float
    mean_precision: float
    total_precision: float
    subvolumes: torch.Tensor | None = None
    precisions: torch.Tensor | None = None
    scores: torch.Tensor | None = None
    shifts_x: torch.Tensor | None = None
    shifts_y: torch.Tensor | None = None
    positions: torch.Tensor | None = None
    alignment_setting: str | tuple | None = None
    grid_movement_x: torch.Tensor | None = None
    grid_movement_y: torch.Tensor | None = None
    grid_volume_warp_x: torch.Tensor | None = None
    grid_volume_warp_y: torch.Tensor | None = None
    grid_volume_warp_z: torch.Tensor | None = None


@dataclass
class OptimizationTracker:
    """Tracks optimization progress and captures intermediate states.

    Parameters
    ----------
    output_dir : Path
        Directory to save captured data.
    capture_frequency : int
        Capture detailed data every N steps. Default: 1 (every step).
    max_subvolumes_per_step : int
        Maximum number of subvolumes to store per step (for memory efficiency).
        Default: 32.
    save_subvolumes : bool
        Whether to save subvolumes. Set to False to only track losses/precisions.
        Default: True.
    alignment_setting : str | tuple | None
        Alignment setting being used (e.g., "global", (3,3), (3,3,2,10)).
    """

    output_dir: Path
    capture_frequency: int = 1
    max_subvolumes_per_step: int = 32
    save_subvolumes: bool = True
    alignment_setting: str | tuple | None = None

    # Internal state
    step_data: list[OptimizationStepData] = field(default_factory=list)
    current_step: int = 0

    def __post_init__(self):
        """Create output directory if it doesn't exist."""
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def should_capture_detailed(self) -> bool:
        """Check if detailed data should be captured at current step."""
        return self.current_step % self.capture_frequency == 0

    def on_closure_call(
        self,
        loss: float,
        total_precision: float,
        mean_precision: float,
        subvolumes: torch.Tensor | None = None,
        precisions: torch.Tensor | None = None,
        scores: torch.Tensor | None = None,
        shifts_x: torch.Tensor | None = None,
        shifts_y: torch.Tensor | None = None,
        positions: torch.Tensor | None = None,
        grid_movement_x: torch.Tensor | None = None,
        grid_movement_y: torch.Tensor | None = None,
        grid_volume_warp_x: torch.Tensor | None = None,
        grid_volume_warp_y: torch.Tensor | None = None,
        grid_volume_warp_z: torch.Tensor | None = None,
    ):
        """Callback invoked at each closure evaluation.

        Parameters
        ----------
        loss : float
            Current loss value.
        total_precision : float
            Sum of precisions.
        mean_precision : float
            Mean precision.
        subvolumes : torch.Tensor | None
            Reconstructed subvolumes, shape (n, d, h, w).
        precisions : torch.Tensor | None
            Precision values, shape (n,).
        scores : torch.Tensor | None
            Score values, shape (n,).
        shifts_x : torch.Tensor | None
            Current X shifts (global alignment only).
        shifts_y : torch.Tensor | None
            Current Y shifts (global alignment only).
        positions : torch.Tensor | None
            3D positions of subvolumes, shape (n, 3).
        grid_movement_x : torch.Tensor | None
            2D warping grid X values (2D warping only).
        grid_movement_y : torch.Tensor | None
            2D warping grid Y values (2D warping only).
        grid_volume_warp_x : torch.Tensor | None
            3D volume warp grid X values (3D warping only).
        grid_volume_warp_y : torch.Tensor | None
            3D volume warp grid Y values (3D warping only).
        grid_volume_warp_z : torch.Tensor | None
            3D volume warp grid Z values (3D warping only).
        """
        should_save_detailed = self.should_capture_detailed() and self.save_subvolumes

        # Sample a subset of subvolumes if needed
        if should_save_detailed and subvolumes is not None:
            n_samples = min(self.max_subvolumes_per_step, subvolumes.shape[0])
            if n_samples < subvolumes.shape[0]:
                # Sample uniformly across the volume
                indices = torch.linspace(
                    0, subvolumes.shape[0] - 1, n_samples, dtype=torch.long
                )
                subvolumes = subvolumes[indices].cpu()
                if precisions is not None:
                    precisions = precisions[indices].cpu()
                if scores is not None:
                    scores = scores[indices].cpu()
                if positions is not None:
                    positions = positions[indices].cpu()
            else:
                subvolumes = subvolumes.cpu()
                precisions = precisions.cpu() if precisions is not None else None
                scores = scores.cpu() if scores is not None else None
                positions = positions.cpu() if positions is not None else None
        else:
            subvolumes = None
            precisions = None
            scores = None
            positions = None

        # Store alignment parameter information
        shifts_x_cpu = shifts_x.detach().cpu() if shifts_x is not None else None
        shifts_y_cpu = shifts_y.detach().cpu() if shifts_y is not None else None

        # Store grid parameters if present
        grid_movement_x_cpu = (
            grid_movement_x.detach().cpu() if grid_movement_x is not None else None
        )
        grid_movement_y_cpu = (
            grid_movement_y.detach().cpu() if grid_movement_y is not None else None
        )
        grid_volume_warp_x_cpu = (
            grid_volume_warp_x.detach().cpu()
            if grid_volume_warp_x is not None
            else None
        )
        grid_volume_warp_y_cpu = (
            grid_volume_warp_y.detach().cpu()
            if grid_volume_warp_y is not None
            else None
        )
        grid_volume_warp_z_cpu = (
            grid_volume_warp_z.detach().cpu()
            if grid_volume_warp_z is not None
            else None
        )

        # Create step data
        step_data = OptimizationStepData(
            step=self.current_step,
            loss=loss,
            mean_precision=mean_precision,
            total_precision=total_precision,
            subvolumes=subvolumes,
            precisions=precisions,
            scores=scores,
            shifts_x=shifts_x_cpu,
            shifts_y=shifts_y_cpu,
            positions=positions,
            alignment_setting=self.alignment_setting,
            grid_movement_x=grid_movement_x_cpu,
            grid_movement_y=grid_movement_y_cpu,
            grid_volume_warp_x=grid_volume_warp_x_cpu,
            grid_volume_warp_y=grid_volume_warp_y_cpu,
            grid_volume_warp_z=grid_volume_warp_z_cpu,
        )

        self.step_data.append(step_data)

        # Save to disk incrementally
        self._save_step(step_data)

        self.current_step += 1

    def _save_step(self, step_data: OptimizationStepData):
        """Save step data to disk."""
        step_file = self.output_dir / f"step_{step_data.step:04d}.pt"

        save_dict = {
            "step": step_data.step,
            "loss": step_data.loss,
            "mean_precision": step_data.mean_precision,
            "total_precision": step_data.total_precision,
            "alignment_setting": str(step_data.alignment_setting)
            if step_data.alignment_setting is not None
            else None,
        }

        if step_data.subvolumes is not None:
            save_dict["subvolumes"] = step_data.subvolumes
        if step_data.precisions is not None:
            save_dict["precisions"] = step_data.precisions
        if step_data.scores is not None:
            save_dict["scores"] = step_data.scores
        if step_data.shifts_x is not None:
            save_dict["shifts_x"] = step_data.shifts_x
        if step_data.shifts_y is not None:
            save_dict["shifts_y"] = step_data.shifts_y
        if step_data.positions is not None:
            save_dict["positions"] = step_data.positions

        # Save grid parameters if present
        if step_data.grid_movement_x is not None:
            save_dict["grid_movement_x"] = step_data.grid_movement_x
        if step_data.grid_movement_y is not None:
            save_dict["grid_movement_y"] = step_data.grid_movement_y
        if step_data.grid_volume_warp_x is not None:
            save_dict["grid_volume_warp_x"] = step_data.grid_volume_warp_x
        if step_data.grid_volume_warp_y is not None:
            save_dict["grid_volume_warp_y"] = step_data.grid_volume_warp_y
        if step_data.grid_volume_warp_z is not None:
            save_dict["grid_volume_warp_z"] = step_data.grid_volume_warp_z

        torch.save(save_dict, step_file)

    def save_summary(self):
        """Save summary statistics to disk."""
        summary = {
            "total_steps": self.current_step,
            "losses": [s.loss for s in self.step_data],
            "mean_precisions": [s.mean_precision for s in self.step_data],
            "total_precisions": [s.total_precision for s in self.step_data],
        }

        summary_file = self.output_dir / "summary.pt"
        torch.save(summary, summary_file)

        # Also save as text for easy inspection
        summary_text = self.output_dir / "summary.txt"
        with open(summary_text, "w") as f:
            f.write("L-BFGS Optimization Summary\n")
            f.write("=" * 50 + "\n\n")
            f.write(f"Total steps: {self.current_step}\n")
            f.write(f"Initial loss: {summary['losses'][0]:.6f}\n")
            f.write(f"Final loss: {summary['losses'][-1]:.6f}\n")
            f.write(
                f"Loss reduction: {summary['losses'][0] - summary['losses'][-1]:.6f}\n"
            )
            f.write(f"\nInitial mean precision: {summary['mean_precisions'][0]:.6f}\n")
            f.write(f"Final mean precision: {summary['mean_precisions'][-1]:.6f}\n")
            f.write("\nStep-by-step losses:\n")
            for step, loss in enumerate(summary["losses"]):
                f.write(f"  Step {step}: {loss:.6f}\n")


def load_optimization_data(
    output_dir: Path,
    load_subvolumes: bool = True,
) -> list[OptimizationStepData]:
    """Load captured optimization data from disk.

    Parameters
    ----------
    output_dir : Path
        Directory containing saved step data.
    load_subvolumes : bool
        Whether to load subvolume data. Set to False to only load
        losses and precisions for faster loading.

    Returns
    -------
    list[OptimizationStepData]
        List of step data in chronological order.
    """
    step_files = sorted(output_dir.glob("step_*.pt"))

    step_data_list = []
    for step_file in step_files:
        data = torch.load(step_file, map_location="cpu", weights_only=True)

        step_data = OptimizationStepData(
            step=data["step"],
            loss=data["loss"],
            mean_precision=data["mean_precision"],
            total_precision=data["total_precision"],
            subvolumes=data.get("subvolumes") if load_subvolumes else None,
            precisions=data.get("precisions"),
            scores=data.get("scores"),
            shifts_x=data.get("shifts_x"),
            shifts_y=data.get("shifts_y"),
            positions=data.get("positions") if load_subvolumes else None,
            alignment_setting=data.get("alignment_setting"),
            grid_movement_x=data.get("grid_movement_x"),
            grid_movement_y=data.get("grid_movement_y"),
            grid_volume_warp_x=data.get("grid_volume_warp_x"),
            grid_volume_warp_y=data.get("grid_volume_warp_y"),
            grid_volume_warp_z=data.get("grid_volume_warp_z"),
        )
        step_data_list.append(step_data)

    return step_data_list


def optimize_shifts_with_tracking(
    model: MissAlignment,
    tilt_series: TiltSeries,
    images: torch.Tensor,
    pixel_size: float,
    positions: torch.Tensor,
    tracker: OptimizationTracker,
    setting: str | tuple[int, int] | tuple[int, int, int, int] = "global",
    patch_size: int = 96,
    batch_size: int = 16,
    apply_ctf: bool = True,
    device: str | torch.device = "cpu",
):
    """Run alignment optimization with step-by-step tracking.

    This is a modified version of optimize_shifts that captures intermediate
    states during optimization for visualization.

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
    tracker : OptimizationTracker
        Tracker to capture optimization progress.
    setting : str | tuple
        Type of alignment (see optimize_shifts documentation).
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
    import math
    from warpylib.cubic_grid import CubicGrid
    from .optimize_global import AlignmentNanError

    # Move to device
    tilt_series.to(device)
    model.to(device)
    model.freeze()
    model.eval()
    images = images.to(device)

    # Set up parameters based on setting
    parameters = None
    if setting == "global":
        initial_tilt_axis_offset_y = tilt_series.tilt_axis_offset_y.clone()
        initial_tilt_axis_offset_x = tilt_series.tilt_axis_offset_x.clone()

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
    elif len(setting) == 2:
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
    elif len(setting) == 4:
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

    loss_values = []
    device_type = "cuda" if str(device).startswith("cuda") else "cpu"

    # Storage for tracking (we'll accumulate across batches)
    all_subvolumes_list = []
    all_precisions_list = []
    all_scores_list = []
    all_positions_list = []

    def closure():
        alignment_optimizer.zero_grad()

        # Update alignments
        if setting == "global":
            tilt_series.tilt_axis_offset_y = initial_tilt_axis_offset_y + shifts_y
            tilt_series.tilt_axis_offset_x = initial_tilt_axis_offset_x + shifts_x

        batches = int(math.ceil(positions.shape[0] / batch_size))
        total_samples = positions.shape[0]
        total_weighted_score = 0.0
        total_precision = 0.0

        # Clear tracking lists
        all_subvolumes_list.clear()
        all_precisions_list.clear()
        all_scores_list.clear()
        all_positions_list.clear()

        with torch.amp.autocast(device_type=device_type, enabled=False):
            for b in range(batches):
                if b == batches - 1:
                    batch_positions = positions[b * batch_size :]
                else:
                    batch_positions = positions[b * batch_size : (b + 1) * batch_size]

                current_batch_size = batch_positions.shape[0]

                # Reconstruct subvolumes
                subvolumes = tilt_series.reconstruct_subvolumes_single(
                    tilt_data=images,
                    coords=batch_positions.to(device),
                    pixel_size=pixel_size,
                    size=patch_size,
                    apply_ctf=apply_ctf,
                    oversampling=2.0,
                )

                # Normalize
                mean = einops.reduce(subvolumes, "n d h w -> n 1 1 1", reduction="mean")
                std = torch.std(subvolumes, dim=(-3, -2, -1), keepdim=True)
                eps = 1e-8
                subvolumes = (subvolumes - mean) / (std + eps)
                subvolumes_input = einops.rearrange(subvolumes, "b d h w -> b 1 d h w")

                # Get scores and precisions
                batch_scores, batch_log_precisions = model(subvolumes_input)
                batch_precisions = batch_log_precisions.exp()

                # Store for tracking (before rearranging)
                if tracker.should_capture_detailed():
                    all_subvolumes_list.append(subvolumes.detach())
                    all_precisions_list.append(batch_precisions.detach())
                    all_scores_list.append(batch_scores.detach())
                    all_positions_list.append(batch_positions.detach())

                # Compute loss
                batch_weighted_score = (batch_scores * batch_precisions).sum()
                batch_precision_sum = batch_precisions.sum()

                weighted_loss = batch_weighted_score * (
                    current_batch_size / total_samples
                )
                weighted_loss.backward()

                total_weighted_score += batch_weighted_score.item()
                total_precision += batch_precision_sum.item()

        # Compute final loss
        if total_precision <= 0:
            raise ValueError(f"Total precision is {total_precision}, which is <= 0.")
        avg_score = total_weighted_score / total_precision

        if math.isnan(avg_score):
            raise AlignmentNanError("Loss value is NaN")

        loss_values.append(avg_score)

        # Concatenate all tracked data
        if tracker.should_capture_detailed() and all_subvolumes_list:
            all_subvolumes = torch.cat(all_subvolumes_list, dim=0)
            all_precisions = torch.cat(all_precisions_list, dim=0)
            all_scores = torch.cat(all_scores_list, dim=0)
            all_positions = torch.cat(all_positions_list, dim=0)
        else:
            all_subvolumes = None
            all_precisions = None
            all_scores = None
            all_positions = None

        # Prepare alignment parameters based on setting
        shifts_x_param = None
        shifts_y_param = None
        grid_movement_x_param = None
        grid_movement_y_param = None
        grid_volume_warp_x_param = None
        grid_volume_warp_y_param = None
        grid_volume_warp_z_param = None

        if setting == "global":
            shifts_x_param = tilt_series.tilt_axis_offset_x
            shifts_y_param = tilt_series.tilt_axis_offset_y
        elif len(setting) == 2:
            # 2D warping mode
            grid_movement_x_param = tilt_series.grid_movement_x.values
            grid_movement_y_param = tilt_series.grid_movement_y.values
        elif len(setting) == 4:
            # 3D volume warping mode
            grid_volume_warp_x_param = tilt_series.grid_volume_warp_x.values
            grid_volume_warp_y_param = tilt_series.grid_volume_warp_y.values
            grid_volume_warp_z_param = tilt_series.grid_volume_warp_z.values

        # Call tracker
        tracker.on_closure_call(
            loss=avg_score,
            total_precision=total_precision,
            mean_precision=total_precision / total_samples
            if total_samples > 0
            else 0.0,
            subvolumes=all_subvolumes,
            precisions=all_precisions,
            scores=all_scores,
            shifts_x=shifts_x_param,
            shifts_y=shifts_y_param,
            positions=all_positions,
            grid_movement_x=grid_movement_x_param,
            grid_movement_y=grid_movement_y_param,
            grid_volume_warp_x=grid_volume_warp_x_param,
            grid_volume_warp_y=grid_volume_warp_y_param,
            grid_volume_warp_z=grid_volume_warp_z_param,
        )

        return avg_score

    # Run optimization
    n_iters = 1
    for _ in range(n_iters):
        alignment_optimizer.step(closure)

    # Finalize
    if setting == "global":
        tilt_series.tilt_axis_offset_y = initial_tilt_axis_offset_y + shifts_y.detach()
        tilt_series.tilt_axis_offset_x = initial_tilt_axis_offset_x + shifts_x.detach()
    elif len(setting) == 2:
        tilt_series.grid_movement_x.values = tilt_series.grid_movement_x.values.detach()
        tilt_series.grid_movement_y.values = tilt_series.grid_movement_y.values.detach()
    elif len(setting) == 4:
        tilt_series.grid_volume_warp_x.values = (
            tilt_series.grid_volume_warp_x.values.detach()
        )
        tilt_series.grid_volume_warp_y.values = (
            tilt_series.grid_volume_warp_y.values.detach()
        )
        tilt_series.grid_volume_warp_z.values = (
            tilt_series.grid_volume_warp_z.values.detach()
        )

    # Save summary
    tracker.save_summary()

    # Move back to CPU
    tilt_series.to("cpu")
    model.to("cpu")

    return tilt_series, loss_values


def create_3d_scatter_plot(
    positions: torch.Tensor,
    values: torch.Tensor,
    title: str,
    colorbar_label: str,
    cmap: str = "viridis",
    figsize: tuple[int, int] = (10, 8),
    elev: float = 30,
    azim: float = 45,
):
    """Create a 3D scatter plot colored by values.

    Parameters
    ----------
    positions : torch.Tensor
        3D positions, shape (n, 3) in (x, y, z) order.
    values : torch.Tensor
        Values to color points by, shape (n,).
    title : str
        Plot title.
    colorbar_label : str
        Label for the colorbar.
    cmap : str
        Matplotlib colormap name.
    figsize : tuple[int, int]
        Figure size (width, height).
    elev : float
        Elevation angle for 3D view.
    azim : float
        Azimuth angle for 3D view.

    Returns
    -------
    matplotlib.figure.Figure
        The created figure.
    """
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection="3d")

    # Convert to numpy
    pos = positions.cpu().numpy()
    vals = values.cpu().numpy()

    # Create scatter plot
    scatter = ax.scatter(
        pos[:, 0],
        pos[:, 1],
        pos[:, 2],
        c=vals,
        cmap=cmap,
        s=50,
        alpha=0.6,
        edgecolors="k",
        linewidths=0.5,
    )

    # Set labels
    ax.set_xlabel("X (Angstroms)", fontsize=12)
    ax.set_ylabel("Y (Angstroms)", fontsize=12)
    ax.set_zlabel("Z (Angstroms)", fontsize=12)
    ax.set_title(title, fontsize=14, fontweight="bold")

    # Set viewing angle
    ax.view_init(elev=elev, azim=azim)

    # Add colorbar
    cbar = plt.colorbar(scatter, ax=ax, pad=0.1, shrink=0.8)
    cbar.set_label(colorbar_label, fontsize=12)

    plt.tight_layout()

    return fig


def generate_3d_animation_frames(
    step_data_list: list[OptimizationStepData],
    value_key: str = "precision",
    cmap: str = "viridis",
    elev: float = 30,
    azim: float = 45,
    figsize: tuple[int, int] = (10, 8),
) -> list:
    """Generate frames for 3D animation.

    Parameters
    ----------
    step_data_list : list[OptimizationStepData]
        List of optimization step data.
    value_key : str
        Which value to color by: "precision" or "loss".
    cmap : str
        Matplotlib colormap name.
    elev : float
        Elevation angle for 3D view.
    azim : float
        Azimuth angle for 3D view.
    figsize : tuple[int, int]
        Figure size (width, height).

    Returns
    -------
    list
        List of figure objects for each step.
    """
    import matplotlib.pyplot as plt

    frames = []

    # Determine global value range for consistent coloring
    all_values = []
    for step_data in step_data_list:
        if step_data.positions is None:
            continue
        if value_key == "precision" and step_data.precisions is not None:
            all_values.append(step_data.precisions)
        elif value_key == "loss" and step_data.scores is not None:
            all_values.append(step_data.scores)

    if not all_values:
        raise ValueError(f"No {value_key} data found in step data")

    all_values = torch.cat(all_values)
    vmin, vmax = all_values.min().item(), all_values.max().item()

    for step_data in step_data_list:
        if step_data.positions is None:
            continue

        # Get values to plot
        if value_key == "precision":
            values = step_data.precisions
            title = f"Step {step_data.step}: Precision (Loss={step_data.loss:.4f})"
            colorbar_label = "Precision"
        elif value_key == "loss":
            values = step_data.scores
            title = f"Step {step_data.step}: Loss (Avg={step_data.loss:.4f})"
            colorbar_label = "Loss"
        else:
            raise ValueError(f"Unknown value_key: {value_key}")

        if values is None:
            continue

        # Create plot
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111, projection="3d")

        pos = step_data.positions.cpu().numpy()
        vals = values.cpu().numpy()

        scatter = ax.scatter(
            pos[:, 0],
            pos[:, 1],
            pos[:, 2],
            c=vals,
            cmap=cmap,
            s=50,
            alpha=0.6,
            edgecolors="k",
            linewidths=0.5,
            vmin=vmin,
            vmax=vmax,
        )

        ax.set_xlabel("X (Angstroms)", fontsize=12)
        ax.set_ylabel("Y (Angstroms)", fontsize=12)
        ax.set_zlabel("Z (Angstroms)", fontsize=12)
        ax.set_title(title, fontsize=14, fontweight="bold")
        ax.view_init(elev=elev, azim=azim)

        cbar = plt.colorbar(scatter, ax=ax, pad=0.1, shrink=0.8)
        cbar.set_label(colorbar_label, fontsize=12)

        plt.tight_layout()

        frames.append(fig)

    return frames


def save_animation_as_gif(
    frames: list,
    output_path: Path,
    duration: float = 500,
    dpi: int = 100,
):
    """Save animation frames as an animated GIF.

    Parameters
    ----------
    frames : list
        List of matplotlib figure objects.
    output_path : Path
        Output GIF file path.
    duration : float
        Duration of each frame in milliseconds.
    dpi : int
        DPI for rendering frames.
    """
    import io

    import matplotlib.pyplot as plt
    from PIL import Image

    images = []

    for fig in frames:
        # Render figure to image
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=dpi, bbox_inches="tight")
        buf.seek(0)
        img = Image.open(buf)
        images.append(img.copy())
        buf.close()
        plt.close(fig)

    # Save as GIF
    if images:
        images[0].save(
            output_path,
            save_all=True,
            append_images=images[1:],
            duration=duration,
            loop=0,
            optimize=False,
        )


def create_3d_visualizations(
    tracking_dir: Path,
    output_dir: Path,
    create_gifs: bool = True,
    gif_duration: float = 500,
    elev: float = 30,
    azim: float = 45,
):
    """Create 3D visualizations from optimization tracking data.

    Creates static plots for first and last steps, and optionally
    animated GIFs showing the evolution across all steps.

    Parameters
    ----------
    tracking_dir : Path
        Directory containing step data.
    output_dir : Path
        Directory to save visualizations.
    create_gifs : bool
        Whether to create animated GIFs.
    gif_duration : float
        Duration of each GIF frame in milliseconds.
    elev : float
        Elevation angle for 3D view.
    azim : float
        Azimuth angle for 3D view.
    """
    import matplotlib.pyplot as plt

    output_dir.mkdir(parents=True, exist_ok=True)

    # Load data
    print("Loading optimization data...")
    step_data = load_optimization_data(tracking_dir, load_subvolumes=False)

    if len(step_data) == 0:
        print("Warning: No step data found!")
        return

    # Filter steps with position data
    steps_with_positions = [s for s in step_data if s.positions is not None]

    if len(steps_with_positions) == 0:
        print("Warning: No position data found!")
        return

    print(f"Found {len(steps_with_positions)} steps with position data")

    # Create static plots for first and last steps
    print("Creating static plots...")

    # First step - Precision
    if steps_with_positions[0].precisions is not None:
        fig = create_3d_scatter_plot(
            positions=steps_with_positions[0].positions,
            values=steps_with_positions[0].precisions,
            title=f"Step {steps_with_positions[0].step}: Initial Precision",
            colorbar_label="Precision",
            cmap="viridis",
            elev=elev,
            azim=azim,
        )
        fig.savefig(
            output_dir / "3d_precision_initial.png", dpi=300, bbox_inches="tight"
        )
        plt.close(fig)

    # Last step - Precision
    if steps_with_positions[-1].precisions is not None:
        fig = create_3d_scatter_plot(
            positions=steps_with_positions[-1].positions,
            values=steps_with_positions[-1].precisions,
            title=f"Step {steps_with_positions[-1].step}: Final Precision",
            colorbar_label="Precision",
            cmap="viridis",
            elev=elev,
            azim=azim,
        )
        fig.savefig(output_dir / "3d_precision_final.png", dpi=300, bbox_inches="tight")
        plt.close(fig)

    # First step - Loss
    if steps_with_positions[0].scores is not None:
        fig = create_3d_scatter_plot(
            positions=steps_with_positions[0].positions,
            values=steps_with_positions[0].scores,
            title=f"Step {steps_with_positions[0].step}: Initial Loss",
            colorbar_label="Loss",
            cmap="coolwarm",
            elev=elev,
            azim=azim,
        )
        fig.savefig(output_dir / "3d_loss_initial.png", dpi=300, bbox_inches="tight")
        plt.close(fig)

    # Last step - Loss
    if steps_with_positions[-1].scores is not None:
        fig = create_3d_scatter_plot(
            positions=steps_with_positions[-1].positions,
            values=steps_with_positions[-1].scores,
            title=f"Step {steps_with_positions[-1].step}: Final Loss",
            colorbar_label="Loss",
            cmap="coolwarm",
            elev=elev,
            azim=azim,
        )
        fig.savefig(output_dir / "3d_loss_final.png", dpi=300, bbox_inches="tight")
        plt.close(fig)

    # Create animated GIFs
    if create_gifs:
        print("Creating animated GIFs...")

        # Precision animation
        if steps_with_positions[0].precisions is not None:
            print("  Generating precision animation...")
            frames = generate_3d_animation_frames(
                steps_with_positions,
                value_key="precision",
                cmap="viridis",
                elev=elev,
                azim=azim,
            )
            save_animation_as_gif(
                frames,
                output_dir / "3d_precision_evolution.gif",
                duration=gif_duration,
            )
            print(f"    Saved: {output_dir / '3d_precision_evolution.gif'}")

        # Loss animation
        if steps_with_positions[0].scores is not None:
            print("  Generating loss animation...")
            frames = generate_3d_animation_frames(
                steps_with_positions,
                value_key="loss",
                cmap="coolwarm",
                elev=elev,
                azim=azim,
            )
            save_animation_as_gif(
                frames,
                output_dir / "3d_loss_evolution.gif",
                duration=gif_duration,
            )
            print(f"    Saved: {output_dir / '3d_loss_evolution.gif'}")

    print("3D visualizations complete!")

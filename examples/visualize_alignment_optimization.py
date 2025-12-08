"""Example: Visualize alignment optimization progress.

This script demonstrates how to track and visualize the L-BFGS optimization
process during tilt-series alignment. It captures subvolumes, precisions, and
losses at each optimization step for analysis and presentation.

Usage
-----
python examples/visualize_alignment_optimization.py \\
    --model-checkpoint /path/to/model.ckpt \\
    --tilt-series /path/to/series.xml \\
    --output-dir /path/to/output \\
    --device cuda:0 \\
    --alignment-setting global

Alignment settings:
- 'global': Single shift per image (default)
- '(3,3)': 2D warping field per image
- '(3,3,2,10)': 3D volume warp grid

The script will:
1. Run alignment optimization with tracking enabled
2. Save intermediate states (subvolumes, precisions, losses) to disk
3. Generate visualization plots showing optimization progress
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import torch
from warpylib.tilt_series.reconstruct_volume import preprocess_tilt_data

from miss_alignment.alignment.tilt_series import generate_position_grid
from miss_alignment.alignment.visualize_alignment import (
    OptimizationTracker,
    create_3d_visualizations,
    load_optimization_data,
    optimize_shifts_with_tracking,
)
from miss_alignment.data.io import TiltSeriesData
from miss_alignment.models import MissAlignment


def parse_alignment_setting(setting_str: str):
    """Parse alignment setting string.

    Parameters
    ----------
    setting_str : str
        Either 'global' or a tuple string like '(3,3)' or '(3,3,2,10)'.

    Returns
    -------
    str or tuple
        Parsed alignment setting.
    """
    setting_str = setting_str.strip()
    if setting_str.lower() == "global":
        return "global"

    # Parse tuple string like "(3,3)" or "(3,3,2,10)"
    try:
        # Remove parentheses and split by comma
        tuple_str = setting_str.strip("()")
        values = tuple(int(x.strip()) for x in tuple_str.split(","))
        return values
    except (ValueError, AttributeError):
        raise ValueError(
            f"Invalid alignment setting: {setting_str}. "
            "Must be 'global' or tuple like '(3,3)' or '(3,3,2,10)'"
        )


def main():
    parser = argparse.ArgumentParser(
        description="Visualize alignment optimization progress"
    )
    parser.add_argument(
        "--model-checkpoint",
        type=Path,
        required=True,
        help="Path to trained model checkpoint",
    )
    parser.add_argument(
        "--tilt-series",
        type=Path,
        required=True,
        help="Path to tilt-series XML file",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory to save visualization outputs",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda:0",
        help="Device to run optimization on (default: cuda:0)",
    )
    parser.add_argument(
        "--patch-size",
        type=int,
        default=96,
        help="Reconstruction patch size (default: 96)",
    )
    parser.add_argument(
        "--patch-overlap",
        type=float,
        default=0.1,
        help="Patch overlap fraction (default: 0.1)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
        help="Batch size for reconstruction (default: 16)",
    )
    parser.add_argument(
        "--downsample",
        type=int,
        default=1,
        help="Downsampling factor for images (default: 1)",
    )
    parser.add_argument(
        "--max-subvolumes-per-step",
        type=int,
        default=32,
        help="Maximum number of subvolumes to save per step (default: 32)",
    )
    parser.add_argument(
        "--capture-frequency",
        type=int,
        default=1,
        help="Capture detailed data every N steps (default: 1)",
    )
    parser.add_argument(
        "--no-subvolumes",
        action="store_true",
        help="Don't save subvolumes (only track losses/precisions)",
    )
    parser.add_argument(
        "--alignment-setting",
        type=str,
        default="global",
        help=(
            "Alignment setting: 'global' for single shift per image, "
            "or tuple string like '(3,3)' for 2D warping field per image, "
            "or '(3,3,2,10)' for 3D volume warp grid (default: global)"
        ),
    )

    args = parser.parse_args()

    # Parse alignment setting
    alignment_setting = parse_alignment_setting(args.alignment_setting)

    # Create output directory
    args.output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("Alignment Optimization Visualization")
    print("=" * 60)
    print(f"Model: {args.model_checkpoint}")
    print(f"Tilt-series: {args.tilt_series}")
    print(f"Output: {args.output_dir}")
    print(f"Device: {args.device}")
    print(f"Alignment setting: {alignment_setting}")
    print("=" * 60)

    # Load model
    print("\nLoading model...")
    model = MissAlignment.load_from_checkpoint(
        args.model_checkpoint,
        map_location="cpu",
    )

    # Load tilt-series data
    print("Loading tilt-series...")
    tilt_series_data = TiltSeriesData(xml_metadata_path=args.tilt_series)
    tilt_series, images, pixel_size = tilt_series_data.load_metadata_and_stack(
        downsample=args.downsample
    )

    # Preprocess images
    print("Preprocessing images...")
    images = preprocess_tilt_data(
        tilt_data=images,
        normalize=True,
        invert=False,
        subvolume_size=args.patch_size,
    )

    # Generate position grid
    print("Generating position grid...")
    position_grid = generate_position_grid(
        volume_dimensions_physical=tilt_series.volume_dimensions_physical,
        pixel_size=pixel_size,
        patch_size=args.patch_size,
        patch_overlap=args.patch_overlap,
    )
    print(f"  Total positions: {position_grid.shape[0]}")

    # Create optimization tracker
    tracking_dir = args.output_dir / "optimization_steps"
    tracker = OptimizationTracker(
        output_dir=tracking_dir,
        capture_frequency=args.capture_frequency,
        max_subvolumes_per_step=args.max_subvolumes_per_step,
        save_subvolumes=not args.no_subvolumes,
        alignment_setting=alignment_setting,
    )

    # Run optimization with tracking
    print("\nRunning alignment optimization with tracking...")
    print(f"  Saving step data to: {tracking_dir}")
    tilt_series, loss_values = optimize_shifts_with_tracking(
        model=model,
        tilt_series=tilt_series,
        images=images,
        pixel_size=pixel_size,
        positions=position_grid,
        tracker=tracker,
        setting=alignment_setting,
        patch_size=args.patch_size,
        batch_size=args.batch_size,
        apply_ctf=True,
        device=args.device,
    )

    print(f"\nOptimization complete!")
    print(f"  Total steps: {tracker.current_step}")
    print(f"  Initial loss: {loss_values[0]:.6f}")
    print(f"  Final loss: {loss_values[-1]:.6f}")
    print(f"  Loss reduction: {loss_values[0] - loss_values[-1]:.6f}")

    # Save optimized tilt-series
    output_xml = args.output_dir / "aligned.xml"
    new_tilt_series_data = tilt_series_data.replace(xml_metadata_path=output_xml)
    new_tilt_series_data.save_metadata_to_xml(tilt_series)
    print(f"\nSaved aligned tilt-series to: {output_xml}")

    # Generate visualizations
    print("\nGenerating 2D visualizations...")
    plot_optimization_progress(tracking_dir, args.output_dir)

    # Generate 3D visualizations
    print("\nGenerating 3D visualizations...")
    create_3d_visualizations(
        tracking_dir=tracking_dir,
        output_dir=args.output_dir,
        create_gifs=True,
        gif_duration=500,
    )

    print("\n" + "=" * 60)
    print("Visualization complete!")
    print("=" * 60)
    print(f"\nOutputs saved to: {args.output_dir}")
    print(f"  - optimization_steps/: Step-by-step data")
    print(f"  - summary.txt: Text summary of optimization")
    print(f"  - loss_curve.png: Loss vs. step plot")
    print(f"  - precision_curve.png: Precision vs. step plot")
    print(f"  - shift_evolution.png: Shift evolution plot")
    print(f"  - 3d_precision_initial.png: Initial 3D precision scatter plot")
    print(f"  - 3d_precision_final.png: Final 3D precision scatter plot")
    print(f"  - 3d_loss_initial.png: Initial 3D loss scatter plot")
    print(f"  - 3d_loss_final.png: Final 3D loss scatter plot")
    print(f"  - 3d_precision_evolution.gif: Animated precision evolution")
    print(f"  - 3d_loss_evolution.gif: Animated loss evolution")


def plot_optimization_progress(tracking_dir: Path, output_dir: Path):
    """Generate visualization plots from optimization data.

    Parameters
    ----------
    tracking_dir : Path
        Directory containing step data.
    output_dir : Path
        Directory to save plots.
    """
    # Load data
    print("  Loading optimization data...")
    step_data = load_optimization_data(tracking_dir, load_subvolumes=False)

    if len(step_data) == 0:
        print("  Warning: No step data found!")
        return

    # Extract data for plotting
    steps = [s.step for s in step_data]
    losses = [s.loss for s in step_data]
    mean_precisions = [s.mean_precision for s in step_data]

    # Plot loss curve
    print("  Creating loss curve plot...")
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(steps, losses, marker="o", linewidth=2, markersize=6)
    ax.set_xlabel("Optimization Step", fontsize=12)
    ax.set_ylabel("Loss (Precision-Weighted Score)", fontsize=12)
    ax.set_title("L-BFGS Alignment Optimization Progress", fontsize=14, fontweight="bold")
    ax.grid(True, alpha=0.3)
    ax.set_xlim(left=0)
    plt.tight_layout()
    plt.savefig(output_dir / "loss_curve.png", dpi=300, bbox_inches="tight")
    plt.close()

    # Plot precision curve
    print("  Creating precision curve plot...")
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(steps, mean_precisions, marker="s", linewidth=2, markersize=6, color="C1")
    ax.set_xlabel("Optimization Step", fontsize=12)
    ax.set_ylabel("Mean Precision", fontsize=12)
    ax.set_title("Model Precision During Optimization", fontsize=14, fontweight="bold")
    ax.grid(True, alpha=0.3)
    ax.set_xlim(left=0)
    plt.tight_layout()
    plt.savefig(output_dir / "precision_curve.png", dpi=300, bbox_inches="tight")
    plt.close()

    # Plot shift evolution if available
    if step_data[0].shifts_x is not None:
        print("  Creating shift evolution plot...")
        plot_shift_evolution(step_data, output_dir)

    # Combined plot
    print("  Creating combined plot...")
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10))

    # Loss subplot
    ax1.plot(steps, losses, marker="o", linewidth=2, markersize=6)
    ax1.set_xlabel("Optimization Step", fontsize=12)
    ax1.set_ylabel("Loss", fontsize=12)
    ax1.set_title("Loss Evolution", fontsize=12, fontweight="bold")
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(left=0)

    # Precision subplot
    ax2.plot(steps, mean_precisions, marker="s", linewidth=2, markersize=6, color="C1")
    ax2.set_xlabel("Optimization Step", fontsize=12)
    ax2.set_ylabel("Mean Precision", fontsize=12)
    ax2.set_title("Precision Evolution", fontsize=12, fontweight="bold")
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(left=0)

    plt.tight_layout()
    plt.savefig(output_dir / "optimization_summary.png", dpi=300, bbox_inches="tight")
    plt.close()

    print("  Plots saved!")


def plot_shift_evolution(step_data: list, output_dir: Path):
    """Plot evolution of shifts during optimization.

    Parameters
    ----------
    step_data : list
        List of OptimizationStepData.
    output_dir : Path
        Directory to save plots.
    """
    n_tilts = step_data[0].shifts_x.shape[0]
    steps = [s.step for s in step_data]

    # Extract shifts for each tilt
    shifts_x = torch.stack([s.shifts_x for s in step_data], dim=0)  # (n_steps, n_tilts)
    shifts_y = torch.stack([s.shifts_y for s in step_data], dim=0)  # (n_steps, n_tilts)

    # Compute shift magnitude
    shift_magnitude = torch.sqrt(shifts_x**2 + shifts_y**2)  # (n_steps, n_tilts)

    # Plot shift magnitude evolution for all tilts
    fig, ax = plt.subplots(figsize=(12, 6))

    # Plot mean shift magnitude
    mean_magnitude = shift_magnitude.mean(dim=1)
    ax.plot(
        steps,
        mean_magnitude,
        linewidth=3,
        color="black",
        label="Mean shift magnitude",
        zorder=10,
    )

    # Plot individual tilts with transparency
    for i in range(n_tilts):
        ax.plot(steps, shift_magnitude[:, i], alpha=0.2, color="C0", linewidth=1)

    ax.set_xlabel("Optimization Step", fontsize=12)
    ax.set_ylabel("Shift Magnitude (Angstroms)", fontsize=12)
    ax.set_title(
        "Shift Evolution During Optimization", fontsize=14, fontweight="bold"
    )
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(left=0)
    plt.tight_layout()
    plt.savefig(output_dir / "shift_evolution.png", dpi=300, bbox_inches="tight")
    plt.close()

    # Plot X and Y shifts separately
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))

    # X shifts
    mean_x = shifts_x.mean(dim=1)
    ax1.plot(steps, mean_x, linewidth=3, color="black", label="Mean X shift", zorder=10)
    for i in range(n_tilts):
        ax1.plot(steps, shifts_x[:, i], alpha=0.2, color="C0", linewidth=1)
    ax1.set_xlabel("Optimization Step", fontsize=12)
    ax1.set_ylabel("X Shift (Angstroms)", fontsize=12)
    ax1.set_title("X Shift Evolution", fontsize=12, fontweight="bold")
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(left=0)

    # Y shifts
    mean_y = shifts_y.mean(dim=1)
    ax2.plot(steps, mean_y, linewidth=3, color="black", label="Mean Y shift", zorder=10)
    for i in range(n_tilts):
        ax2.plot(steps, shifts_y[:, i], alpha=0.2, color="C0", linewidth=1)
    ax2.set_xlabel("Optimization Step", fontsize=12)
    ax2.set_ylabel("Y Shift (Angstroms)", fontsize=12)
    ax2.set_title("Y Shift Evolution", fontsize=12, fontweight="bold")
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(left=0)

    plt.tight_layout()
    plt.savefig(output_dir / "shift_evolution_xy.png", dpi=300, bbox_inches="tight")
    plt.close()


if __name__ == "__main__":
    main()

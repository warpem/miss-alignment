"""Statistics tracking and visualization for alignment optimization.

This module provides utilities for tracking alignment loss values across
tilt-series, storing them to JSON, and visualizing the distribution with
outlier detection.
"""

import json
from pathlib import Path
from typing import Optional

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats


def save_loss_to_json(
    tilt_series_name: str,
    loss_values: list[float],
    output_directory: Path,
) -> Path:
    """Save alignment loss values to a JSON file.

    Parameters
    ----------
    tilt_series_name : str
        Name of the tilt-series (without extension).
    loss_values : list[float]
        List of loss values from alignment optimization.
    output_directory : Path
        Directory to save the JSON file.

    Returns
    -------
    Path
        Path to the saved JSON file.
    """
    # Extract final loss value (last value in the list)
    final_loss = float(loss_values[-1]) if loss_values else None

    data = {
        "tilt_series": tilt_series_name,
        "final_loss": final_loss,
        "all_loss_values": [float(v) for v in loss_values],
        "n_optimization_steps": len(loss_values),
    }

    json_path = output_directory / f"{tilt_series_name}_alignment_loss.json"
    with open(json_path, "w") as f:
        json.dump(data, f, indent=2)

    return json_path


def load_all_losses(directory: Path) -> dict[str, float]:
    """Load all alignment loss values from JSON files in a directory.

    Parameters
    ----------
    directory : Path
        Directory containing alignment loss JSON files.

    Returns
    -------
    dict[str, float]
        Dictionary mapping tilt-series names to their final loss values.
    """
    losses = {}
    for json_path in directory.glob("*_alignment_loss.json"):
        with open(json_path, "r") as f:
            data = json.load(f)
            losses[data["tilt_series"]] = data["final_loss"]
    return losses


def identify_outliers(
    losses: dict[str, float],
    n_std: float = 3.0,
) -> tuple[list[str], float, float]:
    """Identify outlier tilt-series based on loss values.

    Parameters
    ----------
    losses : dict[str, float]
        Dictionary mapping tilt-series names to final loss values.
    n_std : float
        Number of standard deviations for outlier threshold.

    Returns
    -------
    tuple[list[str], float, float]
        List of outlier tilt-series names, mean loss, and std loss.
    """
    if not losses:
        return [], 0.0, 0.0

    loss_values = np.array(list(losses.values()))
    mean_loss = float(np.mean(loss_values))
    std_loss = float(np.std(loss_values))

    threshold = mean_loss + n_std * std_loss

    outliers = [name for name, loss in losses.items() if loss > threshold]

    return outliers, mean_loss, std_loss


def plot_loss_distribution(
    losses: dict[str, float],
    output_path: Path,
    n_std: float = 3.0,
    iteration: Optional[int] = None,
) -> None:
    """Plot histogram of loss values with fitted normal distribution.

    Parameters
    ----------
    losses : dict[str, float]
        Dictionary mapping tilt-series names to final loss values.
    output_path : Path
        Path to save the plot.
    n_std : float
        Number of standard deviations for outlier threshold.
    iteration : int, optional
        Iteration number to include in title.
    """
    if not losses:
        print("No loss values to plot.")
        return

    loss_values = np.array(list(losses.values()))
    outliers, mean_loss, std_loss = identify_outliers(losses, n_std)

    # Create figure
    fig, ax = plt.subplots(figsize=(10, 6))

    # Plot histogram
    n_bins = min(30, len(loss_values) // 2) if len(loss_values) > 5 else 10
    counts, bins, patches = ax.hist(
        loss_values,
        bins=n_bins,
        density=True,
        alpha=0.7,
        color="steelblue",
        edgecolor="black",
        label="Observed losses",
    )

    # Fit and plot normal distribution
    x = np.linspace(loss_values.min(), loss_values.max(), 100)
    fitted_normal = stats.norm.pdf(x, mean_loss, std_loss)
    ax.plot(x, fitted_normal, "r-", linewidth=2, label="Fitted normal")

    # Mark mean and threshold
    ax.axvline(
        mean_loss,
        color="green",
        linestyle="--",
        linewidth=2,
        label=f"Mean: {mean_loss:.3f}",
    )
    threshold = mean_loss + n_std * std_loss
    ax.axvline(
        threshold,
        color="red",
        linestyle="--",
        linewidth=2,
        label=f"Threshold ({n_std}σ): {threshold:.3f}",
    )

    # Add labels and title
    ax.set_xlabel("Final Loss Value", fontsize=12)
    ax.set_ylabel("Density", fontsize=12)
    title = "Alignment Loss Distribution"
    if iteration is not None:
        title += f" (Iteration {iteration})"
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.legend(loc="upper right", fontsize=10)
    ax.grid(True, alpha=0.3)

    # Add text with statistics
    stats_text = (
        f"N = {len(loss_values)}\n"
        f"Mean = {mean_loss:.3f}\n"
        f"Std = {std_loss:.3f}\n"
        f"Outliers = {len(outliers)}"
    )
    ax.text(
        0.02,
        0.98,
        stats_text,
        transform=ax.transAxes,
        fontsize=10,
        verticalalignment="top",
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
    )

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()

    print("\nAlignment statistics:")
    print(f"  Mean loss: {mean_loss:.4f}")
    print(f"  Std loss: {std_loss:.4f}")
    print(f"  Outlier threshold ({n_std}σ): {threshold:.4f}")
    print(f"  Number of outliers: {len(outliers)}/{len(loss_values)}")
    if outliers:
        print("  Outlier tilt-series:")
        for name in outliers:
            print(f"    - {name} (loss: {losses[name]:.4f})")
    print(f"  Plot saved to: {output_path}")


def filter_outlier_xml_files(
    training_directory: Path,
    outliers: list[str],
    iteration: int,
) -> None:
    """Move outlier XML files to a separate directory.

    Parameters
    ----------
    training_directory : Path
        Main training directory.
    outliers : list[str]
        List of outlier tilt-series names (without extension).
    iteration : int
        Current iteration number.
    """
    if not outliers:
        return

    # Create outliers directory
    outliers_dir = training_directory / f"iter{iteration}_outliers"
    outliers_dir.mkdir(exist_ok=True, parents=True)

    # Move outlier XML files
    for name in outliers:
        xml_path = training_directory / f"{name}.xml"
        if xml_path.exists():
            dest_path = outliers_dir / xml_path.name
            xml_path.rename(dest_path)
            print(f"Moved outlier {name}.xml to {outliers_dir}")
        else:
            print(f"Warning: Could not find {xml_path}")

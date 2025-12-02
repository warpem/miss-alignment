"""
Plot comparison of alignment errors between unaligned and aligned tilt-series.

Takes two JSON files produced by compare_to_ground_truth.py and plots
the average per-tilt misalignment magnitude across all models.
"""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def load_and_compute_magnitudes(json_path: Path) -> dict[str, np.ndarray]:
    """
    Load JSON file and compute error magnitudes for each model.

    Parameters
    ----------
    json_path : Path
        Path to JSON file with structure:
        {model_name: {x_error_angstrom: [...], y_error_angstrom: [...]}}

    Returns
    -------
    dict[str, np.ndarray]
        Dictionary mapping model names to error magnitude arrays
    """
    with open(json_path) as f:
        data = json.load(f)

    magnitudes = {}
    for model_name, errors in data.items():
        x_errors = np.array(errors["x_error_angstrom"])
        y_errors = np.array(errors["y_error_angstrom"])
        magnitudes[model_name] = np.sqrt(x_errors**2 + y_errors**2)

    return magnitudes


def compute_stats_per_tilt(
    magnitudes: dict[str, np.ndarray],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute average, min, and max error magnitude per tilt across all models.

    Parameters
    ----------
    magnitudes : dict[str, np.ndarray]
        Dictionary mapping model names to error magnitude arrays

    Returns
    -------
    tuple[np.ndarray, np.ndarray, np.ndarray]
        Average, min, and max error magnitude for each tilt index
    """
    # Stack all magnitude arrays (assuming all have same length)
    all_magnitudes = np.stack(list(magnitudes.values()), axis=0)
    avg = np.mean(all_magnitudes, axis=0)
    min_val = np.min(all_magnitudes, axis=0)
    max_val = np.max(all_magnitudes, axis=0)
    return avg, min_val, max_val


def main():
    parser = argparse.ArgumentParser(
        description="Plot alignment error comparison between unaligned and aligned data"
    )
    parser.add_argument(
        "--unaligned",
        type=Path,
        required=True,
        help="JSON file with unaligned alignment errors",
    )
    parser.add_argument(
        "--aligned",
        type=Path,
        required=True,
        help="JSON file with aligned alignment errors",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output path for the plot (optional, shows interactively if not provided)",
    )

    args = parser.parse_args()

    # Load and compute magnitudes
    print(f"Loading unaligned data from {args.unaligned}")
    unaligned_magnitudes = load_and_compute_magnitudes(args.unaligned)

    print(f"Loading aligned data from {args.aligned}")
    aligned_magnitudes = load_and_compute_magnitudes(args.aligned)

    # Compute stats per tilt
    unaligned_avg, unaligned_min, unaligned_max = compute_stats_per_tilt(
        unaligned_magnitudes
    )
    aligned_avg, aligned_min, aligned_max = compute_stats_per_tilt(aligned_magnitudes)

    # Create tilt indices (assuming 0-indexed)
    n_tilts = len(unaligned_avg)
    tilt_indices = np.arange(n_tilts)

    # Plot
    fig, ax = plt.subplots(figsize=(10, 6))

    # Plot min-max ranges as shaded regions
    ax.fill_between(
        tilt_indices, unaligned_min, unaligned_max, alpha=0.2, color="C0"
    )
    ax.fill_between(
        tilt_indices, aligned_min, aligned_max, alpha=0.2, color="C1"
    )

    # Plot average curves
    ax.plot(tilt_indices, unaligned_avg, label="Unaligned", linewidth=2, color="C0")
    ax.plot(tilt_indices, aligned_avg, label="Aligned", linewidth=2, color="C1")

    ax.set_xlabel("Tilt Index")
    ax.set_ylabel("Average Alignment Error (Å)")
    ax.set_title("Per-Tilt Alignment Error Comparison")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if args.output:
        plt.savefig(args.output, dpi=150)
        print(f"Plot saved to {args.output}")
    else:
        plt.show()


if __name__ == "__main__":
    main()
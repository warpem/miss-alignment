"""Statistics tracking for alignment optimization.

This module provides utilities for tracking alignment loss values across
tilt-series and storing them to JSON.
"""

import json
from pathlib import Path


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

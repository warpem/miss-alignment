"""Tests for alignment statistics tracking."""

import json
from pathlib import Path
import tempfile

from miss_alignment.alignment.statistics import save_loss_to_json


def test_save_loss_to_json():
    """Test saving loss values to JSON file."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        loss_values = [1.5, 1.2, 0.9, 0.8]
        tilt_series_name = "test_series"

        json_path = save_loss_to_json(
            tilt_series_name=tilt_series_name,
            loss_values=loss_values,
            output_directory=tmpdir,
        )

        assert json_path.exists()
        assert json_path.name == f"{tilt_series_name}_alignment_loss.json"

        with open(json_path, "r") as f:
            data = json.load(f)

        assert data["tilt_series"] == tilt_series_name
        assert data["final_loss"] == 0.8
        assert data["all_loss_values"] == loss_values
        assert data["n_optimization_steps"] == 4


def test_save_loss_to_json_empty():
    """Test saving empty loss values."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        loss_values = []
        tilt_series_name = "empty_series"

        json_path = save_loss_to_json(
            tilt_series_name=tilt_series_name,
            loss_values=loss_values,
            output_directory=tmpdir,
        )

        with open(json_path, "r") as f:
            data = json.load(f)

        assert data["final_loss"] is None
        assert data["n_optimization_steps"] == 0

"""Tests for alignment statistics tracking and visualization."""

import json
from pathlib import Path
import tempfile

import numpy as np

from miss_alignment.alignment.statistics import (
    save_loss_to_json,
    load_all_losses,
    identify_outliers,
    plot_loss_distribution,
    filter_outlier_xml_files,
)


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


def test_load_all_losses():
    """Test loading all loss values from directory."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)

        # Create multiple JSON files
        losses_data = {
            "series1": [1.5, 1.2, 0.9],
            "series2": [2.0, 1.8, 1.5],
            "series3": [0.5, 0.4, 0.3],
        }

        for name, values in losses_data.items():
            save_loss_to_json(name, values, tmpdir)

        # Load all losses
        loaded_losses = load_all_losses(tmpdir)

        assert len(loaded_losses) == 3
        assert loaded_losses["series1"] == 0.9
        assert loaded_losses["series2"] == 1.5
        assert loaded_losses["series3"] == 0.3


def test_identify_outliers():
    """Test outlier identification with known distribution."""
    # Create losses with known mean and std
    np.random.seed(42)
    normal_losses = np.random.normal(1.0, 0.2, 100)

    # Add some clear outliers
    losses = {
        **{f"normal_{i}": v for i, v in enumerate(normal_losses)},
        "outlier1": 2.0,  # way above mean + 3*std
        "outlier2": 2.5,
    }

    outliers, mean_loss, std_loss = identify_outliers(losses, n_std=3.0)

    # Check that mean and std are close to expected
    assert abs(mean_loss - 1.0) < 0.1
    assert abs(std_loss - 0.2) < 0.1

    # Check that outliers are identified
    assert len(outliers) >= 2
    assert "outlier1" in outliers
    assert "outlier2" in outliers


def test_identify_outliers_empty():
    """Test outlier identification with empty dict."""
    outliers, mean_loss, std_loss = identify_outliers({}, n_std=3.0)

    assert outliers == []
    assert mean_loss == 0.0
    assert std_loss == 0.0


def test_identify_outliers_no_outliers():
    """Test with tight normal distribution (no outliers)."""
    losses = {f"series_{i}": 1.0 + 0.01 * i for i in range(10)}

    outliers, mean_loss, std_loss = identify_outliers(losses, n_std=3.0)

    assert len(outliers) == 0


def test_plot_loss_distribution():
    """Test plotting loss distribution."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)

        # Create sample losses
        np.random.seed(42)
        losses = {
            f"series_{i}": v for i, v in enumerate(np.random.normal(1.0, 0.2, 50))
        }
        losses["outlier"] = 2.5

        output_path = tmpdir / "test_plot.png"

        # Should not raise an error
        plot_loss_distribution(
            losses=losses,
            output_path=output_path,
            n_std=3.0,
            iteration=1,
        )

        assert output_path.exists()
        assert output_path.stat().st_size > 0


def test_plot_loss_distribution_empty(capsys):
    """Test plotting with empty losses."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        output_path = tmpdir / "empty_plot.png"

        plot_loss_distribution(
            losses={},
            output_path=output_path,
            n_std=3.0,
        )

        # Should print message and not create file
        captured = capsys.readouterr()
        assert "No loss values to plot" in captured.out
        assert not output_path.exists()


def test_filter_outlier_xml_files():
    """Test filtering outlier XML files."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)

        # Create some mock XML files
        xml_files = ["series1.xml", "series2.xml", "outlier1.xml", "outlier2.xml"]
        for fname in xml_files:
            (tmpdir / fname).write_text("mock xml content")

        # Filter outliers
        outliers = ["outlier1", "outlier2"]
        filter_outlier_xml_files(
            training_directory=tmpdir,
            outliers=outliers,
            iteration=1,
        )

        # Check that outliers are moved
        outliers_dir = tmpdir / "iter1_outliers"
        assert outliers_dir.exists()
        assert (outliers_dir / "outlier1.xml").exists()
        assert (outliers_dir / "outlier2.xml").exists()

        # Check that normal files remain
        assert (tmpdir / "series1.xml").exists()
        assert (tmpdir / "series2.xml").exists()

        # Check that outliers are removed from main directory
        assert not (tmpdir / "outlier1.xml").exists()
        assert not (tmpdir / "outlier2.xml").exists()


def test_filter_outlier_xml_files_empty():
    """Test filtering with no outliers."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)

        # Create some mock XML files
        (tmpdir / "series1.xml").write_text("mock xml content")

        # Filter with empty outliers list
        filter_outlier_xml_files(
            training_directory=tmpdir,
            outliers=[],
            iteration=1,
        )

        # Should not create outliers directory
        outliers_dir = tmpdir / "iter1_outliers"
        assert not outliers_dir.exists()

        # Files should remain
        assert (tmpdir / "series1.xml").exists()


def test_filter_outlier_xml_files_missing(capsys):
    """Test filtering when XML file doesn't exist."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)

        # Try to filter non-existent file
        filter_outlier_xml_files(
            training_directory=tmpdir,
            outliers=["nonexistent"],
            iteration=1,
        )

        # Should print warning
        captured = capsys.readouterr()
        assert "Warning" in captured.out or "Could not find" in captured.out

import pytest
import torch
from pathlib import Path
from unittest.mock import Mock, mock_open, patch

# Tomogrma class and the io functions
from torch_tomogram import Tomogram
from miss_alignment.data.io import (
    read_tomogram_from_pickle, save_tomogram_to_pickle
)


class TestSaveTomogramToPickle:
    """Test suite for save_tomogram_to_pickle function."""

    def test_valid_save(self, tmp_path):
        """Test successful saving with valid inputs."""
        # Create mock tomogram on CPU
        mock_tomogram = Mock(spec=Tomogram)
        mock_tomogram.device = torch.device("cpu")
        mock_tomogram.images = torch.randn(10, 100, 100)
        mock_tomogram.tilt_angles = torch.randn(10)
        mock_tomogram.tilt_axis_angle = torch.tensor(0.5)
        mock_tomogram.sample_translations = torch.randn(10, 2)

        save_path = tmp_path / "test.pickle"

        # Should not raise any exception
        save_tomogram_to_pickle(mock_tomogram, save_path)

        # Verify file was created
        assert save_path.exists()

    def test_invalid_file_extension(self):
        """Test error when save_path doesn't end with .pickle."""
        mock_tomogram = Mock(spec=Tomogram)
        mock_tomogram.device = torch.device("cpu")
        save_path = Path("test.txt")

        with pytest.raises(ValueError,
                           match="save_path must end with .pickle"):
            save_tomogram_to_pickle(mock_tomogram, save_path)

    def test_tomogram_not_on_cpu(self):
        """Test error when tomogram is not on CPU device."""
        mock_tomogram = Mock(spec=Tomogram)
        mock_tomogram.device = torch.device("cuda")
        save_path = Path("test.pickle")

        with pytest.raises(ValueError,
                           match="the Tomogram data should be on CPU for saving"):
            save_tomogram_to_pickle(mock_tomogram, save_path)


class TestReadTomogramFromPickle:
    """Test suite for read_tomogram_from_pickle function."""

    @patch('builtins.open', new_callable=mock_open)
    @patch('pickle.load')
    @patch('miss_alignment.data.io.Tomogram')
    def test_successful_read(self, mock_tomogram_class, mock_pickle_load,
                             mock_file):
        """Test successful reading and Tomogram reconstruction."""
        # Mock data dictionary
        mock_data = {
            "tilt_angles": torch.randn(10),
            "tilt_axis_angle": torch.tensor(0.5),
            "sample_translations": torch.randn(10, 2),
            "tilt_series": torch.randn(10, 100, 100)
        }
        mock_pickle_load.return_value = mock_data

        mock_tomogram_instance = Mock(spec=Tomogram)
        mock_tomogram_class.return_value = mock_tomogram_instance

        save_path = Path("test.pickle")
        result = read_tomogram_from_pickle(save_path)

        # Verify file was opened correctly
        mock_file.assert_called_once_with(save_path, "rb")

        # Verify Tomogram was constructed with correct parameters
        mock_tomogram_class.assert_called_once_with(
            tilt_angles=mock_data["tilt_angles"],
            tilt_axis_angle=mock_data["tilt_axis_angle"],
            sample_translations=mock_data["sample_translations"],
            images=mock_data["tilt_series"]
        )

        assert result == mock_tomogram_instance

    @patch('builtins.open', side_effect=FileNotFoundError)
    def test_file_not_found(self, mock_file):
        """Test error when pickle file doesn't exist."""
        save_path = Path("nonexistent.pickle")

        with pytest.raises(FileNotFoundError):
            read_tomogram_from_pickle(save_path)


class TestRoundTrip:
    """Integration test for save/load cycle."""

    def test_save_load_cycle(self, tmp_path):
        """Test that saved tomogram can be loaded correctly."""
        # Create a real tomogram-like object
        mock_tomogram = Mock(spec=Tomogram)
        mock_tomogram.device = torch.device("cpu")
        mock_tomogram.images = torch.randn(5, 50, 50)
        mock_tomogram.tilt_angles = torch.randn(5)
        mock_tomogram.tilt_axis_angle = torch.tensor(1.2)
        mock_tomogram.sample_translations = torch.randn(5, 2)

        save_path = tmp_path / "roundtrip.pickle"

        # Save tomogram
        save_tomogram_to_pickle(mock_tomogram, save_path)

        # Load tomogram back
        with patch('miss_alignment.data.io.Tomogram') as mock_tomogram_class:
            mock_loaded = Mock(spec=Tomogram)
            mock_tomogram_class.return_value = mock_loaded

            loaded_tomogram = read_tomogram_from_pickle(save_path)

            # Verify constructor was called with saved data
            args, kwargs = mock_tomogram_class.call_args
            assert torch.allclose(kwargs["tilt_angles"],
                                  mock_tomogram.tilt_angles)
            assert torch.allclose(kwargs["tilt_axis_angle"],
                                  mock_tomogram.tilt_axis_angle)
            assert torch.allclose(kwargs["sample_translations"],
                                  mock_tomogram.sample_translations)
            assert torch.allclose(kwargs["images"], mock_tomogram.images)
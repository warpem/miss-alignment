"""Tests for reconstruction worker module."""

import multiprocessing as mp
import pickle
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

import einops
import pytest
import torch

from miss_alignment.data._reconstruction_worker import (
    _create_pool_reconstruction,
    _generate_translations,
    reconstruction_worker,
)


@pytest.fixture
def temp_dir():
    """Create temporary directory for test files."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def mock_tilt_series():
    """Create mock tilt series object."""
    mock = Mock()
    mock.images = torch.randn(10, 128, 128)
    mock.tilt_angles = torch.linspace(-60, 60, 10)
    mock.tilt_axis_angle = torch.tensor([-96.0] * 10)
    mock.sample_translations = torch.zeros(10, 2)
    mock.reconstruct_subvolume = Mock(return_value=torch.randn(32, 32, 32))
    return mock


@pytest.fixture
def shift_generator():
    """Create simple shift generator function."""

    def generator(n_tilts):
        return torch.randn(n_tilts, 3)

    return generator


class TestGenerateTranslations:
    """Test translation generation functionality."""

    def test_raise_error_with_wrong_matrix_shape(self, shift_generator):
        matrix = torch.eye(3)
        rotation_matrices = einops.repeat(matrix, 'h w -> n h w', n=10)

        with pytest.raises(ValueError):
            translations = _generate_translations(shift_generator,
                                                  rotation_matrices)

    def test_basic_translation_generation(self, shift_generator):
        """Test that translations are generated with correct shape."""
        n_tilts = 10
        matrix = torch.eye(3)
        rotation_matrices = einops.repeat(matrix, 'h w -> n h w', n=n_tilts)
        projection_matrices = rotation_matrices[..., 1:3, :3]

        with patch('random.random', return_value=0.6):
            translations = _generate_translations(shift_generator,
                                                  projection_matrices)

        assert translations.shape == (n_tilts, 2)
        assert translations.dtype == torch.float32

    def test_translation_masking_y_axis(self, shift_generator):
        """Test that y-axis translations are zeroed when die_roll < 0.25."""
        matrix = torch.eye(3)
        rotation_matrices = einops.repeat(matrix, 'h w -> n h w', n=5)
        projection_matrices = rotation_matrices[..., 1:3, :3]

        with patch('random.random', return_value=0.1):
            translations = _generate_translations(shift_generator,
                                                  projection_matrices)

        assert torch.all(translations[:, 0] == 0.0)
        assert not torch.all(translations[:, 1] == 0.0)

    def test_translation_masking_x_axis(self, shift_generator):
        """Test that x-axis translations are zeroed when 0.25 <= die_roll < 0.5."""
        matrix = torch.eye(3)
        rotation_matrices = einops.repeat(matrix, 'h w -> n h w', n=5)
        projection_matrices = rotation_matrices[..., 1:3, :3]

        with patch('random.random', return_value=0.3):
            translations = _generate_translations(shift_generator,
                                                  projection_matrices)

        assert not torch.all(translations[:, 0] == 0.0)
        assert torch.all(translations[:, 1] == 0.0)


class TestCreatePoolReconstruction:
    """Test pool reconstruction creation."""

    @patch(
        'miss_alignment.data._reconstruction_worker.read_tomogram_from_pickle')
    def test_reconstruction_output_format(self, mock_read, mock_tilt_series,
                                          temp_dir, shift_generator):
        """Test that reconstruction returns correct format."""
        mock_read.return_value = mock_tilt_series
        tilt_series_path = temp_dir / "test.pickle"

        result = _create_pool_reconstruction(
            tilt_series_path=tilt_series_path,
            tomogram_shape=(128, 128, 128),
            patch_size=32,
            shift_generator=shift_generator,
        )

        assert len(result) == 3
        assert all(isinstance(item, tuple) for item in result)
        assert all(len(item) == 2 for item in result)
        assert all(isinstance(item[0], torch.Tensor) for item in result)
        assert all(item[1] in [1, -1] for item in result)

    @patch(
        'miss_alignment.data._reconstruction_worker.read_tomogram_from_pickle')
    @patch('random.uniform')
    def test_tilt_angle_augmentation(self, mock_uniform, mock_read,
                                     mock_tilt_series, temp_dir,
                                     shift_generator):
        """Test that tilt angles are augmented correctly."""
        mock_uniform.return_value = 5.0
        mock_read.return_value = mock_tilt_series
        original_angles = mock_tilt_series.tilt_angles.clone()

        _create_pool_reconstruction(
            tilt_series_path=temp_dir / "test.pickle",
            tomogram_shape=(128, 128, 128),
            patch_size=32,
            shift_generator=shift_generator,
        )

        expected_angles = original_angles + 5.0
        torch.testing.assert_close(mock_tilt_series.tilt_angles,
                                   expected_angles)


class TestReconstructionWorker:
    """Test reconstruction worker process."""

    def test_worker_initial_fill(self, temp_dir, shift_generator):
        """Test that worker fills initial pool correctly."""
        assigned_indices = [0, 1, 2]
        ready_flag = mp.Value('i', 0)
        stop_event = mp.Event()

        # Mock the reconstruction creation
        mock_data = [(torch.randn(32, 32, 32), 1),
                     (torch.randn(32, 32, 32), -1),
                     (torch.randn(32, 32, 32), 1)]

        with patch(
                'miss_alignment.data._reconstruction_worker'
                '._create_pool_reconstruction',
                return_value=mock_data):
            # Set stop event immediately to only do initial fill
            stop_event.set()

            reconstruction_worker(
                worker_id=0,
                assigned_indices=assigned_indices,
                pool_dir=temp_dir,
                tilt_series_pickles=[temp_dir / "dummy.pickle"],
                tomogram_shape=(128, 128, 128),
                patch_size=32,
                shift_generator=shift_generator,
                ready_flag=ready_flag,
                stop_event=stop_event,
                monitor=None,
            )

        # Check that files were created
        for idx in assigned_indices:
            file_path = temp_dir / f"recon_{idx}.pickle"
            assert file_path.exists()

            # Verify pickle content
            with open(file_path, 'rb') as f:
                data = pickle.load(f)
                assert len(data) == 3

    def test_worker_ready_flag(self, temp_dir, shift_generator):
        """Test that worker sets ready flag after initial fill."""
        ready_flag = mp.Value('i', 0)
        stop_event = mp.Event()
        stop_event.set()  # Stop immediately after initial fill

        with patch(
                'miss_alignment.data._reconstruction_worker'
                '._create_pool_reconstruction',
                return_value=[(torch.zeros(1), 1)] * 3):
            reconstruction_worker(
                worker_id=0,
                assigned_indices=[0],
                pool_dir=temp_dir,
                tilt_series_pickles=[temp_dir / "dummy.pickle"],
                tomogram_shape=(128, 128, 128),
                patch_size=32,
                shift_generator=shift_generator,
                ready_flag=ready_flag,
                stop_event=stop_event,
                monitor=None,
            )

        assert ready_flag.value == 1

    def test_worker_continuous_update(self, temp_dir, shift_generator):
        """Test that worker continuously updates pool when not stopped."""
        ready_flag = mp.Value('i', 0)
        stop_event = mp.Event()
        assigned_indices = [0, 1]

        update_count = 0
        max_updates = 3

        def mock_create(*args, **kwargs):
            nonlocal update_count
            update_count += 1
            # Stop after initial fill + max_updates
            if update_count > len(assigned_indices) + max_updates:
                stop_event.set()
            return [(torch.randn(32, 32, 32), 1)] * 3

        with patch(
                'miss_alignment.data._reconstruction_worker'
                '._create_pool_reconstruction',
                side_effect=mock_create):
            reconstruction_worker(
                worker_id=0,
                assigned_indices=assigned_indices,
                pool_dir=temp_dir,
                tilt_series_pickles=[temp_dir / "dummy.pickle"],
                tomogram_shape=(128, 128, 128),
                patch_size=32,
                shift_generator=shift_generator,
                ready_flag=ready_flag,
                stop_event=stop_event,
                monitor=None,
            )

        # Should have done initial fill + continuous updates
        assert update_count > len(assigned_indices)

    def test_worker_with_monitor(self, temp_dir, shift_generator):
        """Test that worker reports to monitor when provided."""
        mock_monitor = Mock()
        ready_flag = mp.Value('i', 0)
        stop_event = mp.Event()

        # Run one update cycle
        update_count = 0

        def mock_create(*args, **kwargs):
            nonlocal update_count
            update_count += 1
            if update_count > 2:  # Initial fill (1) + one update
                stop_event.set()
            return [(torch.zeros(1), 1)] * 3

        with patch(
                'miss_alignment.data._reconstruction_worker'
                '._create_pool_reconstruction',
                side_effect=mock_create):
            reconstruction_worker(
                worker_id=0,
                assigned_indices=[0],
                pool_dir=temp_dir,
                tilt_series_pickles=[temp_dir / "dummy.pickle"],
                tomogram_shape=(128, 128, 128),
                patch_size=32,
                shift_generator=shift_generator,
                ready_flag=ready_flag,
                stop_event=stop_event,
                monitor=mock_monitor,
            )

        # Monitor should have been called for continuous updates (not initial fill)
        assert mock_monitor.record_production.call_count >= 1
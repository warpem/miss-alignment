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
    TiltSeriesFetcher,
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

    def generator(n_tilts, device: str = 'cpu'):
        return torch.randn(n_tilts, 3, device=device)

    return generator


class TestTiltSeriesFetcher:
    """Test TiltSeriesFetcher class."""

    def test_initialization(self, temp_dir):
        """Test that TiltSeriesFetcher initializes correctly.

        Parameters
        ----------
        temp_dir : Path
            Temporary directory for test files
        """
        tilt_series_pickles = [temp_dir / "test.pickle"]
        refresh_rate = 5
        device = torch.device("cpu")

        fetcher = TiltSeriesFetcher(
            tilt_series_pickles=tilt_series_pickles,
            refresh_rate=refresh_rate,
            device=device
        )
        
        assert fetcher.tilt_series_pickles == tilt_series_pickles
        assert fetcher.refresh_rate == refresh_rate
        assert fetcher.device == device
        assert fetcher._counter == 0
        assert fetcher._tilt_series is None
    
    @patch('miss_alignment.data._reconstruction_worker.read_tomogram_from_pickle')
    def test_load_next(self, mock_read, mock_tilt_series, temp_dir):
        """Test that _load_next loads and processes a tilt series correctly.

        Parameters
        ----------
        mock_read : MagicMock
            Mock for read_tomogram_from_pickle function
        mock_tilt_series : Mock
            Mock tilt series object
        temp_dir : Path
            Temporary directory for test files
        """
        mock_read.return_value = mock_tilt_series
        tilt_series_pickles = [temp_dir / "test.pickle"]

        fetcher = TiltSeriesFetcher(
            tilt_series_pickles=tilt_series_pickles,
            refresh_rate=5,
            device=torch.device("cpu")
        )
        
        fetcher._load_next()
        
        assert fetcher._tilt_series is not None
        mock_read.assert_called_once()
        mock_tilt_series.to.assert_called_once()
    
    @patch('miss_alignment.data._reconstruction_worker.read_tomogram_from_pickle')
    def test_call_first_time(self, mock_read, mock_tilt_series, temp_dir):
        """Test that __call__ loads a new tilt series on first call."""
        mock_read.return_value = mock_tilt_series
        tilt_series_pickles = [temp_dir / "test.pickle"]
        
        fetcher = TiltSeriesFetcher(
            tilt_series_pickles=tilt_series_pickles,
            refresh_rate=5,
            device=torch.device("cpu")
        )
        
        result = fetcher()
        
        assert result is mock_tilt_series
        assert fetcher._counter == 1
        mock_read.assert_called_once()
    
    @patch('miss_alignment.data._reconstruction_worker.read_tomogram_from_pickle')
    def test_call_reuse(self, mock_read, mock_tilt_series, temp_dir):
        """Test that __call__ reuses tilt series within refresh rate."""
        mock_read.return_value = mock_tilt_series
        tilt_series_pickles = [temp_dir / "test.pickle"]
        
        fetcher = TiltSeriesFetcher(
            tilt_series_pickles=tilt_series_pickles,
            refresh_rate=5,
            device=torch.device("cpu")
        )
        
        # First call loads new
        fetcher()
        mock_read.reset_mock()
        
        # Second call should reuse
        result = fetcher()
        
        assert result is mock_tilt_series
        assert fetcher._counter == 2
        mock_read.assert_not_called()
    
    @patch('miss_alignment.data._reconstruction_worker.read_tomogram_from_pickle')
    def test_call_refresh(self, mock_read, mock_tilt_series, temp_dir):
        """Test that __call__ refreshes after reaching refresh rate."""
        mock_read.return_value = mock_tilt_series
        tilt_series_pickles = [temp_dir / "test.pickle"]
        
        fetcher = TiltSeriesFetcher(
            tilt_series_pickles=tilt_series_pickles,
            refresh_rate=2,
            device=torch.device("cpu")
        )
        
        # First call loads new
        fetcher()
        # Second call reuses
        fetcher()
        mock_read.reset_mock()
        
        # Third call should refresh
        result = fetcher()
        
        assert result is mock_tilt_series
        assert fetcher._counter == 1
        mock_read.assert_called_once()
    
    @patch('miss_alignment.data._reconstruction_worker.read_tomogram_from_pickle')
    def test_alignment_backup_restore(self, mock_read, mock_tilt_series, temp_dir):
        """Test that alignment parameters are backed up and restored correctly."""
        mock_read.return_value = mock_tilt_series
        tilt_series_pickles = [temp_dir / "test.pickle"]
        
        fetcher = TiltSeriesFetcher(
            tilt_series_pickles=tilt_series_pickles,
            refresh_rate=5,
            device=torch.device("cpu")
        )
        
        # First call loads and backs up
        fetcher()
        
        # Modify alignment parameters
        original_translations = mock_tilt_series.sample_translations.clone()
        mock_tilt_series.sample_translations = torch.ones_like(original_translations)
        
        # Second call should restore original values
        fetcher()
        
        # Check that original values were restored
        torch.testing.assert_close(mock_tilt_series.sample_translations, original_translations)


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

    def test_reconstruction_output_format(self, mock_tilt_series, shift_generator):
        """Test that reconstruction returns correct format."""
        result = _create_pool_reconstruction(
            tilt_series=mock_tilt_series,
            tomogram_shape=(128, 128, 128),
            patch_size=32,
            shift_generator=shift_generator,
        )

        assert len(result) == 3
        assert all(isinstance(item, tuple) for item in result)
        assert all(len(item) == 2 for item in result)
        assert all(isinstance(item[0], torch.Tensor) for item in result)
        assert 1 in [item[1] for item in result]
        assert -1 in [item[1] for item in result]

    @patch('random.uniform')
    def test_tilt_angle_augmentation(self, mock_uniform, mock_tilt_series, shift_generator):
        """Test that tilt angles are augmented correctly."""
        mock_uniform.return_value = 5.0
        original_angles = mock_tilt_series.tilt_angles.clone()

        _create_pool_reconstruction(
            tilt_series=mock_tilt_series,
            tomogram_shape=(128, 128, 128),
            patch_size=32,
            shift_generator=shift_generator,
        )

        expected_angles = original_angles + 5.0
        torch.testing.assert_close(mock_tilt_series.tilt_angles,
                                   expected_angles)


class TestReconstructionWorker:
    """Test reconstruction worker process."""

    @patch('miss_alignment.data._reconstruction_worker.TiltSeriesFetcher')
    def test_worker_initial_fill(self, mock_fetcher_class, temp_dir, shift_generator, mock_tilt_series):
        """Test that worker fills initial pool correctly."""
        assigned_indices = [0, 1, 2]
        ready_flag = mp.Value('i', 0)
        stop_event = mp.Event()

        # Mock the TiltSeriesFetcher instance
        mock_fetcher_instance = mock_fetcher_class.return_value
        mock_fetcher_instance.return_value = mock_tilt_series

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
                tilt_series_refresh_rate=10,
            )

        # Check that files were created
        for idx in assigned_indices:
            file_path = temp_dir / f"recon_{idx}.pickle"
            assert file_path.exists()

            # Verify pickle content
            with open(file_path, 'rb') as f:
                data = pickle.load(f)
                assert len(data) == 3

    @patch('miss_alignment.data._reconstruction_worker.TiltSeriesFetcher')
    def test_worker_ready_flag(self, mock_fetcher_class, temp_dir, shift_generator, mock_tilt_series):
        """Test that worker sets ready flag after initial fill."""
        ready_flag = mp.Value('i', 0)
        stop_event = mp.Event()
        stop_event.set()  # Stop immediately after initial fill

        # Mock the TiltSeriesFetcher instance
        mock_fetcher_instance = mock_fetcher_class.return_value
        mock_fetcher_instance.return_value = mock_tilt_series

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
                tilt_series_refresh_rate=10,
            )

        assert ready_flag.value == 1

    @patch('miss_alignment.data._reconstruction_worker.TiltSeriesFetcher')
    def test_worker_continuous_update(self, mock_fetcher_class, temp_dir, shift_generator, mock_tilt_series):
        """Test that worker continuously updates pool when not stopped."""
        ready_flag = mp.Value('i', 0)
        stop_event = mp.Event()
        assigned_indices = [0, 1]

        # Mock the TiltSeriesFetcher instance
        mock_fetcher_instance = mock_fetcher_class.return_value
        mock_fetcher_instance.return_value = mock_tilt_series

        update_count = 0
        max_updates = 3

        def mock_create(*args, **kwargs):
            nonlocal update_count
            update_count += 1
            # Stop after initial fill + max_updates
            if update_count > len(assigned_indices) + max_updates:
                stop_event.set()
            return [(torch.randn(32, 32, 32), 1),
                     (torch.randn(32, 32, 32), -1),
                     (torch.randn(32, 32, 32), 1)]

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
                tilt_series_refresh_rate=10,
            )

        # Should have done initial fill + continuous updates
        assert update_count > len(assigned_indices)

    @patch('miss_alignment.data._reconstruction_worker.TiltSeriesFetcher')
    def test_worker_with_monitor(self, mock_fetcher_class, temp_dir, shift_generator, mock_tilt_series):
        """Test that worker reports to monitor when provided."""
        mock_monitor = Mock()
        ready_flag = mp.Value('i', 0)
        stop_event = mp.Event()

        # Mock the TiltSeriesFetcher instance
        mock_fetcher_instance = mock_fetcher_class.return_value
        mock_fetcher_instance.return_value = mock_tilt_series

        # Run one update cycle
        update_count = 0

        def mock_create(*args, **kwargs):
            nonlocal update_count
            update_count += 1
            if update_count > 2:  # Initial fill (1) + one update
                stop_event.set()
            return [(torch.zeros(1), 1),
                     (torch.zeros(1), -1),
                     (torch.zeros(1), 1)]

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
                tilt_series_refresh_rate=10,
            )

        # Monitor should have been called for continuous updates (not initial fill)
        assert mock_monitor.record_production.call_count >= 1
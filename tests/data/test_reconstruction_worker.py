"""Tests for reconstruction worker module."""

import multiprocessing as mp
import pickle
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch

import einops
import pytest
import torch
import mrcfile

from warpylib import TiltSeries
from miss_alignment.data.io import TiltSeriesData
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
def mock_tilt_series_data(temp_dir):
    """Create mock TiltSeriesData with actual files."""
    # Create XML metadata
    xml_path = temp_dir / "test.xml"
    stack_path = temp_dir / "test.st"

    n_tilts = 10
    tilt_series = TiltSeries(n_tilts=n_tilts)
    tilt_series.angles = torch.linspace(-60, 60, n_tilts)
    tilt_series.save_meta(xml_path)

    # Create stack - needs to be larger than 2 * patch_size * subvolume_padding
    # With patch_size=32 and subvolume_padding=2.0, we need > 128 pixels
    images = torch.randn(n_tilts, 512, 512)
    with mrcfile.new(stack_path, overwrite=True) as mrc:
        mrc.set_data(images.numpy())

    # Create TiltSeriesData
    data = TiltSeriesData(
        xml_metadata_path=xml_path,
        stack_path=stack_path,
        stack_pixel_size=10.0,
        original_pixel_size=10.0,
        original_stack_shape=(512, 512),
        volume_shape=(512, 512, 256),
    )

    # Save to JSON
    json_path = temp_dir / "test.json"
    data.to_json(json_path)

    return json_path


@pytest.fixture
def shift_generator():
    """Create simple shift generator function."""

    def generator(n_tilts, device: str = "cpu"):
        return torch.randn(n_tilts, 3, device=device)

    return generator


class TestTiltSeriesFetcher:
    """Test TiltSeriesFetcher class."""

    def test_initialization(self, mock_tilt_series_data):
        """Test that TiltSeriesFetcher initializes correctly."""
        tilt_series_jsons = [mock_tilt_series_data]
        patch_size = 32
        refresh_rate = 5
        downsample = 1
        device = torch.device("cpu")

        fetcher = TiltSeriesFetcher(
            tilt_series_jsons=tilt_series_jsons,
            patch_size=patch_size,
            refresh_rate=refresh_rate,
            downsample=downsample,
            device=device,
        )

        assert fetcher.tilt_series_jsons == tilt_series_jsons
        assert fetcher.patch_size == patch_size
        assert fetcher.refresh_rate == refresh_rate
        assert fetcher.downsample == downsample
        assert fetcher.device == device
        assert fetcher._counter == 0
        assert fetcher._tilt_series is None

    def test_load_next(self, mock_tilt_series_data):
        """Test that _load_next loads and processes a tilt series correctly."""
        tilt_series_jsons = [mock_tilt_series_data]

        fetcher = TiltSeriesFetcher(
            tilt_series_jsons=tilt_series_jsons,
            patch_size=32,
            refresh_rate=5,
            downsample=1,
            device=torch.device("cpu"),
        )

        fetcher._load_next()

        assert fetcher._tilt_series is not None
        assert fetcher._images is not None
        assert fetcher._pixel_size is not None
        assert isinstance(fetcher._tilt_series, TiltSeries)
        assert isinstance(fetcher._images, torch.Tensor)
        assert isinstance(fetcher._pixel_size, float)

    def test_call_first_time(self, mock_tilt_series_data):
        """Test that __call__ loads a new tilt series on first call."""
        tilt_series_jsons = [mock_tilt_series_data]

        fetcher = TiltSeriesFetcher(
            tilt_series_jsons=tilt_series_jsons,
            patch_size=32,
            refresh_rate=5,
            downsample=1,
            device=torch.device("cpu"),
        )

        tilt_series, images, pixel_size = fetcher()

        assert isinstance(tilt_series, TiltSeries)
        assert isinstance(images, torch.Tensor)
        assert isinstance(pixel_size, float)
        assert fetcher._counter == 1

    def test_call_reuse(self, mock_tilt_series_data):
        """Test that __call__ reuses tilt series within refresh rate."""
        tilt_series_jsons = [mock_tilt_series_data]

        fetcher = TiltSeriesFetcher(
            tilt_series_jsons=tilt_series_jsons,
            patch_size=32,
            refresh_rate=5,
            downsample=1,
            device=torch.device("cpu"),
        )

        # First call loads new
        ts1, _, _ = fetcher()
        # Second call should reuse
        ts2, _, _ = fetcher()

        assert fetcher._counter == 2
        # Should be the same object
        assert ts1 is ts2

    def test_call_refresh(self, mock_tilt_series_data):
        """Test that __call__ refreshes after reaching refresh rate."""
        tilt_series_jsons = [mock_tilt_series_data]

        fetcher = TiltSeriesFetcher(
            tilt_series_jsons=tilt_series_jsons,
            patch_size=32,
            refresh_rate=2,
            downsample=1,
            device=torch.device("cpu"),
        )

        # First call loads new
        fetcher()
        # Second call reuses (counter = 2, which equals refresh_rate)
        fetcher()

        # Third call should refresh (counter >= refresh_rate)
        tilt_series, images, pixel_size = fetcher()

        assert isinstance(tilt_series, TiltSeries)
        assert fetcher._counter == 1

    def test_alignment_backup_restore(self, mock_tilt_series_data):
        """Test that alignment parameters are backed up and restored correctly."""
        tilt_series_jsons = [mock_tilt_series_data]

        fetcher = TiltSeriesFetcher(
            tilt_series_jsons=tilt_series_jsons,
            patch_size=32,
            refresh_rate=5,
            downsample=1,
            device=torch.device("cpu"),
        )

        # First call loads and backs up
        tilt_series1, _, _ = fetcher()
        original_angles = tilt_series1.angles.clone()
        original_tilt_axis_offset_x = tilt_series1.tilt_axis_offset_x.clone()

        # Modify alignment parameters
        tilt_series1.angles += 10.0
        tilt_series1.tilt_axis_offset_x += 5.0

        # Second call should restore original values
        tilt_series2, _, _ = fetcher()

        # Check that original values were restored
        torch.testing.assert_close(tilt_series2.angles, original_angles)
        torch.testing.assert_close(
            tilt_series2.tilt_axis_offset_x, original_tilt_axis_offset_x
        )


class TestGenerateTranslations:
    """Test translation generation functionality."""

    def test_raise_error_with_wrong_matrix_shape(self, shift_generator):
        matrix = torch.eye(3)
        rotation_matrices = einops.repeat(matrix, "h w -> n h w", n=10)

        with pytest.raises(ValueError):
            _ = _generate_translations(shift_generator, rotation_matrices)

    def test_basic_translation_generation(self, shift_generator):
        """Test that translations are generated with correct shape."""
        n_tilts = 10
        matrix = torch.eye(3)
        rotation_matrices = einops.repeat(matrix, "h w -> n h w", n=n_tilts)
        projection_matrices = rotation_matrices[..., 1:3, :3]

        with patch("random.random", return_value=0.6):
            translations = _generate_translations(shift_generator, projection_matrices)

        assert translations.shape == (n_tilts, 2)
        assert translations.dtype == torch.float32

    def test_translation_masking_y_axis(self, shift_generator):
        """Test that y-axis translations are zeroed when die_roll < 0.25."""
        matrix = torch.eye(3)
        rotation_matrices = einops.repeat(matrix, "h w -> n h w", n=5)
        projection_matrices = rotation_matrices[..., 1:3, :3]

        with patch("random.random", return_value=0.1):
            translations = _generate_translations(shift_generator, projection_matrices)

        assert torch.all(translations[:, 0] == 0.0)
        assert not torch.all(translations[:, 1] == 0.0)

    def test_translation_masking_x_axis(self, shift_generator):
        """Test that x-axis translations are zeroed when 0.25 <= die_roll < 0.5."""
        matrix = torch.eye(3)
        rotation_matrices = einops.repeat(matrix, "h w -> n h w", n=5)
        projection_matrices = rotation_matrices[..., 1:3, :3]

        with patch("random.random", return_value=0.3):
            translations = _generate_translations(shift_generator, projection_matrices)

        assert not torch.all(translations[:, 0] == 0.0)
        assert torch.all(translations[:, 1] == 0.0)


class TestCreatePoolReconstruction:
    """Test pool reconstruction creation."""

    def test_reconstruction_output_format(self, mock_tilt_series_data, shift_generator):
        """Test that reconstruction returns correct format."""
        # Load the tilt series data
        tilt_series_data = TiltSeriesData.from_json(mock_tilt_series_data)
        tilt_series, images, pixel_size = tilt_series_data.load_metadata_and_stack(
            downsample=1
        )

        result = _create_pool_reconstruction(
            tilt_series=tilt_series,
            images=images,
            pixel_size=pixel_size,
            patch_size=32,
            shift_generator=shift_generator,
            apply_ctf=False,
            device="cpu",
        )

        assert len(result) == 3
        assert all(isinstance(item, tuple) for item in result)
        assert all(len(item) == 2 for item in result)
        assert all(isinstance(item[0], torch.Tensor) for item in result)
        assert 1 in [item[1] for item in result]
        assert -1 in [item[1] for item in result]


class TestReconstructionWorker:
    """Test reconstruction worker process."""

    @patch("miss_alignment.data._reconstruction_worker.TiltSeriesFetcher")
    def test_worker_initial_fill(
        self, mock_fetcher_class, temp_dir, shift_generator, mock_tilt_series_data
    ):
        """Test that worker fills initial pool correctly."""
        assigned_indices = [0, 1, 2]
        ready_flag = mp.Value("i", 0)
        stop_event = mp.Event()

        # Create mock tilt series, images, and pixel size
        mock_tilt_series = Mock(spec=TiltSeries)
        mock_tilt_series.angles = torch.linspace(-60, 60, 10)
        mock_images = torch.randn(10, 128, 128)
        mock_pixel_size = 10.0

        # Mock the TiltSeriesFetcher instance
        mock_fetcher_instance = mock_fetcher_class.return_value
        mock_fetcher_instance.return_value = (
            mock_tilt_series,
            mock_images,
            mock_pixel_size,
        )

        # Mock the reconstruction creation
        mock_data = [
            (torch.randn(32, 32, 32), 1),
            (torch.randn(32, 32, 32), -1),
            (torch.randn(32, 32, 32), 1),
        ]

        with patch(
            "miss_alignment.data._reconstruction_worker._create_pool_reconstruction",
            return_value=mock_data,
        ):
            # Set stop event immediately to only do initial fill
            stop_event.set()

            reconstruction_worker(
                worker_id=0,
                assigned_indices=assigned_indices,
                pool_dir=temp_dir,
                tilt_series_jsons=[mock_tilt_series_data],
                patch_size=32,
                apply_ctf=False,
                downsample=1,
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
            with open(file_path, "rb") as f:
                data = pickle.load(f)
                assert len(data) == 3

    @patch("miss_alignment.data._reconstruction_worker.TiltSeriesFetcher")
    def test_worker_ready_flag(
        self, mock_fetcher_class, temp_dir, shift_generator, mock_tilt_series_data
    ):
        """Test that worker sets ready flag after initial fill."""
        ready_flag = mp.Value("i", 0)
        stop_event = mp.Event()
        stop_event.set()  # Stop immediately after initial fill

        # Create mock returns
        mock_tilt_series = Mock(spec=TiltSeries)
        mock_tilt_series.angles = torch.linspace(-60, 60, 10)
        mock_images = torch.randn(10, 128, 128)
        mock_pixel_size = 10.0

        # Mock the TiltSeriesFetcher instance
        mock_fetcher_instance = mock_fetcher_class.return_value
        mock_fetcher_instance.return_value = (
            mock_tilt_series,
            mock_images,
            mock_pixel_size,
        )

        with patch(
            "miss_alignment.data._reconstruction_worker._create_pool_reconstruction",
            return_value=[(torch.zeros(1), 1)] * 3,
        ):
            reconstruction_worker(
                worker_id=0,
                assigned_indices=[0],
                pool_dir=temp_dir,
                tilt_series_jsons=[mock_tilt_series_data],
                patch_size=32,
                apply_ctf=False,
                downsample=1,
                shift_generator=shift_generator,
                ready_flag=ready_flag,
                stop_event=stop_event,
                monitor=None,
                tilt_series_refresh_rate=10,
            )

        assert ready_flag.value == 1

    @patch("miss_alignment.data._reconstruction_worker.TiltSeriesFetcher")
    def test_worker_continuous_update(
        self, mock_fetcher_class, temp_dir, shift_generator, mock_tilt_series_data
    ):
        """Test that worker continuously updates pool when not stopped."""
        ready_flag = mp.Value("i", 0)
        stop_event = mp.Event()
        assigned_indices = [0, 1]

        # Create mock returns
        mock_tilt_series = Mock(spec=TiltSeries)
        mock_tilt_series.angles = torch.linspace(-60, 60, 10)
        mock_images = torch.randn(10, 128, 128)
        mock_pixel_size = 10.0

        # Mock the TiltSeriesFetcher instance
        mock_fetcher_instance = mock_fetcher_class.return_value
        mock_fetcher_instance.return_value = (
            mock_tilt_series,
            mock_images,
            mock_pixel_size,
        )

        update_count = 0
        max_updates = 3

        def mock_create(*args, **kwargs):
            nonlocal update_count
            update_count += 1
            # Stop after initial fill + max_updates
            if update_count > len(assigned_indices) + max_updates:
                stop_event.set()
            return [
                (torch.randn(32, 32, 32), 1),
                (torch.randn(32, 32, 32), -1),
                (torch.randn(32, 32, 32), 1),
            ]

        with patch(
            "miss_alignment.data._reconstruction_worker._create_pool_reconstruction",
            side_effect=mock_create,
        ):
            reconstruction_worker(
                worker_id=0,
                assigned_indices=assigned_indices,
                pool_dir=temp_dir,
                tilt_series_jsons=[mock_tilt_series_data],
                patch_size=32,
                apply_ctf=False,
                downsample=1,
                shift_generator=shift_generator,
                ready_flag=ready_flag,
                stop_event=stop_event,
                monitor=None,
                tilt_series_refresh_rate=10,
            )

        # Should have done initial fill + continuous updates
        assert update_count > len(assigned_indices)

    @patch("miss_alignment.data._reconstruction_worker.TiltSeriesFetcher")
    def test_worker_with_monitor(
        self, mock_fetcher_class, temp_dir, shift_generator, mock_tilt_series_data
    ):
        """Test that worker reports to monitor when provided."""
        mock_monitor = Mock()
        ready_flag = mp.Value("i", 0)
        stop_event = mp.Event()

        # Create mock returns
        mock_tilt_series = Mock(spec=TiltSeries)
        mock_tilt_series.angles = torch.linspace(-60, 60, 10)
        mock_images = torch.randn(10, 128, 128)
        mock_pixel_size = 10.0

        # Mock the TiltSeriesFetcher instance
        mock_fetcher_instance = mock_fetcher_class.return_value
        mock_fetcher_instance.return_value = (
            mock_tilt_series,
            mock_images,
            mock_pixel_size,
        )

        # Run one update cycle
        update_count = 0

        def mock_create(*args, **kwargs):
            nonlocal update_count
            update_count += 1
            if update_count > 2:  # Initial fill (1) + one update
                stop_event.set()
            return [(torch.zeros(1), 1), (torch.zeros(1), -1), (torch.zeros(1), 1)]

        with patch(
            "miss_alignment.data._reconstruction_worker._create_pool_reconstruction",
            side_effect=mock_create,
        ):
            reconstruction_worker(
                worker_id=0,
                assigned_indices=[0],
                pool_dir=temp_dir,
                tilt_series_jsons=[mock_tilt_series_data],
                patch_size=32,
                apply_ctf=False,
                downsample=1,
                shift_generator=shift_generator,
                ready_flag=ready_flag,
                stop_event=stop_event,
                monitor=mock_monitor,
                tilt_series_refresh_rate=10,
            )

        # Monitor should have been called for continuous updates (not initial fill)
        assert mock_monitor.record_production.call_count >= 1

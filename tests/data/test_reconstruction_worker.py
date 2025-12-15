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
    _count_partition_files,
    reconstruction_worker,
    TiltSeriesFetcher,
    sample_positions,
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

    n_tilts = 10
    stack_pixel_size = 10.0
    original_pixel_size = 10.0
    original_stack_shape = (512, 512)
    volume_shape = (512, 512, 256)

    # Create TiltSeries object with proper metadata
    tilt_series = TiltSeries(path=xml_path, n_tilts=n_tilts)
    tilt_series.angles = torch.linspace(-60, 60, n_tilts)
    tilt_series.tilt_axis_angles = torch.zeros(n_tilts)
    tilt_series.tilt_axis_offset_x = torch.zeros(n_tilts)
    tilt_series.tilt_axis_offset_y = torch.zeros(n_tilts)

    # Set physical dimensions
    tilt_series.image_dimensions_physical = torch.tensor(
        [
            original_stack_shape[0] * original_pixel_size,
            original_stack_shape[1] * original_pixel_size,
        ],
        dtype=torch.float32,
    )
    tilt_series.volume_dimensions_physical = torch.tensor(
        [
            volume_shape[0] * stack_pixel_size,
            volume_shape[1] * stack_pixel_size,
            volume_shape[2] * stack_pixel_size,
        ],
        dtype=torch.float32,
    )

    # Get the expected stack path from warpylib
    stack_path = Path(tilt_series.tilt_stack_path)
    # Create parent directories if they don't exist
    stack_path.parent.mkdir(parents=True, exist_ok=True)

    # Create stack - needs to be larger than 2 * patch_size * subvolume_padding
    # With patch_size=32 and subvolume_padding=2.0, we need > 128 pixels
    images = torch.randn(n_tilts, 512, 512)
    with mrcfile.new(stack_path, overwrite=True) as mrc:
        mrc.set_data(images.numpy())
        mrc.voxel_size = stack_pixel_size

    # Save metadata to XML
    tilt_series.save_meta(xml_path)

    # Return the xml_path directly (TiltSeriesData only needs xml_metadata_path)
    return xml_path


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
        tilt_series_xmls = [mock_tilt_series_data]
        patch_size = 32
        refresh_rate = 5
        downsample = 1
        device = torch.device("cpu")

        fetcher = TiltSeriesFetcher(
            tilt_series_xmls=tilt_series_xmls,
            patch_size=patch_size,
            refresh_rate=refresh_rate,
            downsample=downsample,
            device=device,
        )

        assert fetcher.tilt_series_xmls == tilt_series_xmls
        assert fetcher.patch_size == patch_size
        assert fetcher.refresh_rate == refresh_rate
        assert fetcher.downsample == downsample
        assert fetcher.device == device
        assert fetcher._counter == 0
        assert fetcher._tilt_series is None

    def test_load_next(self, mock_tilt_series_data):
        """Test that _load_next loads and processes a tilt series correctly."""
        tilt_series_xmls = [mock_tilt_series_data]

        fetcher = TiltSeriesFetcher(
            tilt_series_xmls=tilt_series_xmls,
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
        tilt_series_xmls = [mock_tilt_series_data]

        fetcher = TiltSeriesFetcher(
            tilt_series_xmls=tilt_series_xmls,
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
        tilt_series_xmls = [mock_tilt_series_data]

        fetcher = TiltSeriesFetcher(
            tilt_series_xmls=tilt_series_xmls,
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
        tilt_series_xmls = [mock_tilt_series_data]

        fetcher = TiltSeriesFetcher(
            tilt_series_xmls=tilt_series_xmls,
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
        tilt_series_xmls = [mock_tilt_series_data]

        fetcher = TiltSeriesFetcher(
            tilt_series_xmls=tilt_series_xmls,
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


class TestSamplePositions:
    """Test position sampling functionality."""

    def test_output_shape(self):
        """Test that output has correct shape."""
        n_particles = 5
        volume_dimensions_angstrom = torch.tensor([1000.0, 1000.0, 500.0])
        patch_size_angstrom = 200.0

        positions = sample_positions(
            n_particles=n_particles,
            volume_dimensions_angstrom=volume_dimensions_angstrom,
            patch_size_angstrom=patch_size_angstrom,
        )

        assert positions.shape == (n_particles, 3)
        assert positions.dtype == torch.float32

    def test_all_dimensions_larger_than_patch(self):
        """Test normal case where all dimensions are larger than patch size."""
        n_particles = 10
        volume_dimensions_angstrom = torch.tensor([1000.0, 1000.0, 500.0])
        patch_size_angstrom = 200.0

        positions = sample_positions(
            n_particles=n_particles,
            volume_dimensions_angstrom=volume_dimensions_angstrom,
            patch_size_angstrom=patch_size_angstrom,
        )

        # Check positions are within valid range for each dimension
        patch_offset = patch_size_angstrom / 2
        for dim in range(3):
            dim_size = volume_dimensions_angstrom[dim].item()
            assert torch.all(positions[:, dim] >= patch_offset)
            assert torch.all(positions[:, dim] <= dim_size - patch_offset)

    def test_all_dimensions_smaller_than_patch(self):
        """Test case where all dimensions are smaller than patch size."""
        n_particles = 10
        volume_dimensions_angstrom = torch.tensor([100.0, 150.0, 80.0])
        patch_size_angstrom = 200.0

        positions = sample_positions(
            n_particles=n_particles,
            volume_dimensions_angstrom=volume_dimensions_angstrom,
            patch_size_angstrom=patch_size_angstrom,
        )

        # All positions should be centered at half the dimension size
        for dim in range(3):
            expected = volume_dimensions_angstrom[dim] / 2
            assert torch.all(positions[:, dim] == expected)

    def test_one_dimension_smaller_than_patch(self):
        """Test case where one dimension is smaller than patch size."""
        n_particles = 10
        volume_dimensions_angstrom = torch.tensor([1000.0, 1000.0, 100.0])
        patch_size_angstrom = 200.0

        positions = sample_positions(
            n_particles=n_particles,
            volume_dimensions_angstrom=volume_dimensions_angstrom,
            patch_size_angstrom=patch_size_angstrom,
        )

        # Z dimension (index 2) should be centered
        expected_z = volume_dimensions_angstrom[2] / 2
        assert torch.all(positions[:, 2] == expected_z)

        # X and Y dimensions should be in valid range
        patch_offset = patch_size_angstrom / 2
        for dim in [0, 1]:
            dim_size = volume_dimensions_angstrom[dim].item()
            assert torch.all(positions[:, dim] >= patch_offset)
            assert torch.all(positions[:, dim] <= dim_size - patch_offset)

    def test_dimension_equals_patch_size(self):
        """Test edge case where dimension equals patch size."""
        n_particles = 10
        volume_dimensions_angstrom = torch.tensor([1000.0, 200.0, 500.0])
        patch_size_angstrom = 200.0

        positions = sample_positions(
            n_particles=n_particles,
            volume_dimensions_angstrom=volume_dimensions_angstrom,
            patch_size_angstrom=patch_size_angstrom,
        )

        # Y dimension equals patch size, so range is zero
        # All positions should be at patch_offset (which equals dim_size/2)
        patch_offset = patch_size_angstrom / 2
        assert torch.all(positions[:, 1] == patch_offset)

    def test_realistic_cryo_et_parameters(self):
        """Test with realistic cryo-ET parameters."""
        # Typical tomogram: 4096 x 4096 x 1024 Angstroms
        # Patch size: 640 Angstroms (64 pixels * 10 A/pixel)
        n_particles = 2
        volume_dimensions_angstrom = torch.tensor([4096.0, 4096.0, 1024.0])
        patch_size_angstrom = 640.0

        positions = sample_positions(
            n_particles=n_particles,
            volume_dimensions_angstrom=volume_dimensions_angstrom,
            patch_size_angstrom=patch_size_angstrom,
        )

        # Check all positions are within valid range
        patch_offset = patch_size_angstrom / 2
        for dim in range(3):
            dim_size = volume_dimensions_angstrom[dim].item()
            assert torch.all(positions[:, dim] >= patch_offset)
            assert torch.all(positions[:, dim] <= dim_size - patch_offset)


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


class TestCountPartitionFiles:
    """Test partition file counting."""

    def test_count_partition_files(self, temp_dir):
        """Test counting files in a partition."""
        # Create some partition files
        for i in range(5):
            (temp_dir / f"partition_0_seq_{i}.pickle").touch()

        # Create files for another partition
        for i in range(3):
            (temp_dir / f"partition_1_seq_{i}.pickle").touch()

        # Create a temp file (starts with tmp_ so won't match pattern)
        (temp_dir / "tmp_partition_0_xyz.pickle").touch()

        assert _count_partition_files(temp_dir, 0) == 5
        assert _count_partition_files(temp_dir, 1) == 3
        assert _count_partition_files(temp_dir, 2) == 0


class TestCreatePoolReconstruction:
    """Test pool reconstruction creation."""

    def test_reconstruction_output_format(self, mock_tilt_series_data, shift_generator):
        """Test that reconstruction returns 4 triplets with correct format."""
        # Load the tilt series data
        tilt_series_data = TiltSeriesData(xml_metadata_path=mock_tilt_series_data)
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

        # Should return 4 triplets (2 particles × 2 mirror combinations each)
        assert len(result) == 4

        for triplet in result:
            # Each triplet should have 3 examples
            assert len(triplet) == 3
            assert all(isinstance(item, tuple) for item in triplet)
            assert all(len(item) == 2 for item in triplet)
            assert all(isinstance(item[0], torch.Tensor) for item in triplet)

            # Each triplet must contain both positive and negative labels
            labels = [item[1] for item in triplet]
            assert 1 in labels
            assert -1 in labels

    def test_mirror_combinations_used(self, mock_tilt_series_data, shift_generator):
        """Test that 4 triplets are generated (2 particles × 2 mirror combinations)."""
        tilt_series_data = TiltSeriesData(xml_metadata_path=mock_tilt_series_data)
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

        # Check we get 4 triplets (2 particles × 2 sampled mirror combinations)
        assert len(result) == 4


class TestReconstructionWorker:
    """Test reconstruction worker process."""

    @patch("miss_alignment.data._reconstruction_worker.TiltSeriesFetcher")
    def test_worker_writes_partition_files(
        self, mock_fetcher_class, temp_dir, shift_generator, mock_tilt_series_data
    ):
        """Test that worker writes files with correct partition naming."""
        partition_id = 0
        partition_size = 10
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

        # Mock triplets (4 triplets per call, each with 3 examples)
        mock_triplets = [
            [
                (torch.randn(32, 32, 32), 1),
                (torch.randn(32, 32, 32), -1),
                (torch.randn(32, 32, 32), 1),
            ]
            for _ in range(4)
        ]

        call_count = 0

        def mock_create(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            # Stop after writing one batch of 8 files (2 calls × 4 triplets)
            if call_count > 1:
                stop_event.set()
            return mock_triplets

        with patch(
            "miss_alignment.data._reconstruction_worker._create_pool_reconstruction",
            side_effect=mock_create,
        ):
            reconstruction_worker(
                partition_id=partition_id,
                partition_size=partition_size,
                pool_dir=temp_dir,
                tilt_series_xmls=[mock_tilt_series_data],
                patch_size=32,
                apply_ctf=False,
                downsample=1,
                shift_generator=shift_generator,
                stop_event=stop_event,
                tilt_series_refresh_rate=10,
            )

        # Check that files were created with correct naming
        files = list(temp_dir.glob(f"partition_{partition_id}_seq_*.pickle"))
        assert len(files) == 8  # 2 calls × 4 triplets

        # Verify file contents
        for file_path in files:
            with open(file_path, "rb") as f:
                data = pickle.load(f)
                assert len(data) == 3  # Triplet
                # Should be fp16
                assert data[0][0].dtype == torch.float16

    @patch("miss_alignment.data._reconstruction_worker.TiltSeriesFetcher")
    def test_worker_pauses_when_partition_full(
        self, mock_fetcher_class, temp_dir, shift_generator, mock_tilt_series_data
    ):
        """Test that worker pauses when partition is full."""
        partition_id = 0
        partition_size = 5  # Small partition
        stop_event = mp.Event()

        # Pre-fill the partition
        for i in range(partition_size):
            (temp_dir / f"partition_{partition_id}_seq_{i}.pickle").touch()

        # Create mock returns
        mock_tilt_series = Mock(spec=TiltSeries)
        mock_tilt_series.angles = torch.linspace(-60, 60, 10)
        mock_images = torch.randn(10, 128, 128)
        mock_pixel_size = 10.0

        mock_fetcher_instance = mock_fetcher_class.return_value
        mock_fetcher_instance.return_value = (
            mock_tilt_series,
            mock_images,
            mock_pixel_size,
        )

        # Track how many times _create_pool_reconstruction is called
        create_call_count = 0

        def mock_create(*args, **kwargs):
            nonlocal create_call_count
            create_call_count += 1
            return [
                [(torch.zeros(1), 1), (torch.zeros(1), -1), (torch.zeros(1), 1)]
            ] * 8

        # Stop after a short time to avoid infinite loop
        import threading
        import time

        def stop_after_delay():
            time.sleep(0.2)
            stop_event.set()

        stop_thread = threading.Thread(target=stop_after_delay)
        stop_thread.start()

        with patch(
            "miss_alignment.data._reconstruction_worker._create_pool_reconstruction",
            side_effect=mock_create,
        ):
            reconstruction_worker(
                partition_id=partition_id,
                partition_size=partition_size,
                pool_dir=temp_dir,
                tilt_series_xmls=[mock_tilt_series_data],
                patch_size=32,
                apply_ctf=False,
                downsample=1,
                shift_generator=shift_generator,
                stop_event=stop_event,
                tilt_series_refresh_rate=10,
            )

        stop_thread.join()

        # Worker should have been paused (no reconstruction created)
        # because partition was already full
        assert create_call_count == 0

    @patch("miss_alignment.data._reconstruction_worker.TiltSeriesFetcher")
    def test_worker_sequential_ids_increment(
        self, mock_fetcher_class, temp_dir, shift_generator, mock_tilt_series_data
    ):
        """Test that sequential IDs increment correctly."""
        partition_id = 0
        partition_size = 100
        stop_event = mp.Event()

        mock_tilt_series = Mock(spec=TiltSeries)
        mock_tilt_series.angles = torch.linspace(-60, 60, 10)
        mock_images = torch.randn(10, 128, 128)
        mock_pixel_size = 10.0

        mock_fetcher_instance = mock_fetcher_class.return_value
        mock_fetcher_instance.return_value = (
            mock_tilt_series,
            mock_images,
            mock_pixel_size,
        )

        call_count = 0

        def mock_create(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count > 2:  # Create 2 batches (4 calls × 4 triplets = 16 files)
                stop_event.set()
            return [
                [(torch.zeros(1), 1), (torch.zeros(1), -1), (torch.zeros(1), 1)]
            ] * 4

        with patch(
            "miss_alignment.data._reconstruction_worker._create_pool_reconstruction",
            side_effect=mock_create,
        ):
            reconstruction_worker(
                partition_id=partition_id,
                partition_size=partition_size,
                pool_dir=temp_dir,
                tilt_series_xmls=[mock_tilt_series_data],
                patch_size=32,
                apply_ctf=False,
                downsample=1,
                shift_generator=shift_generator,
                stop_event=stop_event,
                tilt_series_refresh_rate=10,
            )

        # Check sequential IDs
        files = sorted(temp_dir.glob(f"partition_{partition_id}_seq_*.pickle"))
        assert len(files) == 16  # 4 calls × 4 triplets

        # Extract IDs and verify they're sequential
        ids = []
        for f in files:
            # Extract ID from "partition_0_seq_X.pickle"
            id_str = f.stem.split("_")[-1]
            ids.append(int(id_str))

        ids.sort()
        assert ids == list(range(16))

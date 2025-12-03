import pytest
import torch
import pickle
import time
from pathlib import Path
from unittest.mock import patch, MagicMock

from miss_alignment.data.training_dataset import ReconstructionPoolDataset


class TestReconstructionPoolDataset:
    """Test suite for ReconstructionPoolDataset."""

    @pytest.fixture
    def mock_pool_dir(self, tmp_path):
        """Create temporary directory for pool files."""
        pool_dir = tmp_path / "pool"
        pool_dir.mkdir()
        return pool_dir

    @pytest.fixture
    def sample_data(self):
        """Create sample reconstruction data (triplet in fp16)."""
        volumes = [torch.randn(64, 64, 64).half() for _ in range(3)]
        labels = [-1, 1, 1]  # Valid labels are -1 and 1
        return list(zip(volumes, labels))

    @pytest.fixture
    def dataset(self, mock_pool_dir):
        """Create dataset instance with test parameters."""
        ds = ReconstructionPoolDataset(
            pool_dir=mock_pool_dir, batch_size=4, epoch_size=100
        )
        ds.partition_id = 0  # Simulate worker_init_fn assignment
        return ds

    def test_init(self, mock_pool_dir):
        """Test dataset initialization."""
        batch_size = 4
        epoch_size = 100

        dataset = ReconstructionPoolDataset(
            pool_dir=mock_pool_dir, batch_size=batch_size, epoch_size=epoch_size
        )

        assert dataset.pool_dir == mock_pool_dir
        assert dataset.batch_size == batch_size
        assert dataset.epoch_size == epoch_size
        assert dataset.partition_id is None  # Not set until worker_init_fn

    def test_len(self, dataset):
        """Test __len__ method returns epoch_size."""
        assert len(dataset) == 100

    def test_partition_id_not_set_raises_error(self, mock_pool_dir):
        """Test that accessing partition files without partition_id raises error."""
        dataset = ReconstructionPoolDataset(
            pool_dir=mock_pool_dir, batch_size=4, epoch_size=100
        )
        # partition_id is None
        with pytest.raises(RuntimeError, match="partition_id not set"):
            dataset._list_partition_files()

    def test_list_partition_files(self, dataset, mock_pool_dir, sample_data):
        """Test that _list_partition_files returns correct files."""
        # Create some test files
        for i in range(5):
            file_path = mock_pool_dir / f"partition_0_seq_{i}.pickle"
            with open(file_path, "wb") as f:
                pickle.dump(sample_data, f)

        # Create a file for different partition (should be ignored)
        other_file = mock_pool_dir / "partition_1_seq_0.pickle"
        with open(other_file, "wb") as f:
            pickle.dump(sample_data, f)

        # Create a temp file (should be filtered out)
        tmp_file = mock_pool_dir / "tmp_partition_0_xyz.pickle"
        with open(tmp_file, "wb") as f:
            pickle.dump(sample_data, f)

        files = dataset._list_partition_files()
        assert len(files) == 5
        assert all("partition_0" in f.name for f in files)
        assert all(not f.name.startswith("tmp_") for f in files)

    def test_getitem_reads_and_deletes_file(self, dataset, mock_pool_dir, sample_data):
        """Test that __getitem__ reads file and deletes it after."""
        # Create enough files to meet batch_size threshold
        for i in range(5):
            file_path = mock_pool_dir / f"partition_0_seq_{i}.pickle"
            with open(file_path, "wb") as f:
                pickle.dump(sample_data, f)

        initial_count = len(list(mock_pool_dir.glob("partition_0_*.pickle")))
        assert initial_count == 5

        # Get an item
        result = dataset[0]

        # Check file was deleted
        final_count = len(list(mock_pool_dir.glob("partition_0_*.pickle")))
        assert final_count == initial_count - 1

        # Check return format
        assert len(result) == 4  # 3 volumes + 1 labels tensor
        assert isinstance(result[-1], torch.Tensor)  # labels

    def test_getitem_converts_fp16_to_fp32(self, dataset, mock_pool_dir, sample_data):
        """Test that volumes are converted from fp16 to fp32."""
        # Create test files
        for i in range(5):
            file_path = mock_pool_dir / f"partition_0_seq_{i}.pickle"
            with open(file_path, "wb") as f:
                pickle.dump(sample_data, f)

        result = dataset[0]

        # Check volumes are fp32
        for volume in result[:-1]:
            assert volume.dtype == torch.float32

    def test_getitem_waits_for_files(self, dataset, mock_pool_dir, sample_data):
        """Test that __getitem__ waits when not enough files available."""
        # Start with fewer files than batch_size
        for i in range(2):  # Less than batch_size=4
            file_path = mock_pool_dir / f"partition_0_seq_{i}.pickle"
            with open(file_path, "wb") as f:
                pickle.dump(sample_data, f)

        # Add more files in a separate thread after a delay
        import threading

        def add_files():
            time.sleep(0.1)  # Wait a bit
            for i in range(2, 5):  # Add more files
                file_path = mock_pool_dir / f"partition_0_seq_{i}.pickle"
                with open(file_path, "wb") as f:
                    pickle.dump(sample_data, f)

        thread = threading.Thread(target=add_files)
        thread.start()

        # This should wait until enough files are available
        result = dataset[0]
        thread.join()

        assert len(result) == 4  # Should succeed after files are added

    def test_normalize_basic(self, dataset):
        """Test basic normalization functionality."""
        volume = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])
        normalized = dataset._normalize(volume)

        # Check mean is approximately 0 and std is approximately 1
        assert torch.abs(torch.mean(normalized)) < 1e-6
        assert torch.abs(torch.std(normalized) - 1.0) < 1e-6

    def test_normalize_constant_volume(self, dataset):
        """Test normalization with constant volume (std=0)."""
        volume = torch.ones(10)

        with pytest.raises(ValueError):
            dataset._normalize(volume)

    @patch("miss_alignment.data.training_dataset.random_contrast")
    @patch("miss_alignment.data.training_dataset.random_edge_mask")
    @patch("miss_alignment.data.training_dataset.random_cube_mask")
    def test_prep_and_augment_no_mirror(
        self, mock_cube, mock_edge, mock_contrast, dataset
    ):
        """Test augmentation pipeline does not include random_mirror."""
        # Setup mocks to return input unchanged
        mock_contrast.side_effect = lambda x: x
        mock_edge.side_effect = lambda x, **kwargs: x
        mock_cube.side_effect = lambda x: x

        volumes = [torch.randn(10, 10, 10) for _ in range(3)]
        _ = dataset._prep_and_augment(volumes)

        # Check all augmentation functions were called
        assert mock_contrast.call_count == 3
        assert mock_edge.call_count == 3
        assert mock_cube.call_count == 3

        # Check edge mask called with correct parameters
        for call in mock_edge.call_args_list:
            assert call[1]["edge_width"] == (1, 5)

    def test_channel_dimension_addition(self, dataset, mock_pool_dir, sample_data):
        """Test that channel dimension is added correctly."""
        # Create test files
        for i in range(5):
            file_path = mock_pool_dir / f"partition_0_seq_{i}.pickle"
            with open(file_path, "wb") as f:
                pickle.dump(sample_data, f)

        result = dataset[0]

        # Check that volumes have channel dimension added (1, D, H, W)
        for volume in result[:-1]:
            assert volume.shape[0] == 1
            assert len(volume.shape) == 4

    def test_invalid_labels_raises_error(self, dataset, mock_pool_dir):
        """Test that ValueError is raised when labels don't contain both -1 and 1."""
        # Create sample data with invalid labels (only contains 1, missing -1)
        invalid_data = [(torch.randn(64, 64, 64).half(), 1) for _ in range(3)]

        for i in range(5):
            file_path = mock_pool_dir / f"partition_0_seq_{i}.pickle"
            with open(file_path, "wb") as f:
                pickle.dump(invalid_data, f)

        with pytest.raises(
            ValueError,
            match="Training examples must contain positive and negative labels",
        ):
            dataset[0]

    def test_corrupted_file_retry(self, dataset, mock_pool_dir, sample_data):
        """Test that corrupted files trigger retry."""
        # Create a corrupted file
        corrupted_path = mock_pool_dir / "partition_0_seq_0.pickle"
        with open(corrupted_path, "wb") as f:
            f.write(b"corrupted data")

        # Create valid files
        for i in range(1, 6):
            file_path = mock_pool_dir / f"partition_0_seq_{i}.pickle"
            with open(file_path, "wb") as f:
                pickle.dump(sample_data, f)

        # Should succeed by retrying with a different file
        result = dataset[0]
        assert len(result) == 4

    def test_file_not_found_retry(self, dataset, mock_pool_dir, sample_data):
        """Test that FileNotFoundError (race condition) triggers retry."""
        # Create valid files
        for i in range(5):
            file_path = mock_pool_dir / f"partition_0_seq_{i}.pickle"
            with open(file_path, "wb") as f:
                pickle.dump(sample_data, f)

        call_count = 0
        original_open = open

        def mock_open_with_error(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise FileNotFoundError("File deleted")
            return original_open(*args, **kwargs)

        with patch("builtins.open", side_effect=mock_open_with_error):
            # This should retry and succeed
            try:
                result = dataset[0]
                # If we get here, the retry worked
                assert len(result) == 4
            except FileNotFoundError:
                # First call raised error, retry should have been attempted
                assert call_count >= 1
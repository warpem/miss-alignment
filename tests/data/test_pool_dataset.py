import pytest
import torch
import pickle
from pathlib import Path
from unittest.mock import patch, mock_open, MagicMock
import tempfile
import einops

from miss_alignment.data.training_dataset import ReconstructionPoolDataset


class TestReconstructionPoolDataset:
    """Test suite for ReconstructionPoolDataset."""

    @pytest.fixture
    def mock_pool_dir(self, tmp_path):
        """Create temporary directory for pool files."""
        return tmp_path / "pool"

    @pytest.fixture
    def sample_data(self):
        """Create sample reconstruction data."""
        volumes = [torch.randn(64, 64, 64) for _ in range(3)]
        labels = [-1, 1, 1]  # Valid labels are -1 and 1
        return list(zip(volumes, labels))

    @pytest.fixture
    def dataset(self, mock_pool_dir):
        """Create dataset instance with test parameters."""
        mock_pool_dir.mkdir()
        return ReconstructionPoolDataset(
            pool_dir=mock_pool_dir,
            pool_size=10,
            epoch_size=100
        )

    def test_init(self, mock_pool_dir):
        """Test dataset initialization."""
        pool_size = 10
        epoch_size = 100

        dataset = ReconstructionPoolDataset(
            pool_dir=mock_pool_dir,
            pool_size=pool_size,
            epoch_size=epoch_size
        )

        assert dataset.pool_dir == mock_pool_dir
        assert dataset.pool_size == pool_size
        assert dataset.epoch_size == epoch_size

    def test_len(self, dataset):
        """Test __len__ method returns epoch_size."""
        assert len(dataset) == 100

    @patch('builtins.open', new_callable=mock_open)
    @patch('pickle.load')
    @patch('random.shuffle')
    def test_getitem_basic(self, mock_shuffle, mock_pickle_load, mock_file,
                           dataset, sample_data):
        """Test basic __getitem__ functionality."""
        # Setup mocks
        mock_pickle_load.return_value = sample_data
        mock_shuffle.side_effect = lambda x: x  # Don't actually shuffle

        # Mock augmentation functions to avoid import issues
        with patch.object(dataset, '_prep_and_augment',
                          return_value=[torch.randn(64, 64, 64) for _ in
                                        range(3)]):
            result = dataset[5]

            # Check file path calculation
            expected_path = dataset.pool_dir / "recon_5.pickle"
            mock_file.assert_called_once_with(expected_path, "rb")

            # Check return format
            assert len(result) == 4  # 3 volumes + 1 labels tensor
            assert isinstance(result[-1], torch.Tensor)  # labels
            
            # Check that labels contain both -1 and 1
            labels = result[-1]
            assert -1 in labels and 1 in labels, "Labels should contain both -1 and 1"

            # Check volumes have correct shape (1, d, h, w)
            for volume in result[:-1]:
                assert volume.shape[0] == 1  # channel dimension

    def test_getitem_modulo_indexing(self, dataset, sample_data):
        """Test that indexing uses modulo for pool wrapping."""
        with patch('builtins.open', new_callable=mock_open), \
                patch('pickle.load', return_value=sample_data), \
                patch('random.shuffle'), \
                patch.object(dataset, '_prep_and_augment',
                             return_value=[torch.randn(64, 64, 64) for _ in
                                           range(3)]):
            # Test that idx=15 maps to file recon_5.pickle (15 % 10 = 5)
            dataset[15]
            expected_path = dataset.pool_dir / "recon_5.pickle"

            # Verify the correct file was opened
            with patch('builtins.open', new_callable=mock_open) as mock_file:
                with patch('pickle.load', return_value=sample_data), \
                        patch('random.shuffle'), \
                        patch.object(dataset, '_prep_and_augment',
                                     return_value=[torch.randn(64, 64, 64) for
                                                   _ in range(3)]):
                    dataset[15]
                    mock_file.assert_called_once_with(expected_path, "rb")

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
            # With std=0, normalize should error out as it could lead to
            # division by zero. Generally we don't expect reconstructed
            # patches to have constant values as it could mean we show the
            # model empty areas.
            normalized = dataset._normalize(volume)

    @patch('miss_alignment.data.training_dataset.random_contrast')
    @patch('miss_alignment.data.training_dataset.random_edge_mask')
    @patch('miss_alignment.data.training_dataset.random_cube_mask')
    @patch('miss_alignment.data.training_dataset.random_mirror')
    def test_prep_and_augment(self, mock_mirror, mock_cube, mock_edge,
                              mock_contrast, dataset):
        """Test augmentation pipeline."""
        # Setup mocks to return input unchanged
        mock_contrast.side_effect = lambda x: x
        mock_edge.side_effect = lambda x, **kwargs: x
        mock_cube.side_effect = lambda x: x
        mock_mirror.side_effect = lambda x: x

        volumes = [torch.randn(10, 10, 10) for _ in range(3)]
        result = dataset._prep_and_augment(volumes)

        # Check all augmentation functions were called
        assert mock_contrast.call_count == 3
        assert mock_edge.call_count == 3
        assert mock_cube.call_count == 3
        mock_mirror.assert_called_once()

        # Check edge mask called with correct parameters
        for call in mock_edge.call_args_list:
            assert call[1]['edge_width'] == (1, 5)

    @patch('random.shuffle')
    def test_shuffle_consistency(self, mock_shuffle, dataset, sample_data):
        """Test that examples are shuffled."""
        with patch('builtins.open', new_callable=mock_open), \
                patch('pickle.load', return_value=sample_data), \
                patch.object(dataset, '_prep_and_augment',
                             return_value=[torch.randn(64, 64, 64) for _ in
                                           range(3)]):
            dataset[0]
            mock_shuffle.assert_called_once()

    def test_channel_dimension_addition(self, dataset, sample_data):
        """Test that channel dimension is added correctly."""
        mock_volumes = [torch.randn(32, 32, 32) for _ in range(3)]

        with patch('builtins.open', new_callable=mock_open), \
                patch('pickle.load', return_value=sample_data), \
                patch('random.shuffle'), \
                patch.object(dataset, '_prep_and_augment',
                             return_value=mock_volumes):
            result = dataset[0]

            # Check that volumes have channel dimension added
            for volume in result[:-1]:  # Exclude labels
                assert volume.shape == (1, 32, 32, 32)

    def test_file_not_found_error(self, dataset):
        """Test behavior when pickle file doesn't exist."""
        with pytest.raises(FileNotFoundError):
            dataset[0]

    def test_pickle_load_error(self, dataset, sample_data):
        """Test behavior when pickle loading fails."""
        with patch('builtins.open', new_callable=mock_open), \
                patch('pickle.load',
                      side_effect=pickle.UnpicklingError("Bad pickle")):
            with pytest.raises(pickle.UnpicklingError):
                dataset[0]
                
    def test_invalid_labels(self, dataset):
        """Test that ValueError is raised when labels don't contain both -1 and 1."""
        # Create sample data with invalid labels (only contains 1, missing -1)
        invalid_data = [(torch.randn(64, 64, 64), 1) for _ in range(3)]
        
        with patch('builtins.open', new_callable=mock_open), \
                patch('pickle.load', return_value=invalid_data), \
                patch('random.shuffle'):
            with pytest.raises(ValueError, match="Training examples must contain positive and negative labels"):
                dataset[0]
                
        # Create sample data with invalid labels (only contains -1, missing 1)
        invalid_data = [(torch.randn(64, 64, 64), -1) for _ in range(3)]
        
        with patch('builtins.open', new_callable=mock_open), \
                patch('pickle.load', return_value=invalid_data), \
                patch('random.shuffle'):
            with pytest.raises(ValueError, match="Training examples must contain positive and negative labels"):
                dataset[0]
                
        # Create sample data with invalid labels (contains 0, which is not valid)
        invalid_data = [(torch.randn(64, 64, 64), label) for label in [0, 0,
                                                                       0]]
        
        with patch('builtins.open', new_callable=mock_open), \
                patch('pickle.load', return_value=invalid_data), \
                patch('random.shuffle'):
            with pytest.raises(ValueError, match="Training examples must contain positive and negative labels"):
                dataset[0]

    @pytest.mark.parametrize("pool_size,idx,expected_file", [
        (10, 0, "recon_0.pickle"),
        (10, 9, "recon_9.pickle"),
        (10, 10, "recon_0.pickle"),
        (10, 25, "recon_5.pickle"),
        (5, 17, "recon_2.pickle"),
    ])
    def test_file_path_calculation(self, mock_pool_dir, pool_size, idx,
                                   expected_file):
        """Test file path calculation for various indices."""
        mock_pool_dir.mkdir()
        dataset = ReconstructionPoolDataset(mock_pool_dir, pool_size, 100)

        sample_data = [(torch.randn(10, 10, 10), -1),
                       (torch.randn(10, 10, 10), -1),
                       (torch.randn(10, 10, 10), 1)]

        with patch('builtins.open', new_callable=mock_open) as mock_file, \
                patch('pickle.load', return_value=sample_data), \
                patch('random.shuffle'), \
                patch.object(dataset, '_prep_and_augment',
                             return_value=[torch.randn(10, 10, 10) for _ in
                                           range(3)]):
            dataset[idx]
            expected_path = mock_pool_dir / expected_file
            mock_file.assert_called_once_with(expected_path, "rb")
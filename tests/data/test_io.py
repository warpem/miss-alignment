import pytest
import torch
import mrcfile
import json
from pathlib import Path

from warpylib import TiltSeries
from miss_alignment.data.io import TiltSeriesData


class TestTiltSeriesData:
    """Test suite for TiltSeriesData class."""

    @pytest.fixture
    def sample_tilt_series_data(self, tmp_path):
        """Create a sample TiltSeriesData instance for testing."""
        xml_path = tmp_path / "test.xml"
        stack_path = tmp_path / "test.st"

        # Create dummy files
        xml_path.touch()
        with mrcfile.new(stack_path, overwrite=True) as mrc:
            mrc.set_data(torch.randn(10, 100, 100).numpy())

        return TiltSeriesData(
            xml_metadata_path=xml_path,
            stack_path=stack_path,
            stack_pixel_size=10.0,
            original_pixel_size=5.0,
            original_stack_shape=(200, 200),
            volume_shape=(200, 200, 100),
        )

    def test_initialization(self, tmp_path):
        """Test TiltSeriesData initialization with required parameters."""
        xml_path = tmp_path / "test.xml"
        stack_path = tmp_path / "test.st"

        data = TiltSeriesData(
            xml_metadata_path=xml_path,
            stack_path=stack_path,
            stack_pixel_size=10.0,
            original_pixel_size=5.0,
            original_stack_shape=(200, 200),
            volume_shape=(200, 200, 100),
        )

        assert data.xml_metadata_path == xml_path
        assert data.stack_path == stack_path
        assert data.stack_pixel_size == 10.0
        assert data.original_pixel_size == 5.0
        assert data.original_stack_shape == (200, 200)
        assert data.volume_shape == (200, 200, 100)

    def test_xml_filename_property(self, sample_tilt_series_data):
        """Test xml_filename property returns the stem of xml_metadata_path."""
        assert sample_tilt_series_data.xml_filename == "test"

    def test_replace_method(self, sample_tilt_series_data):
        """Test that replace method creates a new instance with updated values."""
        new_pixel_size = 20.0
        new_data = sample_tilt_series_data.replace(stack_pixel_size=new_pixel_size)

        # Check new value is updated
        assert new_data.stack_pixel_size == new_pixel_size

        # Check other values remain the same
        assert new_data.xml_metadata_path == sample_tilt_series_data.xml_metadata_path
        assert new_data.stack_path == sample_tilt_series_data.stack_path
        assert (
            new_data.original_pixel_size == sample_tilt_series_data.original_pixel_size
        )

        # Check original is unchanged (frozen dataclass)
        assert sample_tilt_series_data.stack_pixel_size == 10.0

    def test_to_dict(self, sample_tilt_series_data):
        """Test to_dict converts TiltSeriesData to dictionary with absolute paths."""
        result = sample_tilt_series_data.to_dict()

        assert isinstance(result, dict)
        assert "xml_metadata_path" in result
        assert "stack_path" in result
        assert "stack_pixel_size" in result
        assert "original_pixel_size" in result
        assert "original_stack_shape" in result
        assert "volume_shape" in result

        # Check paths are converted to absolute strings
        assert isinstance(result["xml_metadata_path"], str)
        assert isinstance(result["stack_path"], str)
        assert Path(result["xml_metadata_path"]).is_absolute()
        assert Path(result["stack_path"]).is_absolute()

        # Check values
        assert result["stack_pixel_size"] == 10.0
        assert result["original_pixel_size"] == 5.0
        assert result["original_stack_shape"] == (200, 200)
        assert result["volume_shape"] == (200, 200, 100)

    def test_from_dict(self, tmp_path):
        """Test from_dict creates TiltSeriesData from dictionary."""
        xml_path = tmp_path / "test.xml"
        stack_path = tmp_path / "test.st"

        data_dict = {
            "xml_metadata_path": str(xml_path),
            "stack_path": str(stack_path),
            "stack_pixel_size": 10.0,
            "original_pixel_size": 5.0,
            "original_stack_shape": [200, 200],
            "volume_shape": [200, 200, 100],
        }

        result = TiltSeriesData.from_dict(data_dict)

        assert isinstance(result, TiltSeriesData)
        assert result.xml_metadata_path == xml_path
        assert result.stack_path == stack_path
        assert result.stack_pixel_size == 10.0
        assert result.original_pixel_size == 5.0
        assert result.original_stack_shape == (200, 200)
        assert result.volume_shape == (200, 200, 100)

    def test_to_json(self, sample_tilt_series_data, tmp_path):
        """Test to_json saves TiltSeriesData to JSON file."""
        json_path = tmp_path / "output.json"

        sample_tilt_series_data.to_json(json_path)

        assert json_path.exists()

        # Verify JSON content
        with open(json_path, "r") as f:
            data = json.load(f)

        assert isinstance(data, dict)
        assert data["stack_pixel_size"] == 10.0
        assert data["original_pixel_size"] == 5.0
        assert data["original_stack_shape"] == [200, 200]

    def test_from_json(self, sample_tilt_series_data, tmp_path):
        """Test from_json loads TiltSeriesData from JSON file."""
        json_path = tmp_path / "output.json"

        # First save to JSON
        sample_tilt_series_data.to_json(json_path)

        # Then load it back
        loaded_data = TiltSeriesData.from_json(json_path)

        assert isinstance(loaded_data, TiltSeriesData)
        assert loaded_data.stack_pixel_size == sample_tilt_series_data.stack_pixel_size
        assert (
            loaded_data.original_pixel_size
            == sample_tilt_series_data.original_pixel_size
        )
        assert (
            loaded_data.original_stack_shape
            == sample_tilt_series_data.original_stack_shape
        )
        assert loaded_data.volume_shape == sample_tilt_series_data.volume_shape

    def test_roundtrip_json(self, sample_tilt_series_data, tmp_path):
        """Test that save->load cycle preserves all data."""
        json_path = tmp_path / "roundtrip.json"

        # Save to JSON
        sample_tilt_series_data.to_json(json_path)

        # Load from JSON
        loaded_data = TiltSeriesData.from_json(json_path)

        # Compare all fields
        assert (
            loaded_data.xml_metadata_path.resolve()
            == sample_tilt_series_data.xml_metadata_path.resolve()
        )
        assert (
            loaded_data.stack_path.resolve()
            == sample_tilt_series_data.stack_path.resolve()
        )
        assert loaded_data.stack_pixel_size == sample_tilt_series_data.stack_pixel_size
        assert (
            loaded_data.original_pixel_size
            == sample_tilt_series_data.original_pixel_size
        )
        assert (
            loaded_data.original_stack_shape
            == sample_tilt_series_data.original_stack_shape
        )
        assert loaded_data.volume_shape == sample_tilt_series_data.volume_shape

    def test_save_metadata_to_xml(self, sample_tilt_series_data):
        """Test save_metadata_to_xml saves TiltSeries metadata to XML file."""
        # Create a mock TiltSeries
        n_tilts = 10
        tilt_series = TiltSeries(n_tilts=n_tilts)
        tilt_series.angles = torch.linspace(-60, 60, n_tilts)

        # Save metadata
        sample_tilt_series_data.save_metadata_to_xml(tilt_series)

        # Verify file exists
        assert sample_tilt_series_data.xml_metadata_path.exists()

        # Load and verify
        loaded_ts = TiltSeries(sample_tilt_series_data.xml_metadata_path)
        assert torch.allclose(loaded_ts.angles, tilt_series.angles)

    def test_load_metadata_and_stack_no_downsample(self, tmp_path):
        """Test load_metadata_and_stack without downsampling."""
        # Create test data
        xml_path = tmp_path / "test.xml"
        stack_path = tmp_path / "test.st"

        n_tilts = 10
        tilt_series = TiltSeries(n_tilts=n_tilts)
        tilt_series.angles = torch.linspace(-60, 60, n_tilts)
        tilt_series.save_meta(xml_path)

        images = torch.randn(n_tilts, 100, 100)
        with mrcfile.new(stack_path, overwrite=True) as mrc:
            mrc.set_data(images.numpy())

        data = TiltSeriesData(
            xml_metadata_path=xml_path,
            stack_path=stack_path,
            stack_pixel_size=10.0,
            original_pixel_size=10.0,
            original_stack_shape=(100, 100),
            volume_shape=(100, 100, 50),
        )

        metadata, loaded_images, pixel_size = data.load_metadata_and_stack(downsample=1)

        assert isinstance(metadata, TiltSeries)
        assert isinstance(loaded_images, torch.Tensor)
        assert pixel_size == 10.0
        assert loaded_images.shape == (n_tilts, 100, 100)
        assert torch.allclose(metadata.angles, tilt_series.angles)

    def test_load_metadata_and_stack_with_downsample(self, tmp_path):
        """Test load_metadata_and_stack with downsampling."""
        # Create test data
        xml_path = tmp_path / "test.xml"
        stack_path = tmp_path / "test.st"

        n_tilts = 10
        tilt_series = TiltSeries(n_tilts=n_tilts)
        tilt_series.angles = torch.linspace(-60, 60, n_tilts)
        tilt_series.save_meta(xml_path)

        images = torch.randn(n_tilts, 100, 100)
        with mrcfile.new(stack_path, overwrite=True) as mrc:
            mrc.set_data(images.numpy())

        data = TiltSeriesData(
            xml_metadata_path=xml_path,
            stack_path=stack_path,
            stack_pixel_size=10.0,
            original_pixel_size=5.0,
            original_stack_shape=(200, 200),
            volume_shape=(200, 200, 100),
        )

        metadata, loaded_images, pixel_size = data.load_metadata_and_stack(downsample=2)

        assert isinstance(metadata, TiltSeries)
        assert isinstance(loaded_images, torch.Tensor)
        assert pixel_size == 20.0  # 10.0 * 2
        # Images should be downsampled
        assert loaded_images.shape[0] == n_tilts
        assert loaded_images.shape[1] <= 100
        assert loaded_images.shape[2] <= 100

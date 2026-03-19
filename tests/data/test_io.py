import pytest
import torch
import mrcfile
import json
from pathlib import Path

from warpylib import TiltSeries
from lxml import etree
from miss_alignment.data.io import TiltSeriesData, _load_settings_metadata


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
        )

    def test_initialization(self, tmp_path):
        """Test TiltSeriesData initialization with required parameters."""
        xml_path = tmp_path / "test.xml"
        stack_path = tmp_path / "test.st"

        data = TiltSeriesData(
            xml_metadata_path=xml_path,
        )

        assert data.xml_metadata_path == xml_path

    def test_xml_filename_property(self, sample_tilt_series_data):
        """Test xml_filename property returns the stem of xml_metadata_path."""
        assert sample_tilt_series_data.xml_filename == "test"

    def test_replace_method(self, sample_tilt_series_data):
        """Test that replace method creates a new instance with updated values."""
        new_metadata_path = "/new_test.xml"
        new_data = sample_tilt_series_data.replace(xml_metadata_path=new_metadata_path)

        # Check new value is updated
        assert new_data.xml_metadata_path == new_metadata_path

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

        n_tilts = 10
        tilt_series = TiltSeries(path=xml_path, n_tilts=n_tilts)
        tilt_series.angles = torch.linspace(-60, 60, n_tilts)
        tilt_series.size_rounding_factors = torch.tensor([1, 1, 1])
        tilt_series.image_dimensions_physical = torch.tensor([1000.0, 1000.0])
        tilt_series.volume_dimensions_physical = torch.tensor([1000.0, 1000.0, 1000.0])
        tilt_series.save_meta(xml_path)

        Path(tilt_series.tilt_stack_path).parent.mkdir(parents=True, exist_ok=True)

        images = torch.randn(n_tilts, 100, 100)
        with mrcfile.new(tilt_series.tilt_stack_path, overwrite=True) as mrc:
            mrc.set_data(images.numpy())
            mrc.voxel_size = 10.0

        data = TiltSeriesData(
            xml_metadata_path=xml_path,
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

        n_tilts = 10
        tilt_series = TiltSeries(path=xml_path, n_tilts=n_tilts)
        tilt_series.angles = torch.linspace(-60, 60, n_tilts)
        tilt_series.size_rounding_factors = torch.tensor([1, 1, 1])
        tilt_series.image_dimensions_physical = torch.tensor([1000.0, 1000.0])
        tilt_series.volume_dimensions_physical = torch.tensor([1000.0, 1000.0, 1000.0])
        tilt_series.save_meta(xml_path)

        Path(tilt_series.tilt_stack_path).parent.mkdir(parents=True, exist_ok=True)

        images = torch.randn(n_tilts, 100, 100)
        with mrcfile.new(tilt_series.tilt_stack_path, overwrite=True) as mrc:
            mrc.set_data(images.numpy())
            mrc.voxel_size = 10.0

        data = TiltSeriesData(
            xml_metadata_path=xml_path,
        )

        metadata, loaded_images, pixel_size = data.load_metadata_and_stack(downsample=2)

        assert isinstance(metadata, TiltSeries)
        assert isinstance(loaded_images, torch.Tensor)
        assert pixel_size == 20.0  # 10.0 * 2
        # Images should be downsampled
        assert loaded_images.shape[0] == n_tilts
        assert loaded_images.shape[1] == 50
        assert loaded_images.shape[2] == 50

    def test_load_settings_metadata(self, tmp_path):
        """Test _load_settings_metadata with a valid settings XML."""
        # Create a dummy XML metadata file
        xml_path = tmp_path / "test.xml"
        ts = TiltSeries(n_tilts=10)
        ts.save_meta(xml_path)

        # Create a dummy settings.xml file
        settings_path = tmp_path / "ts.settings"
        root = etree.Element("Settings")
        import_node = etree.SubElement(root, "Import")
        etree.SubElement(import_node, "Param", Name="PixelSize", Value="2.5")

        tomo_node = etree.SubElement(root, "Tomo")
        etree.SubElement(tomo_node, "Param", Name="DimensionsX", Value="100")
        etree.SubElement(tomo_node, "Param", Name="DimensionsY", Value="120")
        etree.SubElement(tomo_node, "Param", Name="DimensionsZ", Value="80")

        with open(settings_path, "wb") as f:
            f.write(etree.tostring(root))

        # Test loading
        loaded_ts = _load_settings_metadata(settings_path, xml_path)

        assert isinstance(loaded_ts, TiltSeries)
        # 100 * 2.5 = 250, 120 * 2.5 = 300
        assert torch.allclose(
            loaded_ts.image_dimensions_physical, torch.tensor([250.0, 300.0])
        )
        # 80 * 2.5 = 200
        assert torch.allclose(
            loaded_ts.volume_dimensions_physical, torch.tensor([250.0, 300.0, 200.0])
        )

    def test_load_settings_metadata_missing_file(self, tmp_path):
        """Test _load_settings_metadata when settings file is missing."""
        xml_path = tmp_path / "test.xml"
        ts = TiltSeries(n_tilts=10)
        ts.save_meta(xml_path)

        settings_path = tmp_path / "non_existent.settings"

        # Should not raise and should return TiltSeries object
        loaded_ts = _load_settings_metadata(settings_path, xml_path)
        assert isinstance(loaded_ts, TiltSeries)

    def test_load_settings_metadata_none_path(self, tmp_path):
        """Test _load_settings_metadata when settings path is None."""
        xml_path = tmp_path / "test.xml"
        ts = TiltSeries(n_tilts=10)
        ts.save_meta(xml_path)

        # Should not raise and should return TiltSeries object
        loaded_ts = _load_settings_metadata(None, xml_path)
        assert isinstance(loaded_ts, TiltSeries)

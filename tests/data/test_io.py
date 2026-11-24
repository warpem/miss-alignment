import pytest
import torch
import mrcfile
from pathlib import Path
from unittest.mock import patch

from warpylib import TiltSeries
from miss_alignment.data.io import (
    TiltSeriesData,
    exclude_nonfinite_alignment_tilt_series,
    is_deselected,
    list_tilt_series_xmls,
)


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

    def test_load_metadata_and_stack_missing_xml_raises(self, tmp_path):
        """Loading a non-existent XML file raises FileNotFoundError."""
        xml_path = tmp_path / "does_not_exist.xml"

        data = TiltSeriesData(
            xml_metadata_path=xml_path,
        )

        with pytest.raises(FileNotFoundError, match="does_not_exist.xml"):
            data.load_metadata_and_stack()


class TestRetryOnReadError:
    """Tests for the retry_on_read_error flag in load_metadata_and_stack."""

    @pytest.fixture
    def tilt_series_data(self, tmp_path):
        xml_path = tmp_path / "test.xml"
        n_tilts = 5
        ts = TiltSeries(path=xml_path, n_tilts=n_tilts)
        ts.angles = torch.linspace(-60, 60, n_tilts)
        ts.image_dimensions_physical = torch.tensor([1000.0, 1000.0])
        ts.volume_dimensions_physical = torch.tensor([1000.0, 1000.0, 1000.0])
        ts.save_meta(xml_path)
        Path(ts.tilt_stack_path).parent.mkdir(parents=True, exist_ok=True)
        with mrcfile.new(ts.tilt_stack_path, overwrite=True) as mrc:
            mrc.set_data(torch.randn(n_tilts, 50, 50).numpy())
            mrc.voxel_size = 10.0
        return TiltSeriesData(xml_metadata_path=xml_path)

    def test_transient_error_retried_and_succeeds(self, tilt_series_data):
        """retry_on_read_error=True retries a short-read ValueError until success."""
        real_open = mrcfile.open
        call_count = 0

        def fail_twice_then_succeed(path, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise ValueError("Couldn't read enough bytes for MRC header")
            return real_open(path, **kwargs)

        with patch(
            "miss_alignment.data.io.mrcfile.open", side_effect=fail_twice_then_succeed
        ):
            with patch("miss_alignment.data.io.time.sleep"):
                tilt_series_data.load_metadata_and_stack(retry_on_read_error=True)

        assert call_count == 3

    def test_transient_error_propagates_without_flag(self, tilt_series_data):
        """retry_on_read_error=False (default) propagates the error immediately."""
        with patch(
            "miss_alignment.data.io.mrcfile.open",
            side_effect=ValueError("Couldn't read enough bytes for MRC header"),
        ):
            with pytest.raises(ValueError, match="read enough bytes"):
                tilt_series_data.load_metadata_and_stack(retry_on_read_error=False)

    def test_permanent_error_not_retried(self, tilt_series_data):
        """Permanent errors (no 'read' in message) are not retried."""
        call_count = 0

        def permanent_error(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            raise ValueError(
                "Map ID string not found - not an MRC file, or file is corrupt"
            )

        with patch("miss_alignment.data.io.mrcfile.open", side_effect=permanent_error):
            with pytest.raises(ValueError, match="Map ID"):
                tilt_series_data.load_metadata_and_stack(retry_on_read_error=True)

        assert call_count == 1

    def test_all_retries_exhausted_raises(self, tilt_series_data):
        """After 5 failed attempts the error propagates."""
        call_count = 0

        def always_fail(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            raise ValueError(
                "Expected 1024 bytes in data block but could only read 512"
            )

        with patch("miss_alignment.data.io.mrcfile.open", side_effect=always_fail):
            with patch("miss_alignment.data.io.time.sleep"):
                with pytest.raises(ValueError, match="could only read"):
                    tilt_series_data.load_metadata_and_stack(retry_on_read_error=True)

        assert call_count == 5


def _write_tilt_series_xml(xml_path, unselect_manual=None, angles=None):
    n_tilts = 5
    tilt_series = TiltSeries(path=xml_path, n_tilts=n_tilts)
    tilt_series.angles = torch.linspace(-60, 60, n_tilts) if angles is None else angles
    tilt_series.unselect_manual = unselect_manual
    tilt_series.save_meta(xml_path)
    return xml_path


class TestWarpDeselection:
    """Tilt-series deselected in Warp (UnselectManual) are skipped."""

    @pytest.mark.parametrize(
        "unselect_manual, expected", [(True, True), (False, False), (None, False)]
    )
    def test_is_deselected(self, tmp_path, unselect_manual, expected):
        xml_path = _write_tilt_series_xml(tmp_path / "ts.xml", unselect_manual)
        assert is_deselected(xml_path) == expected

    def test_warp_null_value_is_selected(self, tmp_path):
        """Warp writes UnselectManual="" when the flag was never set."""
        xml_path = tmp_path / "ts.xml"
        xml_path.write_text('<TiltSeries UnselectManual="" UnselectFilter="False" />')
        assert not is_deselected(xml_path)

    def test_unparseable_xml_is_selected(self, tmp_path):
        xml_path = tmp_path / "ts.xml"
        xml_path.write_text("not xml")
        assert not is_deselected(xml_path)

    def test_list_tilt_series_xmls_skips_deselected(self, tmp_path):
        selected = _write_tilt_series_xml(tmp_path / "a.xml", unselect_manual=False)
        _write_tilt_series_xml(tmp_path / "b.xml", unselect_manual=True)
        unset = _write_tilt_series_xml(tmp_path / "c.xml")

        assert list_tilt_series_xmls(tmp_path) == [selected, unset]

    def test_exclude_nonfinite_leaves_deselected_in_place(self, tmp_path, capsys):
        """Deselected tilt-series are reported but neither checked nor moved."""
        _write_tilt_series_xml(tmp_path / "good.xml")
        bad_angles = torch.tensor([float("nan")] * 5)
        deselected = _write_tilt_series_xml(
            tmp_path / "deselected.xml", unselect_manual=True, angles=bad_angles
        )

        excluded = exclude_nonfinite_alignment_tilt_series(tmp_path)

        assert excluded == []
        assert deselected.exists()
        assert "deselected in Warp" in capsys.readouterr().out

    def test_exclude_nonfinite_raises_if_all_deselected(self, tmp_path):
        _write_tilt_series_xml(tmp_path / "ts.xml", unselect_manual=True)

        with pytest.raises(ValueError, match="deselected 1"):
            exclude_nonfinite_alignment_tilt_series(tmp_path)

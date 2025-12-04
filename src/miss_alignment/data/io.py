import pickle
import torch
import mrcfile
from pathlib import Path

from warpylib import TiltSeries
from warpylib.ops import rescale
from dataclasses import dataclass, replace


@dataclass(kw_only=True, frozen=True)
class TiltSeriesData:
    # required variables
    xml_metadata_path: Path

    def replace(self, **kwargs):
        return replace(self, **kwargs)

    @property
    def xml_filename(self) -> str:
        return self.xml_metadata_path.stem

    def load_metadata_and_stack(
        self, downsample=1
    ) -> tuple[TiltSeries, torch.Tensor, float]:
        metadata, images, pixel_size = _load_metadata_and_stack(
            self.xml_metadata_path,
            downsample=downsample,
        )
        return metadata, images, pixel_size

    def save_metadata_to_xml(self, tilt_series: TiltSeries) -> None:
        tilt_series.save_meta(self.xml_metadata_path)


def _load_metadata_and_stack(
    metadata_path: Path,
    downsample: int = 1,
) -> tuple[TiltSeries, torch.Tensor, float]:
    # load metadata
    tilt_series = TiltSeries(metadata_path)

    stack_path = tilt_series.tilt_stack_path

    # load the data
    with mrcfile.open(stack_path) as mrc:
        images = torch.tensor(mrc.data)
        stack_pixel_size = float(mrc.voxel_size.x)

    pixel_size = stack_pixel_size
    if downsample > 1:
        pixel_size = stack_pixel_size * downsample
        images = rescale(
            images,
            size=(
                (images.shape[-2] // downsample + 1) // 2 * 2,
                (images.shape[-1] // downsample + 1) // 2 * 2,
            ),
        )

    # set rounding factors
    tilt_series.size_rounding_factors = torch.tensor(
        [
            (images.shape[-1] * pixel_size) / tilt_series.image_dimensions_physical[0],
            (images.shape[-2] * pixel_size) / tilt_series.image_dimensions_physical[1],
            1.0,  # Z dimension (not used for 2D images)
        ],
        dtype=torch.float32,
    )
    return tilt_series, images, pixel_size


def convert_pickle_to_xml_helper(
    pickle_data_path: Path,
    stack_pixel_size: float,
    original_pixel_size: float,
    original_stack_shape: tuple[int, int],
    volume_shape: tuple[int, int, int],
) -> tuple[TiltSeriesData, Path]:
    """
    Convert a pickle file containing tilt-series data to XML format.

    Parameters
    ----------
    pickle_data_path : Path
        Path to the pickle file containing tilt-series data.
    stack_pixel_size : float
        Pixel size of the stack in Angstroms.
    original_pixel_size : float
        Original pixel size before any rescaling in Angstroms.
    original_stack_shape : tuple[int, int]
        Original stack shape as (width, height) in pixels.
    volume_shape : tuple[int, int, int]
        Volume shape as (x, y, z) in pixels.

    Returns
    -------
    tuple[TiltSeriesData, Path]
        TiltSeriesData object and path to the created XML file.
    """
    # read the pickle
    with open(pickle_data_path, "rb") as infile:
        data_dict = pickle.load(infile)
    images = data_dict["tilt_series"]
    n_tilts, _, _ = images.shape

    # prepare output paths
    tilt_series_name = pickle_data_path.stem
    data_directory = pickle_data_path.parent
    stack_path = (data_directory / f"{tilt_series_name}.st").absolute()
    xml_path = (data_directory / f"{tilt_series_name}.xml").absolute()

    # initialize a fresh warpylib TiltSeries with path
    tilt_series = TiltSeries(
        path=xml_path,
        n_tilts=n_tilts,
    )
    tilt_series.angles = data_dict["tilt_angles"]

    # Handle scalar tilt_axis_angle by expanding to array
    tilt_axis_angle = data_dict["tilt_axis_angle"]
    if tilt_axis_angle.ndim == 0:  # scalar tensor
        tilt_axis_angle = torch.full((n_tilts,), tilt_axis_angle.item())
    tilt_series.tilt_axis_angles = tilt_axis_angle

    axis_offsets_angstrom = data_dict["sample_translations"] * stack_pixel_size
    tilt_series.tilt_axis_offset_y = axis_offsets_angstrom[:, 0].clone()
    tilt_series.tilt_axis_offset_x = axis_offsets_angstrom[:, 1].clone()

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

    # Write MRC stack
    with mrcfile.new(stack_path, overwrite=True) as mrc:
        mrc.set_data(images.cpu().numpy())
        mrc.voxel_size = stack_pixel_size

    # Create TiltSeriesData and save metadata to XML
    new_tilt_series_data = TiltSeriesData(
        xml_metadata_path=xml_path,
    )
    new_tilt_series_data.save_metadata_to_xml(tilt_series)

    return new_tilt_series_data, xml_path

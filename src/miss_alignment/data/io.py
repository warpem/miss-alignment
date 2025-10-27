import pickle
import torch
import mrcfile
from warnings import warn
from pathlib import Path
from warpylib import TiltSeries
from typing import Optional


def _read_tomogram_from_pickle(
        data_path: Path,
        stack_pixel_size: float,
        original_pixel_size: float,
        original_stack_shape: tuple[int, int],
) -> tuple[TiltSeries, torch.Tensor]:
    with open(data_path, "rb") as infile:
        data_dict = pickle.load(infile)
    images = data_dict["tilt_series"]
    n_tilts, _, _ = images.shape
    tilt_series = TiltSeries(
        n_tilts=n_tilts,
        image_dimensions_physical=torch.tensor(
            [s * original_pixel_size for s in original_stack_shape], dtype=torch.float32
        ),
    )
    tilt_series.use_tilt = torch.tensor(n_tilts * [True], dtype=torch.bool)
    tilt_series.angles = data_dict["tilt_angles"]
    tilt_series.tilt_axis_angles = data_dict["tilt_axis_angle"]
    axis_offsets_angstrom = data_dict["sample_translations"] * stack_pixel_size
    tilt_series.tilt_axis_offset_y = axis_offsets_angstrom[:, 0].clone()
    tilt_series.tilt_axis_offset_x = axis_offsets_angstrom[:, 1].clone()

    # get scaled width and height
    original_width, original_height = original_stack_shape
    downsample_factor = stack_pixel_size / original_pixel_size
    scaled_width = int(round(original_width / downsample_factor / 2)) * 2
    scaled_height = int(round(original_height / downsample_factor / 2)) * 2

    # set rounding factors
    tilt_series.size_rounding_factors = torch.tensor(
        [
            scaled_width / (original_width / downsample_factor),
            scaled_height / (original_height / downsample_factor),
            1.0,  # Z dimension (not used for 2D images)
        ],
        dtype=torch.float32,
    )

    return tilt_series, images


def _read_warp_metadata(
        metadata_path: Path,
        stack_path: Path,
        stack_pixel_size: float,
        original_pixel_size: float,
        original_stack_shape: tuple[int, int],
) -> tuple[TiltSeries, torch.Tensor]:
    # load the data
    with mrcfile.open(stack_path) as mrc:
        images = torch.tensor(mrc.data)
    tilt_series = TiltSeries(metadata_path)

    # set original physical image dimensions
    original_width, original_height = original_stack_shape
    tilt_series.image_dimensions_physical = torch.tensor(
        [s * original_pixel_size for s in original_stack_shape],
        dtype=torch.float32)

    # get scaled width and height
    downsample_factor = stack_pixel_size / original_pixel_size
    scaled_width = int(round(original_width / downsample_factor / 2)) * 2
    scaled_height = int(round(original_height / downsample_factor / 2)) * 2

    # set rounding factors
    tilt_series.size_rounding_factors = torch.tensor(
        [
            scaled_width / (original_width / downsample_factor),
            scaled_height / (original_height / downsample_factor),
            1.0,  # Z dimension (not used for 2D images)
        ],
        dtype=torch.float32,
    )
    return tilt_series, images


def load_tilt_series(
        metadata_path: Path,
        stack_pixel_size: float,
        original_pixel_size: float,
        original_stack_shape: tuple[int, int],
        stack_path: Optional[Path] = None,
) -> tuple[TiltSeries, torch.Tensor]:
    """Load tilt-series metadata and images.

    Expects to load a Warp .xml file with metadata and a tilt-series stack
    of images together with the pixel size of the stack. It can also read
    torch_tomogram. Tomogram pickle object for backwards compatibility.

    Arguments
    -------
    metadata_path: Path
        Path to the metadata file either .pickle or .xml
    stack_pixel_size: float
        Pixel size of the tilt-series stack in the pickle or the MRC format
        stack
    original_pixel_size: float
        Pixel size of raw data before downsampling
    original_stack_shape: tuple[int, int]
        Shape of images before downsampling, used to correct for rounding
    stack_path: Path, default None
        Path to the stack file with the tilt series images in MRC format
        with either .mrc or .st extension

    Returns
    -------
    metadata: warpylib.TiltSeries
        TiltSeries object from warpylib containing all the metadata
    images: torch.Tensor
        The images of the tilt series
    """
    if metadata_path.suffix == ".pickle":
        metadata, images = (
            _read_tomogram_from_pickle(
                metadata_path,
                stack_pixel_size,
                original_pixel_size,
                original_stack_shape
            )
        )
    elif metadata_path.suffix == ".xml":
        if stack_path is not None:
            metadata, images = (
                _read_warp_metadata(
                    metadata_path,
                    stack_path,
                    stack_pixel_size,
                    original_pixel_size,
                    original_stack_shape,
                )
            )
        else:
            raise ValueError("Warp XML file also requires a tilt stack path.")
    else:
        raise ValueError("Invalid metadata file type; expected .pickle or "
                         f".xml, but got {metadata_path.suffix}")
    return metadata, images


def save_tilt_series(tilt_series: TiltSeries, metadata_path: Path):
    tilt_series.save_meta(metadata_path)
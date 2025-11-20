import pickle
import torch
import mrcfile
import json
from pathlib import Path

from torch_fourier_rescale import fourier_rescale_2d
from warpylib import TiltSeries
from dataclasses import dataclass, replace, asdict


@dataclass(kw_only=True, frozen=True)
class TiltSeriesData:
    # required variables
    xml_metadata_path: Path
    stack_path: Path
    stack_pixel_size: float
    original_pixel_size: float
    original_stack_shape: tuple[int, int]
    volume_shape: tuple[int, int, int]

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
            self.stack_path,
            self.stack_pixel_size,
            self.original_pixel_size,
            self.original_stack_shape,
            self.volume_shape,
            downsample=downsample,
        )
        return metadata, images, pixel_size

    def save_metadata_to_xml(self, tilt_series: TiltSeries) -> None:
        tilt_series.save_meta(self.xml_metadata_path)

    def to_dict(self) -> dict:
        data = asdict(self)
        # Convert to absolute paths while respecting symbolic links
        data["xml_metadata_path"] = str(self.xml_metadata_path.absolute())
        data["stack_path"] = str(self.stack_path.absolute())
        return data

    def to_json(self, path: Path) -> None:
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def from_dict(cls, data: dict) -> "TiltSeriesData":
        """Use as: dict = TiltSeriesData.from_dict(some_dict)"""
        data = data.copy()
        data["xml_metadata_path"] = Path(data["xml_metadata_path"])
        data["stack_path"] = Path(data["stack_path"])
        data["original_stack_shape"] = tuple(data["original_stack_shape"])
        data["volume_shape"] = tuple(data["volume_shape"])
        return cls(**data)

    @classmethod
    def from_json(cls, path: Path) -> "TiltSeriesData":
        """Use as: data = TiltSeriesData.from_json(some_path)"""
        with open(path, "r") as f:
            data = json.load(f)
        return cls.from_dict(data)


def _load_metadata_and_stack(
    metadata_path: Path,
    stack_path: Path,
    stack_pixel_size: float,
    original_pixel_size: float,
    original_stack_shape: tuple[int, int],
    volume_shape: tuple[int, int, int],
    downsample: int = 1,
) -> tuple[TiltSeries, torch.Tensor, float]:
    # load the data
    with mrcfile.open(stack_path) as mrc:
        images = torch.tensor(mrc.data)

    pixel_size = stack_pixel_size
    if downsample > 1:
        pixel_size = stack_pixel_size * downsample
        images, _new_pixel_size = fourier_rescale_2d(
            images, stack_pixel_size, pixel_size
        )
        # not correct, but shortcut to deal with unevenly spaced dims
        pixel_size, _ = _new_pixel_size

    tilt_series = TiltSeries(metadata_path)

    # set original physical image dimensions
    original_width, original_height = original_stack_shape
    tilt_series.image_dimensions_physical = torch.tensor(
        [s * original_pixel_size for s in original_stack_shape], dtype=torch.float32
    )
    tilt_series.volume_dimensions_physical = torch.tensor(
        [s * original_pixel_size for s in volume_shape],
        dtype=torch.float32,
    )

    # get scaled width and height
    downsample_factor = pixel_size / original_pixel_size
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
    return tilt_series, images, pixel_size


def merge_pickle_and_xml_to_json_helper(
    pickle_data_path: Path,
    xml_data_path: Path,
    stack_pixel_size: float,
    original_pixel_size: float,
    original_stack_shape: tuple[int, int],
    volume_shape: tuple[int, int, int],
) -> tuple[TiltSeriesData, Path]:
    # read the pickle
    with open(pickle_data_path, "rb") as infile:
        data_dict = pickle.load(infile)
    images = data_dict["tilt_series"]

    # load existing xml metadata
    tilt_series = TiltSeries(xml_data_path)

    # set all the alignments from the pickle
    tilt_series.angles = data_dict["tilt_angles"]
    tilt_series.tilt_axis_angles = data_dict["tilt_axis_angle"]
    axis_offsets_angstrom = data_dict["sample_translations"] * stack_pixel_size
    tilt_series.tilt_axis_offset_y = axis_offsets_angstrom[:, 0].clone()
    tilt_series.tilt_axis_offset_x = axis_offsets_angstrom[:, 1].clone()

    # write all output files
    tilt_series_name = pickle_data_path.stem
    data_directory = pickle_data_path.parent
    stack_path = (data_directory / f"{tilt_series_name}.st").absolute()
    xml_path = (data_directory / f"{tilt_series_name}.xml").absolute()
    json_path = (data_directory / f"{tilt_series_name}.json").absolute()
    with mrcfile.new(stack_path, overwrite=True) as mrc:
        mrc.set_data(images.cpu().numpy())
        mrc.set_voxel = stack_pixel_size
    new_tilt_series_data = TiltSeriesData(
        xml_metadata_path=xml_path,
        stack_path=stack_path,
        stack_pixel_size=stack_pixel_size,
        original_stack_shape=original_stack_shape,
        original_pixel_size=original_pixel_size,
        volume_shape=volume_shape,
    )
    new_tilt_series_data.save_metadata_to_xml(tilt_series)
    new_tilt_series_data.to_json(json_path)
    return new_tilt_series_data, json_path


def convert_pickle_to_json_helper(
    pickle_data_path: Path,
    stack_pixel_size: float,
    original_pixel_size: float,
    original_stack_shape: tuple[int, int],
    volume_shape: tuple[int, int, int],
) -> tuple[TiltSeriesData, Path]:
    # read the pickle
    with open(pickle_data_path, "rb") as infile:
        data_dict = pickle.load(infile)
    images = data_dict["tilt_series"]
    n_tilts, _, _ = images.shape

    # initialize a fresh warpylib TiltSeries
    tilt_series = TiltSeries(
        n_tilts=n_tilts,
    )
    tilt_series.angles = data_dict["tilt_angles"]

    # Handle scalar tilt_axis_angle by expanding to array
    tilt_axis_angle = data_dict["tilt_axis_angle"]
    if tilt_axis_angle.ndim == 0:  # scalar tensor
        import torch

        tilt_axis_angle = torch.full((n_tilts,), tilt_axis_angle.item())
    tilt_series.tilt_axis_angles = tilt_axis_angle

    axis_offsets_angstrom = data_dict["sample_translations"] * stack_pixel_size
    tilt_series.tilt_axis_offset_y = axis_offsets_angstrom[:, 0].clone()
    tilt_series.tilt_axis_offset_x = axis_offsets_angstrom[:, 1].clone()

    # write all output files
    tilt_series_name = pickle_data_path.stem
    data_directory = pickle_data_path.parent
    stack_path = (data_directory / f"{tilt_series_name}.st").absolute()
    xml_path = (data_directory / f"{tilt_series_name}.xml").absolute()
    json_path = (data_directory / f"{tilt_series_name}.json").absolute()
    with mrcfile.new(stack_path, overwrite=True) as mrc:
        mrc.set_data(images.cpu().numpy())
        mrc.set_voxel = stack_pixel_size
    new_tilt_series_data = TiltSeriesData(
        xml_metadata_path=xml_path,
        stack_path=stack_path,
        stack_pixel_size=stack_pixel_size,
        original_stack_shape=original_stack_shape,
        original_pixel_size=original_pixel_size,
        volume_shape=volume_shape,
    )
    new_tilt_series_data.save_metadata_to_xml(tilt_series)
    new_tilt_series_data.to_json(json_path)
    return new_tilt_series_data, json_path

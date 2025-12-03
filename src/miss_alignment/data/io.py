import pickle
import torch
import mrcfile
import json
from pathlib import Path

from torch_fourier_rescale import fourier_rescale_2d
from warpylib import TiltSeries
from warpylib.ops import rescale
from dataclasses import dataclass, replace, asdict


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
        stack_pixel_size = mrc.voxel_size.x

    pixel_size = stack_pixel_size
    if downsample > 1:
        pixel_size = stack_pixel_size * downsample
        images = rescale(images, size=(
            (images.shape[-2] // downsample + 1) // 2 * 2,
            (images.shape[-1] // downsample + 1) // 2 * 2,
        ))

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
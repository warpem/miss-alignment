"""Tilt series alignment optimization.

This module provides the main entry point for evaluating and optimizing
tilt series alignment using trained models.
"""

from pathlib import Path

import einops
import torch
from warpylib.tilt_series.reconstruct_volume import preprocess_tilt_data

from miss_alignment.data.io import TiltSeriesData
from miss_alignment.models import MissAlignment

# Re-export optimization functions for backwards compatibility
from .optimize_global import optimize_shifts
from .optimize_iterative import optimize_shifts_iterative, run_iterative_anchoring
from .optimize_spline import (
    evaluate_catmull_rom_spline,
    optimize_shifts_coarse_to_fine,
    optimize_shifts_spline,
)

__all__ = [
    "generate_position_grid",
    "evaluate_tilt_series",
    "optimize_shifts",
    "optimize_shifts_iterative",
    "optimize_shifts_spline",
    "optimize_shifts_coarse_to_fine",
    "run_iterative_anchoring",
    "evaluate_catmull_rom_spline",
]


def generate_position_grid(
    volume_dimensions_physical: tuple[float, float, float] | torch.Tensor,
    pixel_size: float,
    patch_size: int,
    patch_overlap: float,
) -> torch.Tensor:
    """Generate a grid of 3D positions for reconstruction.

    Parameters
    ----------
    volume_dimensions_physical : tuple[float, float, float] | torch.Tensor
        Physical dimensions of the volume (x, y, z) in Angstroms.
    pixel_size : float
        Pixel size in Angstroms.
    patch_size : int
        Size of reconstruction patches in pixels.
    patch_overlap : float
        Fractional overlap between patches (0 to 1).

    Returns
    -------
    torch.Tensor
        Grid of (x, y, z) positions, shape (n_positions, 3).
    """
    # calculate patches per dimensions
    stride = int(patch_size * (1 - patch_overlap))
    patches_per_dim = []
    for s in volume_dimensions_physical:
        dim = s // pixel_size
        patches_per_dim += [max(int(dim // stride), 1)]

    # extract patches per dim and physical size
    px, py, pz = patches_per_dim
    x, y, z = volume_dimensions_physical

    # get the reconstruction positions to optimize over
    zs = ((torch.arange(pz) + 0.5) / pz) * z
    ys = ((torch.arange(py) + 0.5) / py) * y
    xs = ((torch.arange(px) + 0.5) / px) * x

    # merge positions in tensor
    position_grid = torch.stack(torch.meshgrid(xs, ys, zs, indexing="ij"), dim=-1)
    # ensure we get a list of xyz positions!!
    position_grid = einops.rearrange(position_grid, "d h w xyz -> (d h w) xyz")
    return position_grid


def evaluate_tilt_series(
    model_checkpoint_path: Path,
    tilt_series_path: Path,
    output_directory: Path,
    setting: str | tuple[int, int, int] | tuple[int, int, int, int] = "anchoring",
    patch_size: int = 96,
    patch_overlap: float = 0.1,
    batch_size: int = 16,
    apply_ctf: bool = True,
    downsample: int = 1,
    device: str = "cpu",
    initial_reliable_fraction: float = 1 / 2,
    n_control_points: int = 7,
) -> tuple[Path, list[float]]:
    """Evaluate and optimize tilt series alignment using trained model.

    Parameters
    ----------
    model_checkpoint_path : Path
        Path to trained model checkpoint.
    tilt_series_path : Path
        Path to tilt series JSON file.
    output_directory : Path
        Directory to write output files.
    setting : str | tuple
        Optimization mode:
        - "global": per-tilt 2D shifts (single pass)
        - "anchoring": iterative anchoring with per-tilt shifts (default)
        - "spline": smooth spline + per-tilt fine adjustment (coarse-to-fine)
        - tuple(int, int, int): 2D warping field per image
        - tuple(int, int, int, int): 3D volume warp grid
    patch_size : int
        Size of reconstruction patches.
    patch_overlap : float
        Overlap between patches.
    batch_size : int
        Batch size for reconstruction.
    apply_ctf : bool
        Whether to apply CTF correction.
    downsample : int
        Downsampling factor for tilt images.
    device : str
        Device to run optimization on.
    initial_reliable_fraction : float
        Initial reliable fraction for "anchoring" setting.
    n_control_points : int
        Number of spline control points for "spline" setting.

    Returns
    -------
    tuple[Path, list[float]]
        Path to output JSON and list of loss values.
    """
    # load the best model and run alignment optimization
    model = MissAlignment.load_from_checkpoint(
        model_checkpoint_path,
        map_location="cpu",
    )

    # load tilt_series and set its name for output
    tilt_series_data = TiltSeriesData.from_json(tilt_series_path)
    tilt_series, images, pixel_size = tilt_series_data.load_metadata_and_stack(
        downsample=downsample
    )
    # run the preprocessing from warp for consistency
    images = preprocess_tilt_data(
        tilt_data=images,
        normalize=True,
        invert=False,
        subvolume_size=patch_size,
    )

    # generate a position grid to optimize over
    position_grid = generate_position_grid(
        volume_dimensions_physical=tilt_series.volume_dimensions_physical,
        pixel_size=pixel_size,
        patch_size=patch_size,
        patch_overlap=patch_overlap,
    )

    # run alignment optimization with the model
    if setting == "anchoring":
        # Iterative anchoring with per-tilt shifts
        tilt_series, loss = optimize_shifts_iterative(
            model=model,
            tilt_series=tilt_series,
            images=images,
            pixel_size=pixel_size,
            positions=position_grid,
            patch_size=patch_size,
            batch_size=batch_size,
            apply_ctf=apply_ctf,
            device=device,
            initial_reliable_fraction=initial_reliable_fraction,
        )
    elif setting == "spline":
        # Coarse-to-fine: smooth spline followed by per-tilt adjustment
        tilt_series, loss = optimize_shifts_coarse_to_fine(
            model=model,
            tilt_series=tilt_series,
            images=images,
            pixel_size=pixel_size,
            positions=position_grid,
            patch_size=patch_size,
            batch_size=batch_size,
            apply_ctf=apply_ctf,
            device=device,
            n_control_points=n_control_points,
        )
    else:
        # "global" or tuple settings for 2D/3D warping
        tilt_series, loss = optimize_shifts(
            model,
            tilt_series,
            images,
            pixel_size,
            position_grid,
            setting=setting,
            patch_size=patch_size,
            batch_size=batch_size,
            apply_ctf=apply_ctf,
            device=device,
        )

    # write all necessary output
    xml_out = (output_directory / (tilt_series_data.xml_filename + ".xml")).absolute()
    json_out = (output_directory / (tilt_series_data.xml_filename + ".json")).absolute()
    new_tilt_series_data = tilt_series_data.replace(
        xml_metadata_path=xml_out,
    )
    new_tilt_series_data.save_metadata_to_xml(tilt_series)
    new_tilt_series_data.to_json(json_out)

    return json_out, loss

#!/usr/bin/env python3
"""
Process IMOD tilt-series data to warpylib TiltSeries format.

This script reads IMOD files (.rawtlt, .st, .xf) and creates
warpylib TiltSeries XML metadata + MRC stack files directly.

No dependencies on miss_alignment or torch_tomogram packages.
"""

import os
from pathlib import Path
import torch
import numpy as np
import mrcfile

from warpylib import TiltSeries
from warpylib.ops.rescale import rescale


def read_tilt_angles(rawtlt_path: Path) -> list[float]:
    """
    Read tilt angles from .rawtlt file.

    Parameters
    ----------
    rawtlt_path : Path
        Path to the .rawtlt file

    Returns
    -------
    list of float
        Tilt angles in degrees
    """
    with open(rawtlt_path, 'r') as f:
        return [float(x.strip()) for x in f.readlines()]


def parse_transform_file(filepath: Path) -> torch.Tensor:
    """
    Parse IMOD .xf transformation file to extract sample translations.

    IMOD .xf files contain 2D transformation matrices for each tilt image:
    matrix[0], matrix[1], matrix[2], matrix[3], shift[0], shift[1]

    The transformation maps from aligned coordinates to raw image coordinates.
    We need to extract the shifts and apply inverse rotation to get translations
    in the aligned coordinate system.

    Parameters
    ----------
    filepath : Path
        Path to the .xf file

    Returns
    -------
    torch.Tensor
        Sample translations in pixels, shape (n_tilts, 2) for YX order
    """
    data = np.loadtxt(filepath)

    # Extract rotation matrices (2x2) and shifts
    rot_matrices = data[:, :4].reshape(-1, 2, 2)  # (N, 2, 2)
    shifts = data[:, 4:6]  # (N, 2) in XY order

    # Compute inverse rotation matrices
    rot_inv = np.linalg.inv(rot_matrices)  # (N, 2, 2)

    # Apply inverse rotation to shifts to get translations in aligned space
    corrected_shifts = np.einsum('nij,nj->ni', rot_inv, shifts)

    # Flip from XY to YX order and negate
    # Negation converts from "where to sample from raw" to "how much to shift aligned"
    corrected_shifts = -1.0 * np.flip(corrected_shifts, axis=1)

    return torch.tensor(corrected_shifts, dtype=torch.float32)


def process_tilt_series(
    folder_path: Path,
    tilt_axis_angle: float,
    volume_shape: tuple[int, int, int],
    output_pixel_size: float | None = None,
    output_directory: Path | None = None,
) -> tuple[TiltSeries, Path]:
    """
    Process a single IMOD tilt-series folder to create warpylib TiltSeries.

    Parameters
    ----------
    folder_path : Path
        Path to the tilt-series folder
    tilt_axis_angle : float
        Tilt axis rotation angle in degrees
    volume_shape : tuple of int
        Target volume shape (x, y, z) in pixels for reconstruction
    output_pixel_size : float, optional
        Target pixel size in Angstroms for downsampling. If None, no downsampling.
    output_directory : Path, optional
        Directory to write output files. If None, uses folder_path.

    Returns
    -------
    tuple[TiltSeries, Path]
        TiltSeries object and path to the created XML file
    """
    folder_name = folder_path.name

    # Construct file paths
    rawtlt_path = folder_path / f"{folder_name}.rawtlt"
    mrc_path = folder_path / f"{folder_name}.st"
    xf_path = folder_path / f"{folder_name}.xf"

    # Check required files exist
    for path, desc in [(rawtlt_path, "tilt angles"), (mrc_path, "image stack"), (xf_path, "transforms")]:
        if not path.exists():
            raise FileNotFoundError(f"{desc} file not found: {path}")

    print(f"Processing {folder_name}...")

    # Read tilt angles
    tilt_angles = read_tilt_angles(rawtlt_path)
    n_tilts = len(tilt_angles)

    # Read sample translations from transform file and convert to Angstroms
    sample_translations = parse_transform_file(xf_path)  # (n_tilts, 2) in YX order, pixels

    # Load tilt images and get pixel size from MRC
    with mrcfile.open(mrc_path) as mrc:
        images = torch.tensor(mrc.data.astype(np.float32))
        pixel_size = float(mrc.voxel_size.x)

    # Convert sample translations to Angstroms
    sample_translations = sample_translations * pixel_size  # now in Angstroms

    print(f"  Tilt series shape: {images.shape}")
    print(f"  Original pixel size: {pixel_size} Å")

    # Track final pixel size (may be updated by downsampling)
    final_pixel_size = pixel_size

    # Apply downsampling if requested
    if output_pixel_size is not None and output_pixel_size > pixel_size:
        original_height, original_width = images.shape[-2:]
        downsample_factor = output_pixel_size / pixel_size

        # Calculate scaled dimensions, ensuring they are even
        scaled_width = int(round(original_width / downsample_factor / 2)) * 2
        scaled_height = int(round(original_height / downsample_factor / 2)) * 2
        scaled_shape = (scaled_height, scaled_width)

        # Calculate size rounding factors
        size_rounding_factors = torch.tensor(
            [
                scaled_width / (original_width / downsample_factor),
                scaled_height / (original_height / downsample_factor),
                1.0,  # Z dimension (not used for 2D images)
            ],
            dtype=torch.float32,
        )

        # Rescale if needed
        if scaled_shape != (original_height, original_width):
            images = rescale(images, size=scaled_shape)

        # Update final pixel size based on actual scaling (accounts for rounding)
        final_pixel_size = pixel_size * original_width / scaled_width
        print(f"  Downsampled to pixel size: {final_pixel_size} Å")
        print(f"  New shape: {images.shape}")
    elif output_pixel_size is not None and output_pixel_size < pixel_size:
        raise ValueError('Invalid output_pixel_size: it is smaller than the input')

    print(f"  Final pixel size: {final_pixel_size} Å")

    # Determine output directory
    if output_directory is None:
        output_directory = folder_path

    # Create tiltstack directory structure expected by warpylib
    # warpylib expects: data_dir/tiltstack/series_name/series_name.st
    stack_dir = output_directory / "tiltstack" / folder_name
    stack_dir.mkdir(parents=True, exist_ok=True)
    stack_path = stack_dir / f"{folder_name}.st"

    # Create XML path
    xml_path = output_directory / f"{folder_name}.xml"

    # Initialize TiltSeries
    tilt_series = TiltSeries(
        path=xml_path,
        n_tilts=n_tilts,
    )

    # Set tilt angles
    tilt_series.angles = torch.tensor(tilt_angles, dtype=torch.float32)

    # Set tilt axis angles (constant for all tilts)
    tilt_series.tilt_axis_angles = torch.full((n_tilts,), tilt_axis_angle, dtype=torch.float32)

    # Set sample translations (already in Angstroms)
    # sample_translations is (n_tilts, 2) in YX order
    tilt_series.tilt_axis_offset_y = sample_translations[:, 0]
    tilt_series.tilt_axis_offset_x = sample_translations[:, 1]

    # Set physical dimensions
    # Image dimensions: actual image size in Angstroms
    image_height, image_width = images.shape[-2:]
    tilt_series.image_dimensions_physical = torch.tensor(
        [image_width * pixel_size, image_height * pixel_size],
        dtype=torch.float32,
    )

    # Volume dimensions: target reconstruction size in Angstroms
    tilt_series.volume_dimensions_physical = torch.tensor(
        [volume_shape[0] * pixel_size, volume_shape[1] * pixel_size, volume_shape[2] * pixel_size],
        dtype=torch.float32,
    )

    # Write MRC stack
    with mrcfile.new(stack_path, overwrite=True) as mrc:
        mrc.set_data(images.cpu().numpy())
        mrc.voxel_size = final_pixel_size

    # Save metadata to XML
    tilt_series.save_meta(xml_path)

    print(f"  Created: {xml_path}")
    print(f"  Created: {stack_path}")

    return tilt_series, xml_path


def main():
    """Main processing loop for TS_* folders."""
    import glob

    # Find all TS_* folders
    ts_folders = glob.glob("TS_*")
    ts_folders = [Path(f) for f in ts_folders if os.path.isdir(f)]

    if not ts_folders:
        print("No TS_* folders found in current directory")
        return

    print(f"Found {len(ts_folders)} TS_* folders")

    # Configuration
    tilt_axis_angle = -85.6099  # Adjust as needed
    volume_shape = (342, 480, 100)  # (x, y, z) in pixels
    output_pixel_size = 10.0  # Target pixel size in Angstroms, or None for no downsampling
    # input pixel size is read directly from the .st -> we assume its correctly annotated

    # Process each folder
    for folder in sorted(ts_folders):
        try:
            process_tilt_series(
                folder_path=folder,
                tilt_axis_angle=tilt_axis_angle,
                volume_shape=volume_shape,
                output_pixel_size=output_pixel_size,
            )
        except FileNotFoundError as e:
            print(f"  Skipping: {e}")


if __name__ == "__main__":
    main()

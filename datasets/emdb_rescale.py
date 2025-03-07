#!/usr/bin/env python
"""
EMDB Half Map Processor

This script processes pairs of EMDB half maps by:
1. Finding all unique EMDB codes in a source directory
2. Reading both half maps using mrcfile
3. Averaging the half maps
4. Downsampling to a target resolution using Fourier rescaling
5. Saving the result to a destination directory with proper voxel size annotation
"""

import os
import re
import torch
import mrcfile
import numpy as np
from collections import defaultdict
from torch_fourier_rescale import fourier_rescale_3d


def find_unique_emdb_codes(directory):
    """
    Find all unique EMDB codes in the given directory based on file patterns.

    Parameters
    ----------
    directory : str
        Directory containing EMDB half map files

    Returns
    -------
    dict
        Dictionary mapping EMDB codes to lists of half map files
    """
    pattern = re.compile(r"emd_(\d+)_half_map_\d\.map")
    code_files = defaultdict(list)

    for filename in os.listdir(directory):
        match = pattern.match(filename)
        if match:
            code = match.group(1)
            code_files[code].append(os.path.join(directory, filename))

    return code_files


def process_emdb_maps(input_dir, output_dir, target_spacing=10.0):
    """
    Process EMDB half maps by averaging and downsampling.

    Parameters
    ----------
    input_dir : str
        Directory containing EMDB half map files
    output_dir : str
        Directory to save processed maps
    target_spacing : float, optional
        Target voxel spacing in Angstroms, by default 10.0
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Find all unique EMDB codes
    code_files = find_unique_emdb_codes(input_dir)

    for code, files in code_files.items():
        if len(files) != 2:
            print(
                f"Warning: Expected 2 half maps for code {code}, found {len(files)}. Skipping."
            )
            continue

        # Sort files to ensure consistent order (half_map_1, half_map_2)
        files.sort()

        # Read half maps
        half_maps = []
        voxel_size = None

        for file_path in files:
            with mrcfile.open(file_path) as mrc:
                data = mrc.data
                # Get voxel size (only need to store once)
                if voxel_size is None:
                    voxel_size = mrc.voxel_size
                half_maps.append(torch.tensor(data, dtype=torch.float32))

        # Check if both maps have the same shape
        if half_maps[0].shape != half_maps[1].shape:
            print(
                f"Warning: Half maps for code {code} have different shapes. Skipping."
            )
            continue

        # Average the half maps
        avg_map = (half_maps[0] + half_maps[1]) / 2.0

        # Convert voxel size to tuple of floats
        # Note: MRC voxel size is typically stored as (x, y, z) but torch_fourier_rescale expects (d, h, w)
        # which corresponds to (z, y, x) for 3D volumes
        source_spacing = (float(voxel_size.z), float(voxel_size.y), float(voxel_size.x))

        # Downsample to target resolution
        target_spacing = (target_spacing, target_spacing, target_spacing)
        downsampled_map, new_spacing = fourier_rescale_3d(
            avg_map, source_spacing, target_spacing
        )

        # Convert back to numpy for saving
        downsampled_map_np = downsampled_map.numpy()

        # Output filename
        output_filename = f"emd_{code}_10A.mrc"
        output_path = os.path.join(output_dir, output_filename)

        # Save downsampled map with correct voxel size
        with mrcfile.new(output_path, overwrite=True) as mrc:
            mrc.set_data(downsampled_map_np.astype(np.float32))
            # Set the voxel size - convert from (d,h,w) back to (x,y,z) for MRC format
            # The API expects (x,y,z) corresponding to the last three dimensions in reverse
            mrc.voxel_size = (new_spacing[2], new_spacing[1], new_spacing[0])

        print(
            f"Processed {code}: {output_filename} saved with voxel size {new_spacing}"
        )


if __name__ == "__main__":
    input_directory = "emdb_half_maps"
    output_directory = "emdb"
    target_resolution = 10.0  # Angstroms

    process_emdb_maps(input_directory, output_directory, target_resolution)

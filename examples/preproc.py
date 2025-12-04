#!/usr/bin/env python3
"""
Process tomogram data from TS_* folders.

This script loops through folders with TS_* naming convention,
reads tilt angles from .rawtlt files, loads MRC images, and
reconstructs tomograms using torch-tomogram.
"""

import glob
import os
from pathlib import Path
import torch
import einops
import mrcfile
import numpy as np
from miss_alignment.data.io import save_tomogram_to_pickle
from torch_tomogram import Tomogram


def read_tilt_angles(rawtlt_path):
    """
    Read tilt angles from .rawtlt file.
    
    Parameters
    ----------
    rawtlt_path : str
        Path to the .rawtlt file
        
    Returns
    -------
    list of float
        Tilt angles in degrees
    """
    with open(rawtlt_path, 'r') as f:
        return [float(x.strip()) for x in f.readlines()]


def read_translations_from_prexg(prexg_path):
    with open(prexg_path, 'r') as f:
        data = [ x.strip().split() for x in f.readlines()]
        shifts = [[float(x[5]), float(x[4])] for x in data]
    return shifts


def parse_transform_file(filepath):
    """
    Parse transformation file with 2D matrices and shifts.

    Parameters
    ----------
    filepath : str or Path
        Path to the .xf file

    Returns
    -------
    transforms : torch.Tensor
        Shape (N, 3, 3) homogeneous transformation matrices
    """
    data = np.loadtxt(filepath)
    n_transforms = data.shape[0]

    # Extract rotation matrices (2x2) and shifts
    rot_matrices = data[:, :4].reshape(-1, 2, 2)  # (N, 2, 2)
    shifts = data[:, 4:6]  # (N, 2)

    # Compute inverse rotation matrices
    rot_inv = np.linalg.inv(rot_matrices)  # (N, 2, 2)

    # Apply inverse rotation to shifts
    corrected_shifts = np.einsum('nij,nj->ni', rot_inv, shifts)
    corrected_shifts = np.flip(corrected_shifts, axis=1)

    return torch.tensor(corrected_shifts.copy())



def process_tomogram_folder(folder_path, output_size=(100, 480, 342), n_iterations=128):
    """
    Process a single tomogram folder.
    
    Parameters
    ----------
    folder_path : str
        Path to TS_* folder
    output_size : tuple of int, optional
        Output volume dimensions (depth, height, width)
    n_iterations : int, optional
        Number of reconstruction iterations
        
    Returns
    -------
    torch.Tensor
        Reconstructed tomogram volume
    """
    folder_name = os.path.basename(folder_path)
    
    # File paths
    rawtlt_path = os.path.join(folder_path, f"{folder_name}.rawtlt")
    mrc_path = os.path.join(folder_path, f"{folder_name}.st")
    prexg_path = os.path.join(folder_path, f"{folder_name}.prexg")
    xf_path = os.path.join(folder_path, f"{folder_name}.xf")
    
    # Check if required files exist
    if not os.path.exists(rawtlt_path):
        print(f"Warning: {rawtlt_path} not found, skipping {folder_name}")
        return None
    if not os.path.exists(mrc_path):
        print(f"Warning: {mrc_path} not found, skipping {folder_name}")
        return None
    if not os.path.exists(prexg_path):
        print(f"Warning: {prexg_path} not found skipping {folder_name}")
        return None
    if not os.path.exists(xf_path):
        print(f"Warning: {xf_path} not found skipping {folder_name}")
        return None
    
    print(f"Processing {folder_name}...")
    
    # Read tilt angles
    tilt_angles = read_tilt_angles(rawtlt_path)
    n_tilts = len(tilt_angles)

    # Read sample translations
    sample_translations_prexg = read_translations_from_prexg(prexg_path)
    sample_translations_prexg = torch.tensor(sample_translations_prexg)

    sample_translations_xf = parse_transform_file(xf_path)
    
    # Load tilt images
    tilt_images = torch.from_numpy(mrcfile.read(mrc_path).astype(np.float32))
    print(f"  Tilt series shape: {tilt_images.shape}")
    tilt_images -= einops.reduce(
            tilt_images, "tilt h w -> tilt 1 1", reduction="mean"
        )
    tilt_images /= torch.std(tilt_images, dim=(-2, -1), keepdim=True)
    
    # Fixed tilt axis angle (adjust as needed)
    tilt_axis_angle = -85.6099
    
    # Create Tomogram object
    tilt_series_prexg = Tomogram(
        tilt_angles=tilt_angles,
        tilt_axis_angle=[tilt_axis_angle] * n_tilts,
        sample_translations=-sample_translations_prexg,
        images=tilt_images
    )
    tilt_series_xf = Tomogram(
        tilt_angles=tilt_angles,
        tilt_axis_angle=[tilt_axis_angle] * n_tilts,
        sample_translations=-sample_translations_xf,
        images=tilt_images
    )
    
    # Reconstruct tomogram
    #print(f"  Reconstructing volume with size {output_size}...")
    #volume = tilt_series.reconstruct_tomogram(output_size, n_iterations)
    
    return tilt_series_prexg, tilt_series_xf


def main():
    """Main processing loop."""
    # Find all TS_* folders
    ts_folders = glob.glob("TS_*")
    ts_folders = [f for f in ts_folders if os.path.isdir(f)]
    
    if not ts_folders:
        print("No TS_* folders found in current directory")
        return
    
    print(f"Found {len(ts_folders)} TS_* folders")
    
    # Process each folder
    for folder in sorted(ts_folders):
        tilt_series, tilt_series_xf = process_tomogram_folder(folder)
        
        path = Path(folder)
        name = path.name + '_xf.pickle'
        save_tomogram_to_pickle(tilt_series_xf, path.parent / name)

        name = path.name + '_xf_reconstruction.mrc'
        volume = tilt_series_xf.reconstruct_tomogram((100, 480, 342), 128)
        mrcfile.write(name, volume.numpy(), overwrite=True)


if __name__ == "__main__":
    main()

"""
Compare AreTomo alignments against ground truth alignments for SHREC benchmark.

This script loads AreTomo .aln files and ground truth XML files, applies the
AreTomo alignments, reconstructs volumes, uses cross-correlation to account for
global shifts, and calculates alignment errors.

Usage
-----
Basic usage comparing AreTomo alignments to ground truth:

    python compare_to_ground_truth_aretomo.py \\
        --ground-truth-dir /path/to/ground_truth \\
        --aretomo-dir /path/to/aretomo

With GPU acceleration and output file:

    python compare_to_ground_truth_aretomo.py \\
        --ground-truth-dir /path/to/ground_truth \\
        --aretomo-dir /path/to/aretomo \\
        --device cuda \\
        --output-file aretomo_errors.json

Example
-------
    python examples/shrec/compare_to_ground_truth_aretomo.py \\
        --ground-truth-dir /home/marten/data/download/shrec_zenodo/ground_truth \\
        --aretomo-dir /home/marten/data/datasets/shrec_v2/aretomo \\
        --device cuda

AreTomo File Format
-------------------
The script expects AreTomo .aln files with the following format:
- Line 1-3: Header comments
- Line 4: Column headers (SEC ROT GMAG TX TY SMEAN SFIT SCALE BASE TILT)
- Lines 5+: Alignment data (one row per tilt image)

The TX and TY columns (in pixels) are converted to Angstroms and applied as
tilt_axis_offset_x and tilt_axis_offset_y respectively.

File Matching
-------------
The script matches AreTomo files to ground truth by filename stem:
- model_0.xml matches model_0.aln or model_0.st.aln
- model_1.xml matches model_1.aln or model_1.st.aln
etc.
"""

import argparse
from pathlib import Path
import torch
import numpy as np
from miss_alignment.data.io import TiltSeriesData
from miss_alignment.alignment.correlation import (
    calculate_cross_correlation,
    get_shift_from_correlation_image,
    project_volume_shift_to_image_alignment,
)
from torch_affine_utils.transforms_3d import Ry, Rz


def parse_aretomo_aln_file(aln_file: Path) -> dict:
    """
    Parse an AreTomo .aln file to extract alignment parameters.

    Parameters
    ----------
    aln_file : Path
        Path to AreTomo .aln file

    Returns
    -------
    dict
        Dictionary with keys:
        - 'sections': list of section numbers (0-indexed)
        - 'tx': numpy array of X translations in pixels
        - 'ty': numpy array of Y translations in pixels
        - 'tilt_angles': numpy array of tilt angles in degrees
        - 'rot': numpy array of rotation angles
        - 'gmag': numpy array of global magnifications
    """
    sections = []
    tx_values = []
    ty_values = []
    tilt_angles = []
    rot_values = []
    gmag_values = []

    with open(aln_file, "r") as f:
        for line in f:
            # Skip comment lines and empty lines
            if line.startswith("#") or not line.strip():
                continue

            parts = line.split()
            if len(parts) < 10:
                continue

            # Parse columns according to AreTomo format
            # SEC ROT GMAG TX TY SMEAN SFIT SCALE BASE TILT
            section = int(parts[0])
            rot = float(parts[1])
            gmag = float(parts[2])
            tx = float(parts[3])
            ty = float(parts[4])
            tilt = float(parts[9])

            sections.append(section)
            tx_values.append(tx)
            ty_values.append(ty)
            tilt_angles.append(tilt)
            rot_values.append(rot)
            gmag_values.append(gmag)

    # Sort by section number to match tilt series order
    sorted_indices = np.argsort(sections)

    return {
        "sections": np.array(sections)[sorted_indices],
        "tx": np.array(tx_values)[sorted_indices],
        "ty": np.array(ty_values)[sorted_indices],
        "tilt_angles": np.array(tilt_angles)[sorted_indices],
        "rot": np.array(rot_values)[sorted_indices],
        "gmag": np.array(gmag_values)[sorted_indices],
    }


def apply_aretomo_alignment_to_tilt_series(
    tilt_series_data: TiltSeriesData,
    aretomo_params: dict,
) -> TiltSeriesData:
    """
    Apply AreTomo alignment parameters to a TiltSeries.

    Parameters
    ----------
    tilt_series_data : TiltSeriesData
        Original tilt series data
    aretomo_params : dict
        AreTomo alignment parameters from parse_aretomo_aln_file

    Returns
    -------
    TiltSeriesData
        New TiltSeriesData with AreTomo alignments applied
    """
    # Load the original tilt series
    tilt_series, images, pixel_size = tilt_series_data.load_metadata_and_stack(
        downsample=1
    )

    # Convert TX and TY from pixels to Angstroms
    # AreTomo TX corresponds to tilt_axis_offset_x
    # AreTomo TY corresponds to tilt_axis_offset_y
    tx_angstrom = torch.tensor(aretomo_params["tx"], dtype=torch.float32) * pixel_size
    ty_angstrom = torch.tensor(aretomo_params["ty"], dtype=torch.float32) * pixel_size

    # Apply the AreTomo alignments
    tilt_series.tilt_axis_offset_x = tx_angstrom
    tilt_series.tilt_axis_offset_y = ty_angstrom

    # Save the modified metadata to a temporary XML file
    import tempfile

    # Create temporary directory for modified metadata
    temp_dir = Path(tempfile.mkdtemp())
    temp_xml = temp_dir / "aretomo_alignment.xml"

    # Create new TiltSeriesData pointing to temp file
    aretomo_tilt_series_data = tilt_series_data.replace(
        xml_metadata_path=temp_xml,
    )

    # Save the modified metadata
    aretomo_tilt_series_data.save_metadata_to_xml(tilt_series)

    return aretomo_tilt_series_data, temp_dir


def reconstruct_full_volume(
    tilt_series_data: TiltSeriesData,
    device: str = "cpu",
) -> torch.Tensor:
    """
    Reconstruct a full volume from a tilt series using warpylib.

    Parameters
    ----------
    tilt_series_data : TiltSeriesData
        Tilt series metadata and stack path
    device : str
        Device to use for computation

    Returns
    -------
    torch.Tensor
        Reconstructed volume with shape (Z, Y, X)
    """
    # Load metadata and stack
    tilt_series, images, pixel_size = tilt_series_data.load_metadata_and_stack(
        downsample=1
    )

    # Move to device
    images = images.to(device)
    tilt_series = tilt_series.to(device)

    # grab the volume dimensions from the metadata
    volume_dimensions_physical = tuple(tilt_series.volume_dimensions_physical)

    # Reconstruct using warpylib's tiled weighted backprojection
    volume = tilt_series.reconstruct_full(
        tilt_data=images,
        pixel_size=pixel_size,
        volume_dimensions_physical=volume_dimensions_physical,
        subvolume_size=64,
        subvolume_oversampling=2.0,
        normalize=True,
        invert=False,
        apply_ctf=False,
        ctf_weighted=False,
        correct_attenuation=True,
        batch_size=8,
    )

    return volume


def calculate_alignment_error(
    ground_truth_data: TiltSeriesData,
    aretomo_data: TiltSeriesData,
    device: str = "cpu",
) -> dict:
    """
    Calculate alignment error between ground truth and AreTomo alignments.

    Uses cross-correlation to find global shift between reconstructions,
    then calculates per-image alignment errors.

    Parameters
    ----------
    ground_truth_data : TiltSeriesData
        Ground truth alignment
    aretomo_data : TiltSeriesData
        AreTomo alignment to compare
    device : str
        Device to use

    Returns
    -------
    dict
        Dictionary with error statistics
    """
    print(f"  Loading ground truth: {ground_truth_data.xml_metadata_path.stem}")
    gt_tilt_series, gt_images, gt_pixel_size = ground_truth_data.load_metadata_and_stack(
        downsample=1
    )

    print(f"  Loading AreTomo alignment: {aretomo_data.xml_metadata_path.stem}")
    aretomo_tilt_series, aretomo_images, aretomo_pixel_size = (
        aretomo_data.load_metadata_and_stack(downsample=1)
    )

    # Verify pixel sizes match
    if not torch.isclose(
        torch.tensor(gt_pixel_size), torch.tensor(aretomo_pixel_size), rtol=1e-5
    ):
        raise ValueError(
            f"Pixel sizes don't match: GT={gt_pixel_size}, AreTomo={aretomo_pixel_size}"
        )

    pixel_size = gt_pixel_size

    # Reconstruct volumes
    print("  Reconstructing ground truth volume...")
    gt_volume = reconstruct_full_volume(ground_truth_data, device)

    print("  Reconstructing AreTomo volume...")
    aretomo_volume = reconstruct_full_volume(aretomo_data, device)

    # Calculate cross-correlation to find global shift
    print("  Calculating cross-correlation...")
    correlation = calculate_cross_correlation(gt_volume, aretomo_volume)
    global_shift_voxels = get_shift_from_correlation_image(correlation)
    global_shift_angstrom = global_shift_voxels * pixel_size

    print(
        f"  Global shift (Angstrom): z={global_shift_angstrom[0]:.2f}, "
        f"y={global_shift_angstrom[1]:.2f}, x={global_shift_angstrom[2]:.2f}"
    )

    # Get projection matrices for converting 3D shift to 2D image shifts
    r0 = Ry(-gt_tilt_series.angles, zyx=True)
    r1 = Rz(gt_tilt_series.tilt_axis_angles, zyx=True)
    rotation_matrices = r1 @ r0
    projection_matrices = rotation_matrices[..., 1:3, :3]

    # Project global 3D shift to 2D image shifts
    global_shift_2d = project_volume_shift_to_image_alignment(
        global_shift_angstrom, projection_matrices
    )

    # Get alignment offsets in Angstrom
    # Ground truth offsets
    gt_offsets = torch.stack(
        [gt_tilt_series.tilt_axis_offset_y, gt_tilt_series.tilt_axis_offset_x], dim=1
    )

    # AreTomo offsets
    aretomo_offsets = torch.stack(
        [
            aretomo_tilt_series.tilt_axis_offset_y,
            aretomo_tilt_series.tilt_axis_offset_x,
        ],
        dim=1,
    )

    # Calculate error after correcting for global shift
    # The global_shift_2d needs to be subtracted from AreTomo to align it to GT
    alignment_error = aretomo_offsets - gt_offsets - global_shift_2d

    # Extract Y and X errors separately
    y_errors = alignment_error[:, 0].cpu().numpy()
    x_errors = alignment_error[:, 1].cpu().numpy()

    # Calculate magnitude for display purposes
    error_magnitude = torch.norm(alignment_error, dim=1)

    results = {
        "mean_error_angstrom": error_magnitude.mean().item(),
        "std_error_angstrom": error_magnitude.std().item(),
        "median_error_angstrom": error_magnitude.median().item(),
        "max_error_angstrom": error_magnitude.max().item(),
        "min_error_angstrom": error_magnitude.min().item(),
        "global_shift_angstrom": global_shift_angstrom.cpu().numpy(),
        "y_error_angstrom": y_errors,
        "x_error_angstrom": x_errors,
        "n_tilts": len(error_magnitude),
    }

    return results


def find_matching_aln_file(xml_file: Path, aretomo_dir: Path) -> Path:
    """
    Find the matching AreTomo .aln file for a given XML file.

    Looks for files matching:
    - {stem}.aln
    - {stem}.st.aln

    Parameters
    ----------
    xml_file : Path
        Path to the XML ground truth file
    aretomo_dir : Path
        Directory containing AreTomo .aln files

    Returns
    -------
    Path
        Path to matching .aln file

    Raises
    ------
    FileNotFoundError
        If no matching .aln file is found
    """
    stem = xml_file.stem

    # Try different possible naming conventions
    possible_names = [
        f"{stem}.aln",
        f"{stem}.st.aln",
    ]

    for name in possible_names:
        aln_file = aretomo_dir / name
        if aln_file.exists():
            return aln_file

    raise FileNotFoundError(
        f"No AreTomo .aln file found for {xml_file.name}. "
        f"Tried: {', '.join(possible_names)}"
    )


def main():
    parser = argparse.ArgumentParser(
        description="Compare AreTomo alignments against ground truth"
    )
    parser.add_argument(
        "--ground-truth-dir",
        type=Path,
        required=True,
        help="Directory containing ground truth XML files",
    )
    parser.add_argument(
        "--aretomo-dir",
        type=Path,
        required=True,
        help="Directory containing AreTomo .aln files",
    )
    parser.add_argument(
        "--output-file",
        type=Path,
        default=None,
        help="Optional output file for raw per-image alignment errors (JSON format)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to use (cpu or cuda)",
    )

    args = parser.parse_args()

    # Find all ground truth XML files
    gt_files = sorted(args.ground_truth_dir.glob("*.xml"))
    if not gt_files:
        raise ValueError(f"No XML files found in {args.ground_truth_dir}")

    print(f"Found {len(gt_files)} ground truth files")
    print(f"Using device: {args.device}")

    all_results = []
    temp_dirs_to_cleanup = []

    try:
        for gt_file in gt_files:
            print(f"\nProcessing {gt_file.name}...")

            # Find corresponding AreTomo .aln file
            try:
                aln_file = find_matching_aln_file(gt_file, args.aretomo_dir)
            except FileNotFoundError as e:
                print(f"Warning: {e}, skipping")
                continue

            print(f"  Found AreTomo file: {aln_file.name}")

            # Parse AreTomo alignment file
            aretomo_params = parse_aretomo_aln_file(aln_file)

            # Load ground truth data
            gt_data = TiltSeriesData(xml_metadata_path=gt_file)

            # Apply AreTomo alignment to create a comparable TiltSeriesData
            aretomo_data, temp_dir = apply_aretomo_alignment_to_tilt_series(
                gt_data, aretomo_params
            )
            temp_dirs_to_cleanup.append(temp_dir)

            # Calculate errors
            results = calculate_alignment_error(gt_data, aretomo_data, args.device)

            # Add filename to results
            results["filename"] = gt_file.stem
            all_results.append(results)

            # Print results
            print(f"  Mean error: {results['mean_error_angstrom']:.3f} Å")
            print(f"  Std error:  {results['std_error_angstrom']:.3f} Å")
            print(f"  Median error: {results['median_error_angstrom']:.3f} Å")
            print(f"  Max error:  {results['max_error_angstrom']:.3f} Å")

        # Calculate overall statistics
        if all_results:
            print("\n" + "=" * 70)
            print("Overall Statistics")
            print("=" * 70)

            all_mean_errors = [r["mean_error_angstrom"] for r in all_results]
            print(
                f"Average mean error across all tilt series: {np.mean(all_mean_errors):.3f} Å"
            )
            print(f"Std of mean errors: {np.std(all_mean_errors):.3f} Å")

            # Save raw errors to JSON if requested
            if args.output_file:
                import json

                # Format output: {filename: {x_error_angstrom: [...], y_error_angstrom: [...]}}
                output_data = {}
                for r in all_results:
                    output_data[r["filename"]] = {
                        "x_error_angstrom": r["x_error_angstrom"].tolist(),
                        "y_error_angstrom": r["y_error_angstrom"].tolist(),
                    }

                with open(args.output_file, "w") as f:
                    json.dump(output_data, f, indent=2)

                print(f"\nRaw alignment errors saved to {args.output_file}")

    finally:
        # Cleanup temporary directories
        import shutil

        for temp_dir in temp_dirs_to_cleanup:
            if temp_dir.exists():
                shutil.rmtree(temp_dir)


if __name__ == "__main__":
    main()

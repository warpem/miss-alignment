"""
Compare aligned tilt-series against ground truth alignments for SHREC benchmark.

This script loads both ground truth and predicted alignments, reconstructs volumes,
uses cross-correlation to account for global shifts, and calculates alignment errors.
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
    test_data: TiltSeriesData,
    device: str = "cpu",
) -> dict:
    """
    Calculate alignment error between ground truth and test alignments.

    Uses cross-correlation to find global shift between reconstructions,
    then calculates per-image alignment errors.

    Parameters
    ----------
    ground_truth_data : TiltSeriesData
        Ground truth alignment
    test_data : TiltSeriesData
        Test alignment to compare
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

    print(f"  Loading test data: {test_data.xml_metadata_path.stem}")
    test_tilt_series, test_images, test_pixel_size = test_data.load_metadata_and_stack(
        downsample=1
    )

    # Verify pixel sizes match
    if not torch.isclose(
        torch.tensor(gt_pixel_size), torch.tensor(test_pixel_size), rtol=1e-5
    ):
        raise ValueError(
            f"Pixel sizes don't match: GT={gt_pixel_size}, Test={test_pixel_size}"
        )

    pixel_size = gt_pixel_size

    # Reconstruct volumes
    print("  Reconstructing ground truth volume...")
    gt_volume = reconstruct_full_volume(ground_truth_data, device)

    print("  Reconstructing test volume...")
    test_volume = reconstruct_full_volume(test_data, device)

    # Calculate cross-correlation to find global shift
    print("  Calculating cross-correlation...")
    correlation = calculate_cross_correlation(gt_volume, test_volume)
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

    # Test offsets
    test_offsets = torch.stack(
        [test_tilt_series.tilt_axis_offset_y, test_tilt_series.tilt_axis_offset_x],
        dim=1,
    )

    # Calculate error after correcting for global shift
    # The global_shift_2d needs to be subtracted from test to align it to GT
    alignment_error = test_offsets - gt_offsets - global_shift_2d

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


def main():
    parser = argparse.ArgumentParser(
        description="Compare aligned tilt-series against ground truth"
    )
    parser.add_argument(
        "--ground-truth-dir",
        type=Path,
        required=True,
        help="Directory containing ground truth JSON files",
    )
    parser.add_argument(
        "--test-dir",
        type=Path,
        required=True,
        help="Directory containing test JSON files to compare",
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

    # Find all ground truth JSON files
    gt_files = sorted(args.ground_truth_dir.glob("*.json"))
    if not gt_files:
        raise ValueError(f"No JSON files found in {args.ground_truth_dir}")

    print(f"Found {len(gt_files)} ground truth files")
    print(f"Using device: {args.device}")

    all_results = []

    for gt_file in gt_files:
        # Find corresponding test file
        test_file = args.test_dir / gt_file.name
        if not test_file.exists():
            print(f"Warning: No matching test file for {gt_file.name}, skipping")
            continue

        print(f"\nProcessing {gt_file.name}...")

        # Load data
        gt_data = TiltSeriesData.from_json(gt_file)
        test_data = TiltSeriesData.from_json(test_file)

        # Calculate errors
        results = calculate_alignment_error(
            gt_data, test_data, args.device
        )

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
        print(f"Average mean error across all tilt series: {np.mean(all_mean_errors):.3f} Å")
        print(f"Std of mean errors: {np.std(all_mean_errors):.3f} Å")

        # Save raw errors to JSON if requested
        if args.output_file:
            import json

            # Format output as requested: {filename: {x_error_angstrom: [...], y_error_angstrom: [...]}}
            output_data = {}
            for r in all_results:
                output_data[r["filename"]] = {
                    "x_error_angstrom": r["x_error_angstrom"].tolist(),
                    "y_error_angstrom": r["y_error_angstrom"].tolist(),
                }

            with open(args.output_file, "w") as f:
                json.dump(output_data, f, indent=2)

            print(f"\nRaw alignment errors saved to {args.output_file}")


if __name__ == "__main__":
    main()

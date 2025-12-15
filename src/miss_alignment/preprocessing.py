"""Preprocessing utilities for tilt-series alignment."""

from pathlib import Path

from torch_tiltxcorr import tiltxcorr_with_pretilt_offset

from .data.io import TiltSeriesData


def run_cross_correlation_alignment(
    training_directory: Path,
    device: int = 0,
    lowpass_cutoff: float = 0.25,
    pretilt_search_range: tuple[float, float] = (-30.0, 30.0),
) -> None:
    """
    Run cross-correlation based alignment with pretilt estimation.

    This performs coarse alignment using cross-correlation to estimate shifts
    and sample pretilt (initial tilt offset) for all tilt-series in the training
    directory.

    Parameters
    ----------
    training_directory : Path
        Directory containing XML metadata files for tilt-series.
    device : int, optional
        CUDA device to use for alignment (default: 0).
    lowpass_cutoff : float, optional
        Low-pass filter cutoff frequency (default: 0.25).
    pretilt_search_range : tuple[float, float], optional
        Search range for pretilt estimation in degrees (default: (-30.0, 30.0)).
    """
    # Get list of all XML files to process
    xml_files = list(training_directory.glob("*.xml"))

    if not xml_files:
        raise ValueError(f"No XML files found in {training_directory}")

    print(f"\nRunning cross-correlation alignment on {len(xml_files)} tilt-series...")
    print(
        f"  Pretilt search range: "
        f"{pretilt_search_range[0]}° to {pretilt_search_range[1]}°"
    )
    print(f"  Low-pass cutoff: {lowpass_cutoff}")
    print(f"  Using device: cuda:{device}\n")

    for xml_file in xml_files:
        print(f"Processing {xml_file.name}...", end=" ", flush=True)

        # Load tilt-series data
        ts_data = TiltSeriesData(xml_metadata_path=xml_file)
        ts, stack, pixel_size = ts_data.load_metadata_and_stack(downsample=1)

        # Extract tilt axis angle from metadata (same for all tilts)
        tilt_axis_angle = ts.tilt_axis_angles[0].item()

        # Run cross-correlation alignment with pretilt estimation
        shifts, pretilt = tiltxcorr_with_pretilt_offset(
            stack.to(f"cuda:{device}"),
            ts.angles.to(f"cuda:{device}"),
            tilt_axis_angle,
            lowpass_cutoff,
            pretilt_search_range,
        )

        # Convert shifts from pixels to Angstroms
        shifts_angstrom = shifts * pixel_size

        # Apply pretilt to angles
        ts.angles = ts.angles + pretilt

        # Apply shifts (note: negation and axis assignment)
        # shifts are in YX order, tilt_axis_offset are X and Y
        ts.tilt_axis_offset_x = -shifts_angstrom[:, 1]
        ts.tilt_axis_offset_y = -shifts_angstrom[:, 0]

        # Save updated metadata
        ts_data.save_metadata_to_xml(ts)

        print(f"✓ (pretilt: {pretilt:.2f}°)")

    print("\nCross-correlation alignment complete!\n")

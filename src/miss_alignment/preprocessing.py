"""Preprocessing utilities for tilt-series alignment."""

import queue
import torch
from pathlib import Path

from ._parallel import run_device_pool
from .data.io import TiltSeriesData


def _run_cross_correlation_single(
    xml_file: Path,
    device: int | None,
    lowpass_cutoff: float,
) -> None:
    """Run cross-correlation alignment on a single tilt-series.

    Parameters
    ----------
    xml_file : Path
        Path to the XML metadata file for the tilt-series.
    device : int | None
        CUDA device index to use. If None, uses default device.
    lowpass_cutoff : float
        Low-pass filter cutoff frequency.
    """
    from torch_tiltxcorr import tiltxcorr

    if device is not None and torch.cuda.is_available():
        torch.cuda.set_device(device)
        device_str = f"cuda:{device}"
    else:
        device_str = "cuda"

    # Load tilt-series data
    ts_data = TiltSeriesData(xml_metadata_path=xml_file)
    ts, stack, pixel_size = ts_data.load_metadata_and_stack(downsample=1)

    # Extract tilt axis angle from metadata (same for all tilts)
    tilt_axis_angle = ts.tilt_axis_angles[0].item()

    # Run cross-correlation alignment
    shifts = tiltxcorr(
        tilt_series=stack.to(device_str),
        tilt_angles=ts.angles.to(device_str),
        tilt_axis_angle=tilt_axis_angle,
        pixel_spacing_angstroms=pixel_size,
        lowpass_angstroms=pixel_size / lowpass_cutoff,
    )

    # Convert shifts from pixels to Angstroms
    shifts_angstrom = shifts * pixel_size

    # Apply shifts (note: negation and axis assignment)
    # shifts are in YX order, tilt_axis_offset are X and Y
    ts.tilt_axis_offset_x = -shifts_angstrom[:, 1]
    ts.tilt_axis_offset_y = -shifts_angstrom[:, 0]

    # Save updated metadata
    ts_data.save_metadata_to_xml(ts)


def _cross_correlation_runner(
    device: int | None, task_queue, result_queue, lowpass_cutoff: float
) -> None:
    """Pull tilt-series off the queue and align them on a single device."""

    torch.set_num_threads(1)
    while True:
        try:
            xml_file = task_queue.get_nowait()
        except queue.Empty:
            break
        _run_cross_correlation_single(
            xml_file,
            device,
            lowpass_cutoff,
        )
        result_queue.put_nowait(xml_file.stem)


def run_cross_correlation_alignment_parallel(
    training_directory: Path,
    devices: list[int] | None = None,
    lowpass_cutoff: float = 0.25,
) -> None:
    """
    Run cross-correlation based alignment in parallel.

    This performs coarse alignment using cross-correlation to estimate shifts
    for all tilt-series in the training directory, processing multiple
    tilt-series in parallel across available GPUs.

    Parameters
    ----------
    training_directory : Path
        Directory containing XML metadata files for tilt-series.
    devices : list[int] | None, optional
        CUDA device indices to distribute work across (one worker process per
        unique device). If None, a single default-device worker is used.
    lowpass_cutoff : float, optional
        Low-pass filter cutoff frequency (default: 0.25).
    """
    # Get list of all XML files to process
    xml_files = list(training_directory.glob("*.xml"))

    if not xml_files:
        raise ValueError(f"No XML files found in {training_directory}")

    print(f"\nRunning cross-correlation alignment on {len(xml_files)} tilt-series...")
    print(f"  Low-pass cutoff: {lowpass_cutoff}")
    if devices:
        print(f"  Distributing across devices: {devices}\n")
    else:
        print("  Using default device assignment\n")

    run_device_pool(
        jobs=xml_files,
        runner=_cross_correlation_runner,
        runner_args=(lowpass_cutoff,),
        devices=devices,
        desc="Cross-correlation alignment",
    )

    print("\nCross-correlation alignment complete!\n")

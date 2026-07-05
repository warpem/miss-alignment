"""Preprocessing utilities for tilt-series alignment."""

from pathlib import Path

from .data.io import TiltSeriesData
from .distributed.manager import run_distributed


def _run_cross_correlation_single(
    xml_file: Path,
    device: int | None,
    lowpass_cutoff: float,
    pretilt_search_range: tuple[float, float],
) -> float:
    """Run cross-correlation alignment on a single tilt-series.

    Parameters
    ----------
    xml_file : Path
        Path to the XML metadata file for the tilt-series.
    device : int | None
        CUDA device index to use. If None, uses default device.
    lowpass_cutoff : float
        Low-pass filter cutoff frequency.
    pretilt_search_range : tuple[float, float]
        Search range for pretilt estimation in degrees.

    Returns
    -------
    float
        The estimated pretilt value in degrees.
    """
    import torch
    from torch_tiltxcorr import tiltxcorr_with_sample_tilt_estimation

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

    # Run cross-correlation alignment with pretilt estimation
    shifts, pretilt = tiltxcorr_with_sample_tilt_estimation(
        tilt_series=stack.to(device_str),
        tilt_angles=ts.angles.to(device_str),
        tilt_axis_angle=tilt_axis_angle,
        pixel_spacing_angstroms=pixel_size,
        lowpass_angstroms=pixel_size / lowpass_cutoff,
        sample_tilt_range=pretilt_search_range,
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

    return float(pretilt)


def run_cross_correlation_alignment_parallel(
    training_directory: Path,
    devices: list[int] | None = None,
    lowpass_cutoff: float = 0.25,
    pretilt_search_range: tuple[float, float] = (-30.0, 30.0),
    n_cluster_workers: int | None = None,
) -> None:
    """
    Run cross-correlation based alignment with pretilt estimation in parallel.

    This performs coarse alignment using cross-correlation to estimate shifts
    and sample pretilt (initial tilt offset) for all tilt-series in the training
    directory, processing multiple tilt-series in parallel across available GPUs.

    Parameters
    ----------
    training_directory : Path
        Directory containing XML metadata files for tilt-series.
    devices : list[int] | None, optional
        CUDA device indices to distribute work across (one worker process per
        unique device). If None, a single default-device worker is used.
    lowpass_cutoff : float, optional
        Low-pass filter cutoff frequency (default: 0.25).
    pretilt_search_range : tuple[float, float], optional
        Search range for pretilt estimation in degrees (default: (-30.0, 30.0)).
    n_cluster_workers : int | None
        Number of cluster jobs to submit. When set, activates cluster mode;
        requires MISS_CLUSTER_CONFIG and MISS_CLUSTER_SCRIPT to be set.
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
    if devices:
        print(f"  Distributing across devices: {devices}\n")
    else:
        print("  Using default device assignment\n")

    queue_root = training_directory / "tasks"
    run_distributed(
        tilt_series_list=xml_files,
        model_checkpoint=Path(""),
        output_directory=training_directory,
        setting="",
        patch_size=0,
        patch_overlap=0.0,
        batch_size=0,
        apply_ctf=False,
        downsample=1,
        devices=devices or [],
        n_cluster_workers=n_cluster_workers,
        queue_root=queue_root,
        task_type="cross_correlation",
        lowpass_cutoff=lowpass_cutoff,
        pretilt_search_range=pretilt_search_range,
    )

    print("\nCross-correlation alignment complete!\n")

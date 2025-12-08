"""Stack preparation utilities for tilt series.

This module provides functionality to load raw tilt images and create
preprocessed tilt stacks ready for training.
"""

from concurrent.futures import ProcessPoolExecutor
from functools import partial
from pathlib import Path

import mrcfile
from tqdm import tqdm
from warpylib import TiltSeries
from warpylib.movie import Movie


def _get_original_pixel_size(tilt_series: TiltSeries) -> float:
    """Get original pixel size from the first tilt image's MRC header.

    Parameters
    ----------
    tilt_series : TiltSeries
        The tilt series object containing movie paths and metadata.

    Returns
    -------
    float
        Original pixel size in Angstroms.

    Raises
    ------
    FileNotFoundError
        If the first tilt image is not found.
    ValueError
        If pixel size cannot be determined from the header.
    """
    if len(tilt_series.tilt_movie_paths) == 0:
        raise ValueError("Tilt series has no movie paths")

    first_tilt_path = tilt_series.tilt_movie_paths[0]
    full_path = str(
        Path(tilt_series.data_directory_name or tilt_series.processing_directory_name)
        / first_tilt_path
    )
    movie = Movie(path=full_path)
    average_path = movie.average_path

    if not Path(average_path).exists():
        raise FileNotFoundError(f"Average image not found: {average_path}")

    with mrcfile.open(average_path, mode="r", permissive=True) as mrc:
        voxel_size = mrc.voxel_size
        # voxel_size is a numpy recarray with x, y, z fields
        pixel_size = float(voxel_size.x)

    if pixel_size <= 0:
        raise ValueError(
            f"Invalid pixel size {pixel_size} in MRC header for {average_path}"
        )

    return pixel_size


def _prepare_single_tilt_series(xml_path: Path, desired_pixel_size: float) -> None:
    """Load raw tilt images and create tilt stack for a single tilt series.

    Parameters
    ----------
    xml_path : Path
        Path to the tilt series XML metadata file.
    desired_pixel_size : float
        Desired pixel size in Angstroms for the output stack.

    Raises
    ------
    FileNotFoundError
        If required image files are not found.
    ValueError
        If pixel size cannot be determined or other validation fails.
    """
    ts = TiltSeries(xml_path)
    original_pixel_size = _get_original_pixel_size(ts)

    images, _, _ = ts.load_images(
        original_pixel_size=original_pixel_size,
        desired_pixel_size=desired_pixel_size,
        use_denoised=False,
        load_averages=True,
        load_half_averages=False,
    )

    ts.stack_tilts(
        tilt_data=images,
        pixel_size=desired_pixel_size,
        create_thumbnails=True,
    )


def prepare_stacks_parallel(
    training_directory: Path,
    desired_pixel_size: float,
    n_processes: int = 4,
) -> None:
    """Prepare tilt stacks for all tilt series in the training directory.

    This function loads raw tilt images for each tilt series, rescales them
    to the desired pixel size, and creates aligned tilt stacks with thumbnails.

    Parameters
    ----------
    training_directory : Path
        Directory containing tilt series XML files.
    desired_pixel_size : float
        Desired pixel size in Angstroms for the output stacks.
    n_processes : int
        Number of parallel processes to use. Default is 4.

    Raises
    ------
    FileNotFoundError
        If no XML files are found in the training directory.
    RuntimeError
        If any tilt series fails to process (terminates on first error).
    """
    xml_files = list(training_directory.glob("*.xml"))
    if not xml_files:
        raise FileNotFoundError(
            f"No XML files found in training directory: {training_directory}"
        )

    print(
        f"Preparing stacks for {len(xml_files)} tilt series at {desired_pixel_size} Å"
    )

    # Use partial to bind the desired_pixel_size argument
    process_func = partial(
        _prepare_single_tilt_series, desired_pixel_size=desired_pixel_size
    )

    with ProcessPoolExecutor(max_workers=n_processes) as executor:
        # Submit all tasks and wrap with tqdm for progress
        futures = [executor.submit(process_func, xml_path) for xml_path in xml_files]

        for future in tqdm(futures, desc="Preparing stacks", unit="tilt series"):
            # This will raise any exception that occurred in the worker
            future.result()

    print(f"Successfully prepared stacks for {len(xml_files)} tilt series")

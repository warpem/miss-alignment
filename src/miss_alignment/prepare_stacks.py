"""Stack preparation utilities for tilt series.

This module provides functionality to load raw tilt images and create
preprocessed tilt stacks ready for training.
"""

import multiprocessing as mp
import queue
import sys
import time
from pathlib import Path

import mrcfile
import tqdm
from warpylib import TiltSeries
from warpylib.movie import Movie


def _run_device_pool(jobs, runner, runner_args, devices, desc):
    """Minimal one-process-per-GPU work queue. Internal to prepare_stacks."""
    ctx = mp.get_context("spawn")
    device_slots = sorted(set(devices)) if devices else [None]

    with ctx.Manager() as manager:
        task_queue = manager.Queue()
        result_queue = manager.Queue()
        for job in jobs:
            task_queue.put_nowait(job)

        procs = [
            ctx.Process(
                target=runner, args=(device, task_queue, result_queue, *runner_args)
            )
            for device in device_slots
        ]
        [p.start() for p in procs]

        results = []
        pbar = tqdm.tqdm(total=len(jobs), desc=desc, file=sys.stdout)
        while len(results) < len(jobs):
            while not result_queue.empty():
                results.append(result_queue.get_nowait())
                pbar.update(1)
            for p in procs:
                if not p.is_alive() and p.exitcode != 0:
                    for x in procs:
                        x.terminate()
                    for x in procs:
                        x.join(timeout=5.0)
                    pbar.close()
                    raise RuntimeError(
                        f"A worker process for '{desc}' stopped unexpectedly."
                    )
            time.sleep(0.1)
        pbar.close()
        [p.join() for p in procs]

    return results


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


def _prepare_single_tilt_series(
    xml_path: Path, desired_pixel_size: float, device: int | None
) -> None:
    """Load raw tilt images and create tilt stack for a single tilt series.

    Parameters
    ----------
    xml_path : Path
        Path to the tilt series XML metadata file.
    desired_pixel_size : float
        Desired pixel size in Angstroms for the output stack.
    device : int | None
        CUDA device index to use for GPU operations. If None, uses CPU.

    Raises
    ------
    FileNotFoundError
        If required image files are not found.
    ValueError
        If pixel size cannot be determined or other validation fails.
    """
    import torch

    if device is not None and torch.cuda.is_available():
        torch.cuda.set_device(device)

    ts = TiltSeries(xml_path)
    original_pixel_size = _get_original_pixel_size(ts)
    # print(f"{xml_path.stem}: original pixel size = {original_pixel_size:.4f} Å")

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


def _prepare_stacks_runner(
    device: int | None,
    task_queue,
    result_queue,
    desired_pixel_size: float,
) -> None:
    """Pull tilt-series off the queue and prepare them on a single device."""
    import torch

    torch.set_num_threads(1)
    while True:
        try:
            xml_path = task_queue.get_nowait()
        except queue.Empty:
            break
        _prepare_single_tilt_series(xml_path, desired_pixel_size, device)
        result_queue.put_nowait(xml_path.stem)


def prepare_stacks_parallel(
    training_directory: Path,
    desired_pixel_size: float,
    devices: list[int] | None = None,
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
    devices : list[int] | None
        CUDA device indices to distribute work across (one worker process per
        unique device). If None, a single default-device worker is used.

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

    _run_device_pool(
        jobs=xml_files,
        runner=_prepare_stacks_runner,
        runner_args=(desired_pixel_size,),
        devices=devices,
        desc="Preparing stacks",
    )

    print(f"Successfully prepared stacks for {len(xml_files)} tilt series")

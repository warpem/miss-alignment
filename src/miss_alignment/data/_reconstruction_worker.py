import multiprocessing as mp
import random
import tempfile
import itertools
from collections.abc import Callable
from pathlib import Path
from typing import Optional
import torch
import pickle
import einops
import os
from torch_affine_utils.transforms_3d import Ry, Rz

from miss_alignment.data.io import read_tomogram_from_pickle
from miss_alignment.data.shift_generation import (
    project_shifts_3d_to_2d
)
from ._pool_monitor import SimplePoolMonitor


def reconstruction_worker(
        worker_id: int,
        assigned_indices: list,
        pool_dir: Path,
        tilt_series_pickles: list[Path],
        tomogram_shape: tuple[int, int, int],
        patch_size: int,
        shift_generator: Callable,
        ready_flag: mp.Value,
        stop_event: mp.Event,
        monitor: Optional[SimplePoolMonitor] = None,
):
    """
    Worker process that maintains a subset of the reconstruction pool.

    Parameters
    ----------
    worker_id : int
        ID of this worker
    assigned_indices : list
        Pool indices assigned to this worker
    pool_dir : Path
        Directory to store reconstructions
    tilt_series_pickles : list[Path]
        List of paths to tilt-series stored as pickles to make
         reconstructions from
    tomogram_shape : tuple[int, int, int]
        Reconstruction shape
    patch_size : int
        Shape of patches will be: (patch_size, ) * 3
    shift_generator : Callable
        Function that generates random sets of shifts in 3D
    ready_flag : mp.Value
        Shared flag indicating pool is ready
    stop_event : mp.Event
        Event to signal worker shutdown
    """
    torch.set_num_threads(1)
    print(
        f"Worker {worker_id} starting with indices {assigned_indices[:5]}..."
    )
    # torch.set_num_threads(1)

    # Initial fill of assigned pool slots
    for idx in assigned_indices:
        if stop_event.is_set():
            print(f'Worker {worker_id} shutting down')

        data_and_labels = _create_pool_reconstruction(
            tilt_series_path=random.choice(tilt_series_pickles),
            tomogram_shape=tomogram_shape,
            patch_size=patch_size,
            shift_generator=shift_generator,
        )

        # Write directly first time (no temp file needed)
        file_path = pool_dir / f"recon_{idx}.pickle"
        with open(file_path, "wb") as outfile:
            pickle.dump(data_and_labels, outfile)
        # ensure correct permissions
        os.chmod(file_path, 0o644)

    print(f"Worker {worker_id} completed initial fill")

    # Signal ready after initial fill
    with ready_flag.get_lock():
        ready_flag.value += 1

    # Create cyclic iterator outside the loop
    idx_cycle = itertools.cycle(assigned_indices)

    # Continuously update pool with new reconstructions
    while not stop_event.is_set():
        # Randomly select one of our indices to update
        idx = next(idx_cycle)

        data_and_labels = _create_pool_reconstruction(
            tilt_series_path=random.choice(tilt_series_pickles),
            tomogram_shape=tomogram_shape,
            patch_size=patch_size,
            shift_generator=shift_generator,
        )

        # Atomic replacement using temp file and rename
        file_path = pool_dir / f"recon_{idx}.pickle"
        with tempfile.NamedTemporaryFile(
                dir=pool_dir,
                prefix=f"tmp_recon_{idx}_",
                suffix='.pickle',
                delete=False
        ) as tmp_file:
            tmp_path = Path(tmp_file.name)
            pickle.dump(data_and_labels, tmp_file)

        # fix permissions
        os.chmod(tmp_path, 0o644)
        # Atomic rename (replaces existing file)
        tmp_path.rename(file_path)

        # Record production
        if monitor is not None:
            monitor.record_production()

    print(f"Worker {worker_id} shutting down")


def _create_pool_reconstruction(
        tilt_series_path: Path,
        tomogram_shape: tuple[int, int, int],
        patch_size: int,
        shift_generator: Callable,
) -> list[tuple[torch.Tensor, int]]:
    # load tilt_series from disk
    tilt_series = read_tomogram_from_pickle(tilt_series_path)
    tilt_series.images -= einops.reduce(
        tilt_series.images, "tilt h w -> tilt 1 1", reduction="mean"
    )
    tilt_series.images /= torch.std(tilt_series.images, dim=(-2, -1), keepdim=True)

    # store original alignment
    sample_translations = tilt_series.sample_translations.clone()

    # add a random rotation to the sample
    tilt_series.tilt_angles += random.uniform(-10, +10)

    # select a random reconstruction position
    _offset = patch_size // 2
    _region = [x // 2 - _offset for x in tomogram_shape]
    reconstruction_location = (
        torch.tensor([random.randint(-r, r) for r in _region])
    )

    # generate aligned and misaligned shifts
    r0 = Ry(tilt_series.tilt_angles, zyx=True)
    r1 = Rz(tilt_series.tilt_axis_angle, zyx=True)
    rotation_matrices = r1 @ r0
    projection_matrices = rotation_matrices[..., 1:3, :3]
    aligned_shifts, misaligned_shifts = (
        _generate_translations(shift_generator, projection_matrices)
    )

    # set the translations needed for reconstruction
    aligned_translations = sample_translations + aligned_shifts
    misaligned_translations = sample_translations + misaligned_shifts

    # reconstruct both volumes
    tilt_series.sample_translations = aligned_translations
    aligned = tilt_series.reconstruct_subvolume(
        reconstruction_location, patch_size
    )
    tilt_series.sample_translations = misaligned_translations
    misaligned = tilt_series.reconstruct_subvolume(
            reconstruction_location, patch_size
    )

    # make tuple with volume and label
    examples = [(aligned, 1), (misaligned, -1)]

    # make a triplet example randomly mimick 1 or 2
    examples += [(aligned.clone(), 1) if random.random() > 0.5 else (
        misaligned.clone(), -1)]

    # create triplet with additional rotations
    # tilt_series.tilt_angles += random.uniform(-5, +5)
    # if random.random() > 0.5:  # create aligned triplet
    #     tilt_series.sample_translations = aligned_translations
    #     example3 = tilt_series.reconstruct_subvolume(
    #         reconstruction_location, patch_size
    #     )
    #     examples += [(example3, 1)]
    # else:  # create misaligned triplet
    #     tilt_series.sample_translations = misaligned_translations
    #     example3 = tilt_series.reconstruct_subvolume(
    #         reconstruction_location, patch_size
    #     )
    #     examples += [(example3, -1)]

    return examples


def _generate_translations(
        generate_shifts: Callable,
        rotation_matrices,
) -> tuple[torch.Tensor, torch.Tensor]:
    n_tilts, _, _ = rotation_matrices.shape
    # generate two sets of shifts, 3d shifts zyx
    shifts_1 = generate_shifts(n_tilts)
    shifts_2 = generate_shifts(n_tilts)

    # project the shifts to 2d
    shifts_1 = project_shifts_3d_to_2d(shifts_1, rotation_matrices)
    shifts_2 = project_shifts_3d_to_2d(shifts_2, rotation_matrices)

    # randomly remove x or y shifts
    die_roll = random.random()
    if die_roll < 0.25:  # set y to 0
        shifts_1[:, 0] = 0.0
        shifts_2[:, 0] = 0.0
    if 0.25 <= die_roll < 0.5:  # set x to 0
        shifts_1[:, 1] = 0.0
        shifts_2[:, 1] = 0.0

    # set (mis)alignment based on the \sigma of the generated shifts
    if torch.sum(torch.abs(shifts_1)) > torch.sum(torch.abs(shifts_2)):
        # if torch.pow(shifts_1, 2).sum() > torch.pow(shifts_2, 2).sum():
        misaligned_translations = shifts_1
        aligned_translations = shifts_2
    else:
        misaligned_translations = shifts_2
        aligned_translations = shifts_1

    return aligned_translations, misaligned_translations

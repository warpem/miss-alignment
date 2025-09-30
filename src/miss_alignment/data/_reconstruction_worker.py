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
from torch_tomogram import Tomogram
from dataclasses import dataclass

from miss_alignment.data.io import read_tomogram_from_pickle
from miss_alignment.data.shift_generation import (
    project_shifts_3d_to_2d
)
from ._pool_monitor import SimplePoolMonitor


@dataclass
class TiltSeriesFetcher:
    tilt_series_pickles: list[Path]
    refresh_rate: int
    device: torch.device
    _counter: int = 0
    _tilt_series: Optional[Tomogram] = None

    # temp storage of shifts, tilt angles, and tilt axis angle
    _tmp_sample_translations: torch.Tensor = None
    _tmp_tilt_angles: torch.Tensor = None
    _tmp_tilt_axis_angle: torch.Tensor = None

    def _backup_alignment(self):
        self._tmp_sample_translations = (
            self._tilt_series.sample_translations.clone())
        self._tmp_tilt_angles = self._tilt_series.tilt_angles.clone()
        self._tmp_tilt_axis_angle = self._tilt_series.tilt_axis_angle.clone()

    def _refresh_alignment(self):
        self._tilt_series.sample_translations = (
            self._tmp_sample_translations.clone())
        self._tilt_series.tilt_angles = self._tmp_tilt_angles.clone()
        self._tilt_series.tilt_axis_angle = self._tmp_tilt_axis_angle.clone()

    def _load_next(self):
        tilt_series = (
            read_tomogram_from_pickle(random.choice(self.tilt_series_pickles))
        )
        tilt_series.to(self.device)
        # ensure normalization per tilt
        tilt_series.images -= einops.reduce(
            tilt_series.images, "tilt h w -> tilt 1 1", reduction="mean"
        )
        tilt_series.images /= torch.std(tilt_series.images, dim=(-2, -1),
                                        keepdim=True)
        self._tilt_series = tilt_series

    def __call__(self) -> Tomogram:
        if self._tilt_series is None or self._counter >= self.refresh_rate:
            self._load_next()
            self._backup_alignment()
            self._counter = 1
            return self._tilt_series
        else:
            self._counter += 1
            self._refresh_alignment()
            return self._tilt_series


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
        tilt_series_refresh_rate: int = 10,
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
    monitor : Optional[SimplePoolMonitor], default None
        Pool monitor instance
    tilt_series_refresh_rate: int, default 10
        Number of times a tilt-series is reused before loading a new one,
         prevents repeatedly loading from disk.
    """
    torch.set_num_threads(1)
    print(
        f"Worker {worker_id} starting with indices {assigned_indices[:5]}..."
    )

    # counter to keep track of how many times reconstructions have been reused
    tilt_series_fetcher = TiltSeriesFetcher(
        tilt_series_pickles=tilt_series_pickles,
        refresh_rate=tilt_series_refresh_rate,
        device=torch.device("cpu"),
    )

    # Initial fill of assigned pool slots
    for idx in assigned_indices:
        if stop_event.is_set():
            print(f'Worker {worker_id} shutting down')

        data_and_labels = _create_pool_reconstruction(
            tilt_series=tilt_series_fetcher(),
            tomogram_shape=tomogram_shape,  # type: tuple[int, int, int]
            patch_size=patch_size,
            shift_generator=shift_generator,
        )

        # Write directly first time (no temp file needed)
        file_path = pool_dir / f"recon_{idx}.pickle"
        with open(file_path, "wb") as outfile:
            pickle.dump(data_and_labels, outfile)
        # ensure correct permissions
        os.chmod(file_path, 0o644)

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
            tilt_series=tilt_series_fetcher(),
            tomogram_shape=tomogram_shape,  # type: tuple[int, int, int]
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


def _create_pool_reconstruction(
        tilt_series: Tomogram,
        tomogram_shape: tuple[int, int, int],
        patch_size: int,
        shift_generator: Callable,
) -> list[tuple[torch.Tensor, int]]:
    # add a random rotation to the sample
    # TODO instead of full tilt angle offset we should use a subtomo rotation
    tilt_series.tilt_angles += random.uniform(-10, +10)

    # select a random reconstruction position
    d, h, w = tomogram_shape
    _offset = patch_size // 2
    _region = [d // 2 - _offset, h // 2 - _offset, w // 2 - _offset]
    reconstruction_location = (
        torch.tensor([random.randint(-r, r) for r in _region])
    )

    # generate aligned and misaligned shifts
    r0 = Ry(tilt_series.tilt_angles, zyx=True)
    r1 = Rz(tilt_series.tilt_axis_angle, zyx=True)
    rotation_matrices = r1 @ r0
    projection_matrices = rotation_matrices[..., 1:3, :3]
    translations = (
        _generate_translations(shift_generator, projection_matrices)
    )

    # reconstruct both volumes
    aligned = tilt_series.reconstruct_subvolume(
        reconstruction_location, patch_size
    )
    # add the extra translations for misaligned example
    tilt_series.sample_translations += translations
    misaligned = tilt_series.reconstruct_subvolume(
            reconstruction_location, patch_size
    )

    # make tuple with volume and label
    examples = [(aligned, 1), (misaligned, -1)]

    # make a triplet example randomly mimick 1 or 2
    examples += [(aligned.clone(), 1) if random.random() > 0.5 else (
        misaligned.clone(), -1)]

    return examples


def _generate_translations(
        generate_shifts: Callable,
        projection_matrices,
) -> torch.Tensor:
    n_tilts, y, x = projection_matrices.shape
    if (y, x) != (2, 3):
        raise ValueError('Projection matrices must have shape (2, 3)')
    # generate two sets of shifts, 3d shifts zyx
    shifts = generate_shifts(n_tilts)
    shifts = project_shifts_3d_to_2d(shifts, projection_matrices)

    # randomly remove x or y shifts
    die_roll = random.random()
    if die_roll < 0.25:  # set y to 0
        shifts[:, 0] = 0.0
    elif 0.25 <= die_roll < 0.5:  # set x to 0
        shifts[:, 1] = 0.0

    return shifts

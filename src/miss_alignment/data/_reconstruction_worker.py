import multiprocessing as mp
import random
import tempfile
import itertools
import math
from collections.abc import Callable
from pathlib import Path
from typing import Optional
import torch
import numpy as np
import einops
import os
from torch_affine_utils.transforms_3d import Ry, Rz
from warpylib import TiltSeries
from warpylib.tilt_series.reconstruct_volume import preprocess_tilt_data
from dataclasses import dataclass

from miss_alignment.data.io import TiltSeriesData
from miss_alignment.data.shift_generation import project_shifts_3d_to_2d
from ._pool_monitor import SimplePoolMonitor

# augmentation parameter
MAX_ANGLE_DEGREES = 10.0


def _normalize(volume: torch.Tensor) -> torch.Tensor:
    """Normalize volume to zero mean and unit standard deviation."""
    mean, std = torch.mean(volume), torch.std(volume)
    if std == 0.0:
        raise ValueError(
            "Cannot normalize patch because the standard deviation is 0."
        )
    return (volume - mean) / std


@dataclass
class TiltSeriesFetcher:
    tilt_series_jsons: list[Path]
    patch_size: int  # needed for filter calculation in warpylib
    refresh_rate: int
    downsample: int
    device: str | torch.device
    cache_dir: Optional[Path] = None  # directory for caching preprocessed stacks
    _counter: int = 0
    _tilt_series: Optional[TiltSeries] = None
    _images: Optional[torch.Tensor] = None
    _pixel_size: Optional[float] = None

    # temp storage of shifts, tilt angles, and tilt axis angle
    _tmp_angles: torch.Tensor = None
    _tmp_tilt_axis_angles: torch.Tensor = None
    _tmp_tilt_axis_offset_x: torch.Tensor = None
    _tmp_tilt_axis_offset_y: torch.Tensor = None

    def _backup_alignment(self):
        self._tmp_angles = self._tilt_series.angles.clone()
        self._tmp_tilt_axis_angles = self._tilt_series.tilt_axis_angles.clone()
        self._tmp_tilt_axis_offset_x = self._tilt_series.tilt_axis_offset_x.clone()
        self._tmp_tilt_axis_offset_y = self._tilt_series.tilt_axis_offset_y.clone()

    def _refresh_alignment(self):
        self._tilt_series.angles = self._tmp_angles.clone()
        self._tilt_series.tilt_axis_angles = self._tmp_tilt_axis_angles.clone()
        self._tilt_series.tilt_axis_offset_x = self._tmp_tilt_axis_offset_x.clone()
        self._tilt_series.tilt_axis_offset_y = self._tmp_tilt_axis_offset_y.clone()

    def _load_next(self):
        tilt_series_json = random.choice(self.tilt_series_jsons)
        tilt_series_data = TiltSeriesData.from_json(tilt_series_json)

        # Generate cache filename with downsample factor
        if self.cache_dir is not None:
            tilt_series_name = tilt_series_json.stem  # filename without extension
            cache_filename = f"preprocessed_{tilt_series_name}_ds{self.downsample}.npz"
            cache_path = self.cache_dir / cache_filename
        else:
            cache_path = None

        # Try to load from cache first
        if cache_path is not None and cache_path.exists():
            # Load preprocessed images from cache
            cached_data = np.load(cache_path)
            images = torch.from_numpy(cached_data['images'])
            pixel_size = float(cached_data['pixel_size'])
            # Load tilt series metadata (lightweight, no image data)
            tilt_series, _, _ = tilt_series_data.load_metadata_and_stack(
                downsample=self.downsample
            )
        else:
            # Load and preprocess from scratch
            tilt_series, images, pixel_size = tilt_series_data.load_metadata_and_stack(
                downsample=self.downsample
            )
            # run the preprocessing from warp for consistency
            images = preprocess_tilt_data(
                tilt_data=images,
                normalize=True,
                invert=False,
                subvolume_size=self.patch_size,
            )

            # Save to cache for future use
            if cache_path is not None:
                np.savez(
                    cache_path,
                    images=images.cpu().numpy(),
                    pixel_size=pixel_size
                )
                os.chmod(cache_path, 0o644)

        tilt_series = tilt_series.to(self.device)
        images = images.to(self.device)
        self._tilt_series = tilt_series
        self._images = images
        self._pixel_size = pixel_size

    def __call__(self) -> tuple[TiltSeries, torch.Tensor, float]:
        if self._tilt_series is None or self._counter >= self.refresh_rate:
            self._load_next()
            self._backup_alignment()
            self._counter = 1
        else:
            self._counter += 1
            self._refresh_alignment()
        return self._tilt_series, self._images, self._pixel_size


def reconstruction_worker(
    worker_id: int,
    assigned_indices: list,
    pool_dir: Path,
    tilt_series_jsons: list[Path],
    patch_size: int,
    apply_ctf: bool,
    downsample: int,
    shift_generator: Callable,
    ready_flag: mp.Value,
    stop_event: mp.Event,
    monitor: Optional[SimplePoolMonitor] = None,
    tilt_series_refresh_rate: int = 1,
    device: str | torch.device = "cpu",
    num_workers: Optional[int] = None,
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
    tilt_series_jsons : list[Path]
        List of paths to tilt-series stored as jsons to make
         reconstructions from
    patch_size : int
        Shape of patches will be: (patch_size, ) * 3
    apply_ctf : bool
        Do CTF correction during reconstruction
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
    device: str | torch.Device, default 'cpu'
        Device to use
    """
    torch.set_num_threads(1)

    print(f"Worker {worker_id} starting with indices {assigned_indices[:5]}...")

    # Divide tilt series among workers to avoid duplicate preprocessing
    # Each worker gets a subset of tilt series to process (round-robin)
    if num_workers is None:
        # Fallback: use all tilt series if num_workers not provided
        assigned_tilt_series = tilt_series_jsons
    else:
        assigned_tilt_series = [ts for i, ts in enumerate(tilt_series_jsons) if i % num_workers == worker_id]

    if not assigned_tilt_series:
        # Fallback: if division resulted in empty list, use all tilt series
        assigned_tilt_series = tilt_series_jsons

    print(f"Worker {worker_id} assigned {len(assigned_tilt_series)}/{len(tilt_series_jsons)} tilt series")

    # counter to keep track of how many times reconstructions have been reused
    tilt_series_fetcher = TiltSeriesFetcher(
        tilt_series_jsons=assigned_tilt_series,
        patch_size=patch_size,
        refresh_rate=tilt_series_refresh_rate,
        downsample=downsample,
        device=device,
        cache_dir=pool_dir,  # use same directory for caching preprocessed stacks
    )

    # Initial fill of assigned pool slots
    for idx in assigned_indices:
        if stop_event.is_set():
            print(f"Worker {worker_id} shutting down")
        tilt_series, images, pixel_size = tilt_series_fetcher()
        data_and_labels = _create_pool_reconstruction(
            tilt_series=tilt_series,
            images=images,
            pixel_size=pixel_size,
            patch_size=patch_size,
            shift_generator=shift_generator,
            apply_ctf=apply_ctf,
            device=device,
        )

        # Write directly first time (no temp file needed)
        # np.savez adds .npz extension automatically
        file_path = pool_dir / f"recon_{idx}"
        np.savez(file_path, **data_and_labels)
        # ensure correct permissions
        os.chmod(f"{file_path}.npz", 0o644)

    # Signal ready after initial fill
    with ready_flag.get_lock():
        ready_flag.value += 1

    # Create cyclic iterator outside the loop
    idx_cycle = itertools.cycle(assigned_indices)

    # Continuously update pool with new reconstructions
    while not stop_event.is_set():
        # Randomly select one of our indices to update
        idx = next(idx_cycle)
        tilt_series, images, pixel_size = tilt_series_fetcher()
        data_and_labels = _create_pool_reconstruction(
            tilt_series=tilt_series,
            images=images,
            pixel_size=pixel_size,
            patch_size=patch_size,
            shift_generator=shift_generator,
            apply_ctf=apply_ctf,
            device=device,
        )

        # Atomic replacement using temp file and rename
        # Create temp file and save with npz
        tmp_fd, tmp_path_str = tempfile.mkstemp(
            dir=pool_dir, prefix=f"tmp_recon_{idx}_", suffix=""
        )
        os.close(tmp_fd)  # Close the file descriptor
        tmp_path = Path(tmp_path_str)

        np.savez(tmp_path, **data_and_labels)
        # np.savez adds .npz extension
        tmp_path_npz = Path(f"{tmp_path}.npz")

        # fix permissions
        os.chmod(tmp_path_npz, 0o644)
        # Atomic rename (replaces existing file)
        final_path = pool_dir / f"recon_{idx}.npz"
        tmp_path_npz.rename(final_path)

        # Record production
        if monitor is not None:
            monitor.record_production()


def _create_pool_reconstruction(
    tilt_series: TiltSeries,
    images: torch.Tensor,
    pixel_size: float,
    patch_size: int,
    shift_generator: Callable,
    apply_ctf: bool = True,
    device: str | torch.device = "cpu",
) -> list[tuple[torch.Tensor, int]]:
    # Generate random rotation as Euler angles (ZYZ convention) for data augmentation
    # This rotation is applied to change the coordinate system of the reconstruction
    random_rotation = (random.random() * 2 - 1) * MAX_ANGLE_DEGREES * math.pi / 180
    rotation_angles = torch.tensor([0.0, random_rotation, 0.0], device=device)

    # select a random reconstruction position
    patch_offset = (patch_size * pixel_size) / 2
    x_dim, y_dim, z_dim = tilt_series.volume_dimensions_physical
    reconstruction_location = [
        random.uniform(patch_offset, x_dim - patch_offset)
        if x_dim > 2 * patch_offset
        else x_dim / 2,
        random.uniform(patch_offset, y_dim - patch_offset)
        if y_dim > 2 * patch_offset
        else y_dim / 2,
        random.uniform(patch_offset, z_dim - patch_offset)
        if z_dim > 2 * patch_offset
        else z_dim / 2,
    ]
    reconstruction_location = torch.tensor(reconstruction_location, device=device)
    reconstruction_location = einops.rearrange(reconstruction_location, "xyz -> 1 xyz")

    # generate a misalignment
    r0 = Ry(tilt_series.angles, zyx=True)
    r1 = Rz(tilt_series.tilt_axis_angles, zyx=True)
    rotation_matrices = r1 @ r0
    projection_matrices = rotation_matrices[..., 1:3, :3]
    translations = _generate_translations(
        shift_generator, projection_matrices, device=device
    )
    # express translations in angstrom
    translations_angstrom = translations * pixel_size

    # reconstruct aligned example with random rotation
    aligned = tilt_series.reconstruct_subvolumes_single(
        tilt_data=images,
        coords=reconstruction_location,
        pixel_size=pixel_size,
        size=patch_size,
        apply_ctf=apply_ctf,
        angles=rotation_angles,
        oversampling=2.0,
    ).squeeze()
    # add the extra translations for the misaligned example
    tilt_series.tilt_axis_offset_y += translations_angstrom[:, 0]
    tilt_series.tilt_axis_offset_x += translations_angstrom[:, 1]
    # reconstruct misaligned example with the same rotation
    misaligned = tilt_series.reconstruct_subvolumes_single(
        tilt_data=images,
        coords=reconstruction_location,
        pixel_size=pixel_size,
        size=patch_size,
        apply_ctf=apply_ctf,
        angles=rotation_angles,
        oversampling=2.0,
    ).squeeze()

    # normalize volumes before saving (do once here instead of every time we load)
    aligned = _normalize(aligned)
    misaligned = _normalize(misaligned)

    # choose third volume randomly (aligned or misaligned)
    if random.random() > 0.5:
        vol2, label2 = aligned, 1
    else:
        vol2, label2 = misaligned, -1

    # convert to fp16 numpy arrays for efficient storage
    return {
        'vol0': aligned.half().cpu().numpy(),
        'vol1': misaligned.half().cpu().numpy(),
        'vol2': vol2.half().cpu().numpy(),
        'labels': np.array([1, -1, label2], dtype=np.int8)
    }


def _generate_translations(
    generate_shifts: Callable,
    projection_matrices,
    device: str = "cpu",
) -> torch.Tensor:
    n_tilts, y, x = projection_matrices.shape
    if (y, x) != (2, 3):
        raise ValueError("Projection matrices must have shape (2, 3)")
    # generate two sets of shifts, 3d shifts zyx
    shifts = generate_shifts(n_tilts, device=device)
    shifts = project_shifts_3d_to_2d(shifts, projection_matrices)

    # randomly remove x or y shifts
    die_roll = random.random()
    if die_roll < 0.25:  # set y to 0
        shifts[:, 0] = 0.0
    elif 0.25 <= die_roll < 0.5:  # set x to 0
        shifts[:, 1] = 0.0

    return shifts

import multiprocessing as mp
import random
import tempfile
import math
import time
from collections.abc import Callable
from pathlib import Path
from typing import Optional
import torch
import pickle
import os
from torch_affine_utils.transforms_3d import Ry, Rz
from warpylib import TiltSeries
from warpylib.tilt_series.reconstruct_volume import preprocess_tilt_data
from dataclasses import dataclass

from miss_alignment.data.io import TiltSeriesData
from miss_alignment.data.shift_generation import project_shifts_3d_to_2d
from ._augmentation import MIRROR_COMBINATIONS, apply_mirror

# augmentation parameter
MAX_ANGLE_DEGREES = 10.0

# sequential ID wrap limit
MAX_SEQUENTIAL_ID = 1_000_000

# pause polling interval in seconds
PAUSE_POLL_INTERVAL = 0.05  # 50ms


@dataclass
class TiltSeriesFetcher:
    tilt_series_xmls: list[Path]
    patch_size: int  # needed for filter calculation in warpylib
    refresh_rate: int
    downsample: int
    device: str | torch.device
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
        tilt_series_data = TiltSeriesData(
            xml_metadata_path=random.choice(self.tilt_series_xmls)
        )
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


def _count_partition_files(pool_dir: Path, partition_id: int) -> int:
    """Count the number of files in a partition.

    Files are named: partition_{partition_id}_worker_{worker_id}_seq_{seq_id}.pickle
    """
    pattern = f"partition_{partition_id}_worker_*.pickle"
    return len(list(pool_dir.glob(pattern)))


def reconstruction_worker(
    partition_id: int,
    partition_size: int,
    pool_dir: Path,
    tilt_series_xmls: list[Path],
    patch_size: int,
    apply_ctf: bool,
    downsample: int,
    shift_generator: Callable,
    stop_event: mp.Event,
    tilt_series_refresh_rate: int = 10,
    device: str | torch.device = "cpu",
):
    """
    Worker process that maintains a partition of the reconstruction pool.

    Parameters
    ----------
    partition_id : int
        ID of this worker's partition
    partition_size : int
        Maximum number of files to maintain in this partition
    pool_dir : Path
        Directory to store reconstructions
    tilt_series_xmls : list[Path]
        List of paths to tilt-series metadata files to make
         reconstructions from
    patch_size : int
        Shape of patches will be: (patch_size, ) * 3
    apply_ctf : bool
        Do CTF correction during reconstruction
    downsample : int
        Downsampling factor for reconstructions
    shift_generator : Callable
        Function that generates random sets of shifts in 3D
    stop_event : mp.Event
        Event to signal worker shutdown
    tilt_series_refresh_rate: int, default 10
        Number of times a tilt-series is reused before loading a new one,
         prevents repeatedly loading from disk.
    device: str | torch.Device, default 'cpu'
        Device to use
    """
    torch.set_num_threads(1)

    print(
        f"Reconstruction worker {partition_id} "
        f"starting (partition size: {partition_size})"
    )

    # Initialize tilt series fetcher
    tilt_series_fetcher = TiltSeriesFetcher(
        tilt_series_xmls=tilt_series_xmls,
        patch_size=patch_size,
        refresh_rate=tilt_series_refresh_rate,
        downsample=downsample,
        device=device,
    )

    # Sequential ID counter for this partition
    sequential_id = 0

    # Continuously generate reconstructions
    while not stop_event.is_set():
        # Check if partition is full, pause if so
        while _count_partition_files(pool_dir, partition_id) >= partition_size:
            if stop_event.is_set():
                print(f"Reconstruction worker {partition_id} shutting down")
                return
            time.sleep(PAUSE_POLL_INTERVAL)

        # Generate reconstruction and 8 mirrored triplets
        tilt_series, images, pixel_size = tilt_series_fetcher()

        for i in range(2):
            triplets = _create_pool_reconstruction(
                tilt_series=tilt_series,
                images=images,
                pixel_size=pixel_size,
                patch_size=patch_size,
                shift_generator=shift_generator,
                apply_ctf=apply_ctf,
                device=device,
            )

            # Write each triplet to a separate file
            for triplet in triplets:
                # Convert to fp16 for storage
                triplet_fp16 = [(vol.half(), label) for vol, label in triplet]

                # Write with atomic rename
                file_path = (
                    pool_dir / f"partition_{partition_id}_seq_{sequential_id}.pickle"
                )
                with tempfile.NamedTemporaryFile(
                    dir=pool_dir,
                    prefix=f"tmp_partition_{partition_id}_",
                    suffix=".pickle",
                    delete=False,
                ) as tmp_file:
                    tmp_path = Path(tmp_file.name)
                    pickle.dump(triplet_fp16, tmp_file)

                os.chmod(tmp_path, 0o644)
                tmp_path.rename(file_path)

                # Increment and wrap sequential ID
                sequential_id = (sequential_id + 1) % MAX_SEQUENTIAL_ID

    print(f"Reconstruction worker {partition_id} shutting down")


def sample_positions(
    n_particles: int,
    volume_dimensions_angstrom: torch.Tensor,
    patch_size_angstrom: float,
    device: str | torch.device = "cpu",
) -> torch.Tensor:
    """
    Sample random reconstruction positions within a volume, handling edge cases.

    For dimensions larger than the patch size, positions are sampled uniformly
    within the valid range. For dimensions smaller than the patch size, positions
    are fixed at half the dimension size.

    Parameters
    ----------
    n_particles : int
        Number of positions to sample
    volume_dimensions_angstrom : torch.Tensor
        Physical volume dimensions in Angstroms, shape (3,) for ZYX
    patch_size_angstrom : float
        Physical patch size in Angstroms (cubic patches assumed)
    device : str | torch.device, default "cpu"
        Device to create the tensor on

    Returns
    -------
    torch.Tensor
        Sampled positions in Angstroms, shape (n_particles, 3) for ZYX
    """
    patch_offset_angstrom = patch_size_angstrom / 2

    # Sample positions uniformly in the valid range
    reconstruction_location = (
        torch.rand([n_particles, 3], device=device)
        * (volume_dimensions_angstrom - patch_size_angstrom)
        + patch_offset_angstrom
    )

    # Fix coordinates in dimensions smaller than the patch size
    # to just half that dimension's size
    for dim, dim_size in enumerate(volume_dimensions_angstrom):
        if dim_size < patch_size_angstrom:
            reconstruction_location[:, dim] = dim_size / 2

    return reconstruction_location


def _create_pool_reconstruction(
    tilt_series: TiltSeries,
    images: torch.Tensor,
    pixel_size: float,
    patch_size: int,
    shift_generator: Callable,
    apply_ctf: bool = True,
    device: str | torch.device = "cpu",
) -> list[list[tuple[torch.Tensor, int]]]:
    """
    Create 8 mirrored triplets from a single reconstruction.

    Returns
    -------
    list[list[tuple[torch.Tensor, int]]]
        List of 8 triplets, where each triplet is a list of 3 (volume, label) tuples.
        Positive and negative examples share the same mirror, anchor has a different
        mirror.
    """

    n_particles = 2

    # Generate random rotation as Euler angles (ZYZ convention) for data augmentation
    # This rotation is applied to change the coordinate system of the reconstruction
    rotation_angles = torch.zeros([n_particles, 3], device=device)
    rotation_angles[:, 1] = (
        (torch.rand([n_particles], device=device) * 2 - 1)
        * MAX_ANGLE_DEGREES
        * math.pi
        / 180
    )

    # Select random reconstruction positions
    patch_size_ang = patch_size * pixel_size
    reconstruction_location = sample_positions(
        n_particles=n_particles,
        volume_dimensions_angstrom=tilt_series.volume_dimensions_physical,
        patch_size_angstrom=patch_size_ang,
        device=device,
    )

    # generate a misalignment
    r0 = Ry(-tilt_series.angles, zyx=True)
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

    # Create anchor (random copy of aligned or misaligned)
    if random.random() > 0.5:
        anchor_base = aligned.clone()
        anchor_label = 1
    else:
        anchor_base = misaligned.clone()
        anchor_label = -1

    # Generate 2 triplets with all mirror combinations
    triplets = []
    for i_batch in range(n_particles):
        combinations_subset = random.sample(MIRROR_COMBINATIONS, 2)
        for i, mirror_combo in enumerate(combinations_subset):
            # Apply same mirror to positive and negative
            mirrored_aligned = apply_mirror(
                aligned[i_batch].clone(), mirror_combo
            ).cpu()
            mirrored_misaligned = apply_mirror(
                misaligned[i_batch].clone(), mirror_combo
            ).cpu()

            # Pick a different mirror for anchor
            other_combos = [c for j, c in enumerate(MIRROR_COMBINATIONS) if j != i]
            anchor_mirror = random.choice(other_combos)
            mirrored_anchor = apply_mirror(
                anchor_base[i_batch].clone(), anchor_mirror
            ).cpu()

            triplet = [
                (mirrored_aligned, 1),
                (mirrored_misaligned, -1),
                (mirrored_anchor, anchor_label),
            ]
            triplets.append(triplet)

    return triplets


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

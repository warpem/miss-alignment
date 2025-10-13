import random
import torch
import einops
import warnings
from torch_cubic_spline_grids import CubicBSplineGrid1d
from dataclasses import dataclass
from typing import List, Callable, Optional


@dataclass
class ShiftConfig:
    """Configuration for a single shift type."""
    name: str
    probability: float
    generator: Callable[[int, str], torch.Tensor]

    def should_apply(self) -> bool:
        """Check if this shift should be applied based on probability."""
        return self.probability > 0.0 and random.random() <= self.probability


class ShiftGenerator:
    """Composable 3D shift generator."""

    def __init__(self, shift_configs: List[ShiftConfig],
                 ensure_at_least_one: bool = True):
        """
        Initialize the shift generator.

        Parameters
        ----------
        shift_configs : List[ShiftConfig]
            List of shift configurations to apply
        ensure_at_least_one : bool
            If True, ensures at least one shift is applied when possible
        """
        self.shift_configs = shift_configs
        self.ensure_at_least_one = ensure_at_least_one

        for config in self.shift_configs:
            if config.probability == 0.0:
                warnings.warn(f"ShiftConfig {config.name} has probability 0.0")

    def __call__(self, num_points: int, device: str = 'cpu') -> torch.Tensor:
        """Generate combined shifts for the given number of points."""
        shifts = torch.zeros((num_points, 3), device=device)

        # Determine which shifts to apply
        shifts_to_apply = [config for config in self.shift_configs if
                           config.should_apply()]

        # If no shifts selected and we need at least one, pick from available configs
        if not shifts_to_apply and self.ensure_at_least_one:
            available_configs = [config for config in self.shift_configs if
                                 config.probability > 0.0]
            if available_configs:
                selected_config = random.choice(available_configs)
                shifts_to_apply = [selected_config]

        # Apply selected shifts
        for config in shifts_to_apply:
            shift_contribution = config.generator(num_points, device)
            shifts = shifts + shift_contribution

        return shifts


def select_random_indices(
    sequence,
    max_sequence_length: int = 5,
    edge_only: bool = False,
):
    """
    Randomly select 1 to 4 connected indices from a sequence.

    Parameters
    ----------
    sequence : torch.Tensor
        A 1D tensor from which to select indices

    Returns
    -------
    torch.Tensor
        A 1D tensor containing the selected indices
    """
    if not isinstance(sequence, torch.Tensor) or sequence.dim() != 1:
        raise ValueError("Input must be a 1D torch tensor")

    seq_length = len(sequence)
    if seq_length == 0:
        return torch.tensor([], dtype=torch.long)

    # Decide how many indices to select (1-4)
    num_indices = min(random.randint(1, max_sequence_length), seq_length)

    # Select a starting index
    max_start = seq_length - num_indices
    if edge_only:
        if random.random() > 0.5:  # only do outliers on the last 0 to 5 tilts
            start_idx = 0
        else:
            start_idx = max_start
    else:
        start_idx = random.randint(0, max(0, max_start))

    # Create the connected indices
    selected_indices = torch.arange(
        start_idx, start_idx + num_indices, dtype=torch.long
    )

    return selected_indices


def generate_smooth_trajectory(
    grid_resolution: int = 3, device: str = 'cpu',
) -> tuple[CubicBSplineGrid1d, CubicBSplineGrid1d, CubicBSplineGrid1d]:
    """Generate a smooth trajectory using cubic B-spline interpolation.

    Parameters
    ----------
    grid_resolution : int, optional
        Resolution of the grid used for generating smooth trajectories, by default 3.
    device : str, optional

    Returns
    -------
    tuple
        A tuple with three CubicSplineGrid1d objects for x/y/z trajectories.
    """
    # Create three 1D spline grids for x, y, and z coordinates
    grid_x = CubicBSplineGrid1d(resolution=grid_resolution)
    grid_y = CubicBSplineGrid1d(resolution=grid_resolution)
    grid_z = CubicBSplineGrid1d(resolution=grid_resolution)
    grid_x.to(device=device)
    grid_y.to(device=device)
    grid_z.to(device=device)

    # Initialize grid data with random values to create a parabola-like curve
    grid_x.data = torch.rand(3, device=device) * 2 - 1
    grid_y.data = torch.rand(3, device=device) * 2 - 1
    grid_z.data = torch.rand(3, device=device) * 2 - 1

    return grid_x, grid_y, grid_z


def generate_shift_trajectory(
        num_points: int,
        grid_resolution: int = 3,
        breakpoint_p: float = 0.5,
        device: str = 'cpu',
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Generate a trajectory with an optional shift in direction.

    This function creates a smooth trajectory in 3D space. With probability
    `breakpoint_p`, it will generate a trajectory with a sudden change in
    direction by stitching together two smooth sub-trajectories.

    Parameters
    ----------
    num_points : int
        Number of points in the final trajectory.
    grid_resolution : int, optional
        Resolution of the grid used for generating smooth trajectories, by default 3.
    breakpoint_p : float, optional
        Probability of generating a trajectory with a breakpoint, by default 0.5.
    device : str, optional

    Returns
    -------
    tuple
        A tuple containing:
            - timesteps : torch.Tensor
                Linear space from 0 to 1 with `num_points` elements.
            - x_positions : torch.Tensor
                X-coordinates of the trajectory points.
            - y_positions : torch.Tensor
                Y-coordinates of the trajectory points.
            - z_positions : torch.Tensor
                Z-coordinates of the trajectory points.
    """
    if num_points < 4:
        raise ValueError(f"num_points must be at least 4, got {num_points}")

    if random.random() > 1.0 - breakpoint_p:
        grid_x1, grid_y1, grid_z1 = (
            generate_smooth_trajectory(grid_resolution, device=device)
        )
        grid_x2, grid_y2, grid_z2 = (
            generate_smooth_trajectory(grid_resolution, device=device)
        )
        grid_x2.data[..., 0] = grid_x1.data[
            ..., grid_resolution - 1
        ]  # stitch grids together
        grid_y2.data[..., 0] = grid_y1.data[..., grid_resolution - 1]
        grid_z2.data[..., 0] = grid_z1.data[..., grid_resolution - 1]
        t1 = random.randint(2, num_points - 2)
        t2 = num_points - t1
        t1 = torch.linspace(0, 1, t1, device=device)
        t2 = torch.linspace(0, 1, t2 + 1, device=device)[1:]
        x_pos1, y_pos1, z_pos1 = grid_x1(t1), grid_y1(t1), grid_z1(t1)
        x_pos2, y_pos2, z_pos2 = grid_x2(t2), grid_y2(t2), grid_z2(t2)
        x_positions = torch.cat([x_pos1, x_pos2])
        y_positions = torch.cat([y_pos1, y_pos2])
        z_positions = torch.cat([z_pos1, z_pos2])
        timesteps = torch.linspace(0, 1, num_points, device=device)
    else:
        grid_x, grid_y, grid_z = (
            generate_smooth_trajectory(grid_resolution, device=device)
        )
        timesteps = torch.linspace(0, 1, num_points, device=device)
        x_positions = grid_x(timesteps)
        y_positions = grid_y(timesteps)
        z_positions = grid_z(timesteps)
    x_positions = x_positions.detach()
    y_positions = y_positions.detach()
    z_positions = z_positions.detach()
    return timesteps, x_positions, y_positions, z_positions


class TrajectoryGenerator:
    """Picklable trajectory shift generator."""

    def __init__(self, trajectory_max_shift: float):
        self.trajectory_max_shift = trajectory_max_shift

    def __call__(self, num_points: int, device: str) -> torch.Tensor:
        _, x_shifts, y_shifts, z_shifts = (
            generate_shift_trajectory(num_points, device=device)
        )
        x_shifts = x_shifts * self.trajectory_max_shift
        y_shifts = y_shifts * self.trajectory_max_shift
        z_shifts = z_shifts * self.trajectory_max_shift
        shifts = torch.stack([z_shifts, y_shifts, x_shifts], dim=1)
        shifts -= einops.reduce(shifts, "n zyx -> 1 zyx",
                                        reduction="mean")
        return shifts


class JitterGenerator:
    """Picklable jitter shift generator."""

    def __init__(self, jitter_max_std: float):
        self.jitter_max_std = jitter_max_std

    def __call__(self, num_points: int, device: str) -> torch.Tensor:
        std = random.random() * self.jitter_max_std
        return torch.normal(mean=0.0, std=float(std), size=(num_points, 3),
                            device=device)


class OutlierGenerator:
    """Picklable outlier shift generator."""

    def __init__(self, outlier_max_shift: float):
        self.outlier_max_shift = outlier_max_shift

    def __call__(self, num_points: int, device: str) -> torch.Tensor:
        shifts = torch.zeros((num_points, 3), device=device)
        ids = select_random_indices(
            torch.arange(num_points),
            max_sequence_length=1,
            edge_only=False,
        )
        outliers = torch.rand(3, device=device) * (
                    2 * self.outlier_max_shift) - self.outlier_max_shift
        shifts[ids] = outliers
        return shifts


class FractureGenerator:
    """Picklable fracture shift generator."""
    def __init__(self, fracture_max_shift: float):
        self.fracture_max_shift = fracture_max_shift

    def __call__(self, num_points: int, device: str) -> torch.Tensor:
        shifts = torch.zeros((num_points, 3), device=device)
        fraction_idx = random.randint(1, num_points)
        outliers = torch.rand(3, device=device) * (
                2 * self.fracture_max_shift) - self.fracture_max_shift
        if random.random() > 0.5:  # make it linear
            z = torch.linspace(outliers[0], 0, fraction_idx, device=device)
            y = torch.linspace(outliers[1], 0, fraction_idx, device=device)
            x = torch.linspace(outliers[2], 0, fraction_idx, device=device)
            outliers = torch.stack([z, y, x], dim=1)
        # randomly invert direction
        if random.random() > 0.5:
            shifts[- fraction_idx:] = torch.flip(outliers, dims=(0,))
        else:
            shifts[:fraction_idx] = outliers
        shifts -= einops.reduce(shifts, "n zyx -> 1 zyx",
                                        reduction="mean")
        return shifts


def create_default_generator(
        trajectory_probability: float = 0.5,
        trajectory_max_shift: float = 10.0,
        jitter_probability: float = 0.5,
        jitter_max_std: float = 2.0,
        outlier_probability: float = 0.4,
        outlier_max_shift: float = 20.0,
        fracture_probability: float = 0.4,
        fracture_max_shift: float = 30.0,
) -> ShiftGenerator:
    """Create a picklable shift generator."""

    shift_configs = [
        ShiftConfig(
            name="trajectory",
            probability=trajectory_probability,
            generator=TrajectoryGenerator(trajectory_max_shift)
        ),
        ShiftConfig(
            name="jitter",
            probability=jitter_probability,
            generator=JitterGenerator(jitter_max_std)
        ),
        ShiftConfig(
            name="outlier",
            probability=outlier_probability,
            generator=OutlierGenerator(outlier_max_shift)
        ),
        ShiftConfig(
            name="fracture",
            probability=fracture_probability,
            generator=FractureGenerator(fracture_max_shift)
        ),
    ]

    return ShiftGenerator(shift_configs, ensure_at_least_one=True)


def project_shifts_3d_to_2d(
    shifts_3d: torch.Tensor,  # contains (n, zyx) shifts
    projection_matrices: torch.Tensor,  # contains (n, 2, 3)
) -> torch.Tensor:
    """Project 3D shifts to 2D."""
    y, x = projection_matrices.shape[-2:]
    if (y, x) != (2, 3):
        raise ValueError('Shift projection matrices must have shape (2, 3)')
    shifts_3d = einops.rearrange(shifts_3d, "b zyx -> b zyx 1")
    shifts_2d = projection_matrices @ shifts_3d
    shifts_2d = einops.rearrange(shifts_2d, "b yx 1 -> b yx")
    return shifts_2d

import random
import torch
import einops
from torch_cubic_spline_grids import CubicBSplineGrid1d


def select_random_indices(sequence):
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
    num_indices = min(random.randint(1, 5), seq_length)

    # Select a starting index
    max_start = seq_length - num_indices
    # if random.random() > .5:  # only do outliers on the last 0 to 5 tilts
    #     start_idx = 0
    # else:
    #     start_idx = max_start
    start_idx = random.randint(0, max(0, max_start))

    # Create the connected indices
    selected_indices = torch.arange(
        start_idx, start_idx + num_indices, dtype=torch.long
    )

    return selected_indices


def generate_smooth_trajectory(
    grid_resolution: int = 3,
) -> tuple[CubicBSplineGrid1d, CubicBSplineGrid1d, CubicBSplineGrid1d]:
    """Generate a smooth trajectory using cubic B-spline interpolation.

    Parameters
    ----------
    grid_resolution : int, optional
        Resolution of the grid used for generating smooth trajectories, by default 3.

    Returns
    -------
    tuple
        A tuple with three CubicSplineGrid1d objects for x/y/z trajectories.
    """
    # Create three 1D spline grids for x, y, and z coordinates
    grid_x = CubicBSplineGrid1d(resolution=grid_resolution)
    grid_y = CubicBSplineGrid1d(resolution=grid_resolution)
    grid_z = CubicBSplineGrid1d(resolution=grid_resolution)

    # Initialize grid data with random values to create a parabola-like curve
    x_values = torch.rand(3) * 2 - 1
    y_values = torch.rand(3) * 2 - 1
    z_values = torch.rand(3) * 2 - 1

    grid_x.data = x_values
    grid_y.data = y_values
    grid_z.data = z_values

    return grid_x, grid_y, grid_z


def generate_shift_trajectory(
    num_points: int,
    grid_resolution: int = 3,
    breakpoint_p: float = 0.5,
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
    if random.random() > 1.0 - breakpoint_p:
        grid_x1, grid_y1, grid_z1 = generate_smooth_trajectory(grid_resolution)
        grid_x2, grid_y2, grid_z2 = generate_smooth_trajectory(grid_resolution)
        grid_x2.data[..., 0] = grid_x1.data[
            ..., grid_resolution - 1
        ]  # stitch grids together
        grid_y2.data[..., 0] = grid_y1.data[..., grid_resolution - 1]
        grid_z2.data[..., 0] = grid_z1.data[..., grid_resolution - 1]
        t1 = random.randint(2, num_points - 2)
        t2 = num_points - t1
        t1 = torch.linspace(0, 1, t1)
        t2 = torch.linspace(0, 1, t2 + 1)[1:]
        x_pos1, y_pos1, z_pos1 = grid_x1(t1), grid_y1(t1), grid_z1(t1)
        x_pos2, y_pos2, z_pos2 = grid_x2(t2), grid_y2(t2), grid_z2(t2)
        x_positions = torch.cat([x_pos1, x_pos2])
        y_positions = torch.cat([y_pos1, y_pos2])
        z_positions = torch.cat([z_pos1, z_pos2])
        timesteps = torch.linspace(0, 1, num_points)
    else:
        grid_x, grid_y, grid_z = generate_smooth_trajectory(grid_resolution)
        timesteps = torch.linspace(0, 1, num_points)
        x_positions = grid_x(timesteps)
        y_positions = grid_y(timesteps)
        z_positions = grid_z(timesteps)
    x_positions = x_positions.detach()
    y_positions = y_positions.detach()
    z_positions = z_positions.detach()
    return timesteps, x_positions, y_positions, z_positions


def generate_shifts(
    num_points: int,
    max_shift: float,
    trajectory_probability: float = 0.5,
    jitter_probability: float = 0.5,
    outlier_probability: float = 0.5,
) -> torch.Tensor:
    """Generate pairs of aligned and misaligned shifts for trajectory simulation.

    Ensures at least one type of shift (trajectory, jitter, or outlier) is applied.

    Parameters
    ----------
    num_points : int
        Number of points to generate.
    max_shift : float
        Maximum shift magnitude allowed.
    trajectory_probability : float, optional
        Probability of generating trajectory-based shifts, default 0.5.
    jitter_probability : float, optional
        Probability of adding jitter shifts, default 0.5.
    outlier_probability : float, optional
        Probability of adding outlier shifts, default 0.5.

    Returns
    -------
    torch.Tensor
        Shifts tensor of shape (num_points, 3).
    """
    shifts = torch.zeros((num_points, 3))

    # Determine which shifts to apply
    apply_trajectory = random.random() <= trajectory_probability
    apply_jitter = random.random() <= jitter_probability
    apply_outlier = random.random() <= outlier_probability

    # If none selected, choose one randomly
    if not (apply_trajectory or apply_jitter or apply_outlier):
        # Choose one shift type randomly
        shift_type = random.randint(0, 2)
        if shift_type == 0:
            apply_trajectory = True
        elif shift_type == 1:
            apply_jitter = True
        else:  # shift_type == 2
            apply_outlier = True

    ##### Part 1: Trajectories
    if apply_trajectory:
        _, x_shifts, y_shifts, z_shifts = generate_shift_trajectory(num_points)
        x_shifts = x_shifts * (max_shift * 0.2)
        y_shifts = y_shifts * (max_shift * 0.2)
        z_shifts = z_shifts * (max_shift * 0.2)
        shifts = shifts + torch.stack([z_shifts, y_shifts, x_shifts], dim=1)

        # substract the mean to center the trajectories to the origin in 3D
        shifts = shifts - einops.reduce(shifts, "n zyx -> 1 zyx", reduction="mean")

    ##### Part 2: Jitter
    if apply_jitter:
        std = random.random() * (max_shift * 0.1)
        shifts += torch.normal(
            mean=0.0,
            std=float(std),
            size=(num_points, 3),
        )

    ##### Part 3: Outliers
    if apply_outlier:
        ids = select_random_indices(torch.arange(num_points))
        outliers = torch.rand(3) * (2 * max_shift) - max_shift
        shifts[ids] = shifts[ids] + outliers

    return shifts


def project_shifts_3d_to_2d(
    shifts_3d: torch.Tensor,  # contains (n, zyx) shifts
    rotation_matrices: torch.Tensor,  # contains (n, 3, 3) but works on
    # xyz coordinates, hence some weird swapping inside
) -> torch.Tensor:
    """Project 3D shifts to 2D."""
    shifts_3d = torch.flip(shifts_3d, dims=(1,))  # (n, zyx) -> (n, xyz)
    shifts_3d = einops.rearrange(shifts_3d, "b xyz -> b xyz 1")
    shifts_3d = rotation_matrices @ shifts_3d
    shifts_3d = einops.rearrange(shifts_3d, "b xyz 1 -> b xyz")
    shifts_2d = torch.flip(  # remove z and swap to yx
        shifts_3d[:, :2], dims=(1,)
    )  # (n, xyz) -> (n, yx)
    return shifts_2d

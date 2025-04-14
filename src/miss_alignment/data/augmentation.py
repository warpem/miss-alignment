import random
import torch
from torch_cubic_spline_grids import CubicBSplineGrid1d


def random_contrast(*images):
    std_change, mean_change = (
        torch.normal(1, 0.1, (1,)), torch.normal(0, 0.1, (1,))
    )
    return [i * std_change + mean_change for i in images]


def random_cube_mask(
        *volumes: torch.Tensor, p=.3, size_range=(0.1, 0.3)
) -> list[torch.Tensor]:
    """Apply the same random cube mask to multiple volumes.

    Args:
        *volumes: 3D volumes of shape [D, H, W] without channel dimension
        p: Probability of applying the mask
        size_range: Range for the size of the cube as a fraction of the volume dimensions

    Returns:
        List of masked volumes
    """
    if random.random() > 1 - p:
        # Get the dimensions of the first volume (assuming all have same dimensions)
        d, h, w = volumes[0].shape

        # Determine mask size as fraction of volume dimensions
        mask_fraction = random.uniform(size_range[0], size_range[1])
        mask_d = max(1, int(d * mask_fraction))
        mask_h = max(1, int(h * mask_fraction))
        mask_w = max(1, int(w * mask_fraction))

        # Random starting positions for the mask
        start_d = random.randint(0, d - mask_d)
        start_h = random.randint(0, h - mask_h)
        start_w = random.randint(0, w - mask_w)

        # Apply the same mask to all volumes
        masked_volumes = []
        for volume in volumes:
            masked = volume.clone()
            masked[
                start_d:start_d + mask_d,
                start_h:start_h + mask_h,
                start_w:start_w + mask_w
            ] = random.random() - .5
            masked_volumes.append(masked)

        return masked_volumes

    return list(volumes)


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
    num_indices = min(random.randint(1, 6), seq_length)

    # Select a starting index
    max_start = seq_length - num_indices
    start_idx = random.randint(0, max(0, max_start))

    # Create the connected indices
    selected_indices = torch.arange(start_idx, start_idx + num_indices,
                                    dtype=torch.long)

    return selected_indices


def generate_smooth_trajectory(
        grid_resolution: int = 3
) -> tuple[CubicBSplineGrid1d, CubicBSplineGrid1d]:
    """Generate a smooth trajectory using cubic B-spline interpolation.

    Parameters
    ----------
    grid_resolution : int, optional
        Resolution of the grid used for generating smooth trajectories, by default 3.

    Returns
    -------
    tuple
        A tuple with two CubicSplineGrid1d object for x/y trajectories.
    """
    # Create two 1D spline grids for x and y coordinates
    grid_x = CubicBSplineGrid1d(resolution=grid_resolution)
    grid_y = CubicBSplineGrid1d(resolution=grid_resolution)

    # Initialize grid data with random values to create a parabola-like curve
    x_values = torch.rand(3) * 2 - 1
    y_values = torch.rand(3) * 2 - 1

    grid_x.data = x_values
    grid_y.data = y_values

    return grid_x, grid_y


def generate_shift_trajectory(
        num_points: int,
        grid_resolution: int = 3,
        breakpoint_p: float = .5,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Generate a trajectory with an optional shift in direction.

    This function creates a smooth trajectory in 2D space. With probability
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
    """
    if random.random() > 1. - breakpoint_p:
        grid_x1, grid_y1 = generate_smooth_trajectory(grid_resolution)
        grid_x2, grid_y2 = generate_smooth_trajectory(grid_resolution)
        grid_x2.data[..., 0] = grid_x1.data[
            ..., grid_resolution - 1]  # stitch grids together
        grid_y2.data[..., 0] = grid_y1.data[..., grid_resolution - 1]
        t1 = random.randint(2, num_points - 2)
        t2 = num_points - t1
        t1 = torch.linspace(0, 1, t1)
        t2 = torch.linspace(0, 1, t2 + 1)[1:]
        x_pos1, y_pos1 = grid_x1(t1), grid_y1(t1)
        x_pos2, y_pos2 = grid_x2(t2), grid_y2(t2)
        x_positions = torch.cat([x_pos1, x_pos2])
        y_positions = torch.cat([y_pos1, y_pos2])
        timesteps = torch.linspace(0, 1, num_points)
    else:
        grid_x, grid_y = generate_smooth_trajectory(grid_resolution)
        timesteps = torch.linspace(0, 1, num_points)
        x_positions = grid_x(timesteps)
        y_positions = grid_y(timesteps)
    x_positions = x_positions.detach()
    y_positions = y_positions.detach()
    return timesteps, x_positions, y_positions


def generate_aligned_and_misaligned_shifts(
        num_points: int,
        max_shift: float,
        trajectory_probability: float = .5,
        outlier_probability: float = .5,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Generate pairs of aligned and misaligned shifts for trajectory simulation.

    Parameters
    ----------
    num_points : int
       Number of points to generate.
    max_shift : float
       Maximum shift magnitude allowed.
    trajectory_probability : float, optional
       Probability of generating trajectory-based shifts, default 0.5.
    outlier_probability : float, optional
       Probability of adding outlier shifts, default 0.5.

    Returns
    -------
    tuple[torch.Tensor, torch.Tensor]
       Aligned and misaligned shift tensors, each of shape (num_points, 2).
    """
    ##### Part 1: Trajectories
    if random.random() > 1. - trajectory_probability:
        _, x_shifts, y_shifts = generate_shift_trajectory(num_points)
        x_shifts = x_shifts * (max_shift * .2)
        shifts = torch.stack([y_shifts, x_shifts], dim=1)
    else:
        shifts = torch.zeros((num_points, 2))
    misaligned = shifts
    # trajectories have two situations:
    # * applied to misaligned and a softer version to aligned
    # * applied to misaligned and not to aligned
    if random.random() > .5:  # reduce by factor from 0. to 1.
        aligned = misaligned * random.random()
    else:
        aligned = torch.zeros((num_points, 2))

    ##### Part 2: Jitter
    # between 0 and fraction of max_shift
    misaligned_std = random.random() * (max_shift * .15)
    aligned_std = random.random() * misaligned_std  # between 0 and misaligned
    if random.random() > .5:  # 50% chance of no jitter
        aligned = aligned + torch.normal(
            mean=0.0,
            std=float(aligned_std),
            size=(num_points, 2),  # batch of number of tilts
        )
    misaligned = misaligned + torch.normal(  # dont always apply
        mean=0.0,
        std=float(misaligned_std),
        size=(num_points, 2),  # batch of number of tilts
    )

    ##### Part 3: Outliers
    if random.random() > 1. - outlier_probability:
        ids = select_random_indices(torch.arange(num_points))
        # if input is 10A, this is a maximum shift of 150A
        outliers = torch.rand((len(ids), 2)) * (2 * max_shift) - max_shift
        misaligned[ids] = outliers
        if random.random() > .5:
            aligned[ids] = outliers * random.random()

    return aligned, misaligned



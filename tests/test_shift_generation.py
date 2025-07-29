import torch
import pytest
from miss_alignment.data.shift_generation import (
    generate_smooth_trajectory,
    generate_shift_trajectory,
    generate_shifts
)


def test_generate_smooth_trajectory():
    """Test that generate_smooth_trajectory returns 3D trajectories."""
    grid_x, grid_y, grid_z = generate_smooth_trajectory()

    # Check that all three grids are returned
    assert grid_x is not None
    assert grid_y is not None
    assert grid_z is not None


def test_generate_shift_trajectory():
    """Test that generate_shift_trajectory returns 3D trajectories."""
    num_points = 10
    timesteps, x_positions, y_positions, z_positions = generate_shift_trajectory(num_points)

    # Check shapes
    assert timesteps.shape == (num_points,)
    assert x_positions.shape == (num_points,)
    assert y_positions.shape == (num_points,)
    assert z_positions.shape == (num_points,)


def test_generate_shifts():
    """Test that generate_shifts returns 3D shifts."""
    num_points = 10
    max_shift = 1.0

    # Test with different combinations of probabilities
    shifts1 = generate_shifts(
        num_points, 
        trajectory_probability=1.0, 
        trajectory_max_shift=max_shift,
        jitter_probability=0.0, 
        outlier_probability=0.0,
        high_tilt_outlier_probability=0.0
    )  # Only trajectory

    shifts2 = generate_shifts(
        num_points, 
        trajectory_probability=0.0, 
        jitter_probability=1.0,
        jitter_max_std=max_shift,
        outlier_probability=0.0,
        high_tilt_outlier_probability=0.0
    )  # Only jitter

    shifts3 = generate_shifts(
        num_points, 
        trajectory_probability=0.0, 
        jitter_probability=0.0,
        outlier_probability=1.0,
        outlier_max_shift=max_shift,
        high_tilt_outlier_probability=0.0
    )  # Only outliers

    shifts4 = generate_shifts(
        num_points, 
        trajectory_probability=1.0,
        trajectory_max_shift=max_shift,
        jitter_probability=1.0,
        jitter_max_std=max_shift,
        outlier_probability=1.0,
        outlier_max_shift=max_shift,
        high_tilt_outlier_probability=1.0,
        high_tilt_max_shift=max_shift
    )  # All types

    # Check shapes
    assert shifts1.shape == (num_points, 3)
    assert shifts2.shape == (num_points, 3)
    assert shifts3.shape == (num_points, 3)
    assert shifts4.shape == (num_points, 3)

    # Check that shifts are within expected range
    # For trajectory-only shifts
    assert torch.all(shifts1 <= max_shift)
    assert torch.all(shifts1 >= -max_shift)

    # Jitter sampled from normal distribution so can be very large
    assert torch.all(shifts2 <= max_shift * 10)  # Allow for 3 standard
    # deviations
    assert torch.all(shifts2 >= -max_shift * 10)

    # For outlier-only shifts
    assert torch.all(shifts3 <= max_shift)
    assert torch.all(shifts3 >= -max_shift)

    # For shifts with all types, we don't check exact bounds as they can be larger
    # due to the combination of different shift types

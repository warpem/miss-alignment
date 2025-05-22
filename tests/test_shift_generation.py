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
    shifts1 = generate_shifts(num_points, max_shift, 1.0, 0.0, 0.0)  # Only trajectory
    shifts2 = generate_shifts(num_points, max_shift, 0.0, 1.0, 0.0)  # Only jitter
    shifts3 = generate_shifts(num_points, max_shift, 0.0, 0.0, 1.0)  # Only outliers
    shifts4 = generate_shifts(num_points, max_shift, 1.0, 1.0, 1.0)  # All types
    
    # Check shapes
    assert shifts1.shape == (num_points, 3)
    assert shifts2.shape == (num_points, 3)
    assert shifts3.shape == (num_points, 3)
    assert shifts4.shape == (num_points, 3)
    
    # Check that shifts are within expected range
    assert torch.all(shifts1 <= max_shift)
    assert torch.all(shifts1 >= -max_shift)
    assert torch.all(shifts2 <= max_shift)
    assert torch.all(shifts2 >= -max_shift)
    assert torch.all(shifts3 <= max_shift)
    assert torch.all(shifts3 >= -max_shift)
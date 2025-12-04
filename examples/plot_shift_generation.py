import torch
import matplotlib.pyplot as plt
from torch_affine_utils.transforms_3d import Ry
from miss_alignment.data.shift_generation import (
    create_default_generator, project_shifts_3d_to_2d
)
from typing import Callable

N_TILTS = 41


def plot_projected_shifts_2d(shift_generator: Callable[[int], torch.Tensor],
                             title_prefix: str = "Projected Shifts"):
    """
    Plot projected 2D shifts from 3D shifts using rotation matrices.

    Creates two subplots: one for x-shifts and one for y-shifts across different
    rotation angles from -54 to 54 degrees.

    Parameters
    ----------
    shifts_3d : torch.Tensor
        3D shifts tensor of shape (num_points, 3) in ZYX order
    title_prefix : str, optional
        Prefix for the plot title, by default "Projected Shifts"
    """
    # Generate rotation matrices and projection matrices
    angles = torch.linspace(-54, 54, N_TILTS)
    rotation_matrices = Ry(angles, zyx=True)
    projection_matrices = rotation_matrices[..., 1:3, :3]  # Shape: (41, 2, 3)
    shifts_3d = shift_generator(N_TILTS)
    shifts_2d = project_shifts_3d_to_2d(shifts_3d, projection_matrices)

    # Extract x and y shifts
    x_shifts = shifts_2d[..., 0]  # Shape: (41, num_points)
    y_shifts = shifts_2d[..., 1]  # Shape: (41, num_points)

    # Create figure with subplots for different rotation angles
    fig, ax = plt.subplots()

    # Plot trajectory
    ax.plot(x_shifts, y_shifts, 'b-', linewidth=2, alpha=0.7,
                 label='Trajectory')
    ax.scatter(x_shifts, y_shifts, c=range(N_TILTS),
                    cmap='viridis',
                    s=30, zorder=5, alpha=0.8)

    # Mark start and end points
    ax.scatter(x_shifts[0], y_shifts[0], c='green', s=100, marker='o',
                    label='Start', zorder=10)
    ax.scatter(x_shifts[-1], y_shifts[-1], c='red', s=100, marker='s',
                    label='End', zorder=10)

    ax.set_xlabel('X-shift')
    ax.set_ylabel('Y-shift')
    ax.grid(True, alpha=0.3)
    ax.legend()
    ax.set_aspect('equal', adjustable='box')

    plt.title(title_prefix, fontsize=16)
    plt.tight_layout()
    plt.show()

    return fig, ax


def demo_shift_plotting():
    """
    Demonstrate the shift plotting function with different types of shifts.
    """
    num_points = 20

    # Create different types of generators
    generators = {
        "Trajectory Only": create_default_generator(
            trajectory_probability=1.0,
            jitter_probability=0.0,
            outlier_probability=0.0,
            high_tilt_outlier_probability=0.0
        ),
        "Jitter Only": create_default_generator(
            trajectory_probability=0.0,
            jitter_probability=1.0,
            outlier_probability=0.0,
            high_tilt_outlier_probability=0.0
        ),
        "Outliers Only": create_default_generator(
            trajectory_probability=0.0,
            jitter_probability=0.0,
            outlier_probability=1.0,
            high_tilt_outlier_probability=0.0
        ),
        "Combined Shifts": create_default_generator(
            trajectory_probability=0.8,
            jitter_probability=0.6,
            outlier_probability=0.4,
            high_tilt_outlier_probability=0.3
        )
    }

    # Generate and plot each type
    for name, generator in generators.items():
        shifts_3d = generator(num_points)
        print(f"\nPlotting {name}:")
        print(f"3D shifts shape: {shifts_3d.shape}")
        print(
            f"3D shifts range: [{shifts_3d.min():.2f}, {shifts_3d.max():.2f}]")

        plot_projected_shifts_2d(shifts_3d, title_prefix=name)


# Example usage
if __name__ == "__main__":
    # Quick example
    generator = create_default_generator()
    plot_projected_shifts_2d(generator)

    # Or run the full demo
    # demo_shift_plotting()